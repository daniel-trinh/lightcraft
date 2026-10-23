//! Imported profiles: `.cube` 3D LUTs used as creative profiles (Amount 0–200 % like the
//! built-in ones). A library on disk keeps a copy of each file in its `Profiles/` folder; the
//! list lives in the library's prefs and the LUTs are registered with the pipeline when the
//! library opens.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, str_param};
use crate::{Result, Session};

/// One imported LUT profile.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LutProfile {
    /// `lut:<slug>`.
    pub id: String,
    pub name: String,
    pub group: String,
    /// The `.cube` file (the library's copy when there is a library on disk).
    pub file: String,
}

fn profiles_dir(s: &Session) -> Option<PathBuf> {
    s.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.join("Profiles"))
}

/// Register every imported profile's LUT with the pipeline (missing files are skipped).
pub fn register_all(s: &Session) {
    for p in &s.lut_profiles {
        if let Some(l) = std::fs::read_to_string(&p.file).ok().and_then(|t| lightcraft_pipeline::lut::Lut3d::parse_cube(&t).ok()) {
            lightcraft_pipeline::lut::register(&p.id, l);
        }
    }
}

/// `.cube` files in `paths` (folders searched, zip bundles opened).
fn cube_files(paths: &[String]) -> Vec<(String, String, Option<String>)> {
    // (name, text, group)
    let mut out = Vec::new();
    for p in paths {
        let path = Path::new(p);
        if path.is_dir() {
            let mut stack = vec![path.to_path_buf()];
            while let Some(d) = stack.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else { continue };
                for e in rd.flatten() {
                    let ep = e.path();
                    if ep.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
                        continue;
                    }
                    if ep.is_dir() {
                        stack.push(ep);
                    } else if ep.extension().is_some_and(|x| x.eq_ignore_ascii_case("cube"))
                        && let Ok(t) = std::fs::read_to_string(&ep)
                    {
                        let group = ep.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().to_string());
                        out.push((ep.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), t, group));
                    }
                }
            }
        } else if path.extension().is_some_and(|x| x.eq_ignore_ascii_case("zip")) {
            let Ok(b) = std::fs::read(path) else { continue };
            for (inner, data) in crate::preset_import::read_zip(&b).unwrap_or_default() {
                if inner.to_ascii_lowercase().ends_with(".cube") {
                    let ip = Path::new(&inner);
                    let group = ip
                        .parent()
                        .and_then(|d| d.file_name())
                        .map(|n| n.to_string_lossy().to_string())
                        .or_else(|| path.file_stem().map(|n| n.to_string_lossy().to_string()));
                    out.push((
                        ip.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                        String::from_utf8_lossy(&data).to_string(),
                        group,
                    ));
                }
            }
        } else if let Ok(t) = std::fs::read_to_string(path) {
            out.push((path.file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), t, None));
        }
    }
    out
}

fn rollback_import(s: &mut Session, old_len: usize) {
    for profile in s.lut_profiles.drain(old_len..) {
        lightcraft_pipeline::lut::unregister(&profile.id);
    }
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "profile.import";
    let paths: Vec<String> =
        p.get("paths").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    if paths.is_empty() {
        return Err(bad(C, "no paths"));
    }
    let dir = profiles_dir(s);
    if let Some(d) = &dir {
        std::fs::create_dir_all(d).map_err(|e| bad(C, format!("{}: {e}", d.display())))?;
    }
    let (mut imported, mut failed) = (Vec::new(), Vec::new());
    let old_len = s.lut_profiles.len();
    for (name, text, group) in cube_files(&paths) {
        let lut = match lightcraft_pipeline::lut::Lut3d::parse_cube(&text) {
            Ok(l) => l,
            Err(e) => {
                failed.push(json!([name, e]));
                continue;
            }
        };
        let display = lut.title.clone().filter(|t| !t.trim().is_empty()).unwrap_or_else(|| name.clone());
        let group = super::super::preset_import::group_from_dir(&group.unwrap_or_default()).unwrap_or_else(|| "Imported".into());
        let slug = crate::presets::slug(&format!("{group}-{display}"));
        let mut new_file = None;
        for n in 1..100_000 {
            let suffix = if n == 1 { String::new() } else { format!("-{n}") };
            let id = format!("lut:{slug}{suffix}");
            if s.lut_profiles.iter().any(|x| x.id == id) {
                continue;
            }
            let path = match &dir {
                Some(d) => d.join(format!("{slug}{suffix}.cube")),
                None => std::env::temp_dir().join(format!("lightcraft-{slug}{suffix}.cube")),
            };
            match lightcraft_catalog::safe_file::write_new(&path, text.as_bytes()) {
                Ok(()) => {
                    new_file = Some((id, path));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    rollback_import(s, old_len);
                    return Err(bad(C, format!("{}: {e}", path.display())));
                }
            }
        }
        let Some((id, path)) = new_file else {
            rollback_import(s, old_len);
            return Err(bad(C, format!("{display}: no free profile file name")));
        };
        let file = path.to_string_lossy().to_string();
        lightcraft_pipeline::lut::register(&id, lut);
        s.lut_profiles.push(LutProfile { id: id.clone(), name: display.clone(), group: group.clone(), file });
        imported.push(json!({"id": id, "name": display, "group": group}));
    }
    if let Err(e) = s.save_prefs() {
        // The complete copies remain on disk for recovery; unlinking by path here could remove
        // another process's file if it replaced one after our write.
        rollback_import(s, old_len);
        return Err(e);
    }
    Ok(json!({"imported": imported, "failed": failed}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "profile.import",
            "Import Profiles",
            [],
            None,
            "{paths: [.cube file, folder or .zip]} — 3D LUTs become creative profiles (Amount 0–200 %; rendered on the CPU) in the profile browser's groups (folder / zip name, else Imported) → {imported: [{id, name, group}], failed}",
            always,
            import
        ),
        cmd!(
            "profile.deleteImported",
            "Delete Imported Profile",
            [],
            None,
            "{id: lut:…} — remove the profile registration; retain its .cube file for recovery",
            always,
            |s, p| {
                let id = str_param(p, "id").ok_or_else(|| bad("profile.deleteImported", "missing id"))?.to_string();
                let Some(i) = s.lut_profiles.iter().position(|x| x.id == id) else {
                    return Err(bad("profile.deleteImported", format!("no imported profile `{id}`")));
                };
                let gone = s.lut_profiles.remove(i);
                if let Err(e) = s.save_prefs() {
                    s.lut_profiles.insert(i, gone);
                    return Err(e);
                }
                lightcraft_pipeline::lut::unregister(&id);
                // Keep the .cube copy recoverable. Removing a path here could delete another
                // process's file if it replaced the copy after we saved preferences.
                Ok(json!({"deleted": id, "retainedFile": gone.file}))
            }
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE: &str = "TITLE \"Proof\"\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";

    fn fixture(tag: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("lc-lut-safety-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("proof.cube");
        std::fs::write(&source, CUBE).unwrap();
        (dir, source)
    }

    #[test]
    fn import_keeps_an_orphaned_profile_and_rolls_back_when_prefs_cannot_save() {
        let (dir, source) = fixture("import");
        let lib = dir.join("library");
        std::fs::create_dir_all(&lib).unwrap();
        // A directory at prefs.json makes the settings unreadable and blocks all preference saves.
        std::fs::create_dir(lib.join("prefs.json")).unwrap();
        let profiles = lib.join("Profiles");
        std::fs::create_dir(&profiles).unwrap();
        let orphan = profiles.join("imported-proof.cube");
        std::fs::write(&orphan, b"previous profile data").unwrap();

        let mut s = Session::new().with_fs();
        s.open_library(&lib, false).unwrap();
        assert!(s.execute("profile.import", &json!({"paths": [source.to_string_lossy()]})).is_err());
        assert!(s.lut_profiles.is_empty());
        assert_eq!(std::fs::read(&orphan).unwrap(), b"previous profile data");
        assert_eq!(std::fs::read_to_string(profiles.join("imported-proof-2.cube")).unwrap(), CUBE, "a complete copy remains recoverable");
        assert_eq!(std::fs::read_to_string(&source).unwrap(), CUBE);
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_preserves_profile_when_prefs_save_fails_and_keeps_files_recoverable() {
        let (dir, source) = fixture("delete");
        let lib = dir.join("library");
        let mut s = Session::new().with_fs();
        s.open_library(&lib, false).unwrap();
        let imported = s.execute("profile.import", &json!({"paths": [source.to_string_lossy()]})).unwrap();
        let id = imported["imported"][0]["id"].as_str().unwrap().to_string();
        let profile = s.lut_profiles.iter().find(|p| p.id == id).unwrap().clone();
        assert_eq!(std::fs::read_to_string(&profile.file).unwrap(), CUBE);

        std::fs::rename(lib.join("prefs.json"), lib.join("prefs-saved.json")).unwrap();
        std::fs::create_dir(lib.join("prefs.json")).unwrap();
        assert!(s.execute("profile.deleteImported", &json!({"id": id})).is_err());
        assert!(s.lut_profiles.iter().any(|p| p.id == id));
        assert!(lightcraft_pipeline::lut::get(&id).is_some());
        assert_eq!(std::fs::read_to_string(&profile.file).unwrap(), CUBE);

        std::fs::remove_dir(lib.join("prefs.json")).unwrap();
        std::fs::rename(lib.join("prefs-saved.json"), lib.join("prefs.json")).unwrap();
        let unrelated = lib.join("Profiles").join("someone-elses.cube");
        std::fs::write(&unrelated, b"keep this file").unwrap();
        s.lut_profiles.iter_mut().find(|p| p.id == id).unwrap().file = unrelated.to_string_lossy().to_string();
        s.execute("profile.deleteImported", &json!({"id": id})).unwrap();
        assert_eq!(std::fs::read(&unrelated).unwrap(), b"keep this file");
        assert!(std::path::Path::new(&profile.file).exists(), "the imported copy remains recoverable");
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
