//! Native inpainting setup is controlled through the same commands as its UI.
use super::{CommandSpec, always, bad, cmd};
use crate::{
    Result, Session,
    generative_model::{Generator, KLEIN_LICENSE_NAME, KLEIN_LICENSE_URL, KLEIN_MODEL_BYTES, LICENSE_NAME, LICENSE_URL},
};
use serde_json::{Value, json};

fn status(s: &mut Session, p: &Value) -> Result<Value> {
    let _ = model_choice(p, "generative.model.status")?;
    Ok(json!({
        "available": Generator::AVAILABLE,
        "installed": s.generator.installed(),
        "removalInstalled": s.generator.removal_model().is_ok(),
        "removalModel": s.generator.removal.as_ref().map(|p| p.to_string_lossy()),
        "dir": s.generator.dir.as_ref().map(|d| d.to_string_lossy()),
        "download": s.generator.download_status(),
        "license": LICENSE_NAME,
        "licenseUrl": LICENSE_URL,
        "device": crate::generative_model::backend_description(),
        "klein": {
            "available": Generator::KLEIN_AVAILABLE,
            "installed": s.generator.klein_installed(),
            "dir": s.generator.klein_dir.as_ref().map(|d| d.to_string_lossy()),
            "download": s.generator.klein_download_status(),
            "bytes": KLEIN_MODEL_BYTES,
            "license": KLEIN_LICENSE_NAME,
            "licenseUrl": KLEIN_LICENSE_URL,
        },
    }))
}

fn model_choice(p: &Value, command: &str) -> Result<bool> {
    match p.get("model") {
        None => Ok(false),
        Some(Value::String(value)) if value == "native" => Ok(false),
        Some(Value::String(value)) if value == "klein" => Ok(true),
        Some(_) => Err(bad(command, "model must be native or klein")),
    }
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "generative.model.download";
    let klein = model_choice(p, C)?;
    if p.get("acknowledged").and_then(Value::as_bool) != Some(true) {
        return Err(bad(C, "Downloading requires explicit acknowledgement of the model licence and download size"));
    }
    let started = if klein { s.generator.start_klein_download() } else { s.generator.start_download() }.map_err(|e| bad(C, e))?;
    Ok(json!({"started": started}))
}

fn cancel_download(s: &mut Session, p: &Value) -> Result<Value> {
    let klein = model_choice(p, "generative.model.cancel")?;
    Ok(json!({"cancelled": if klein { s.generator.cancel_klein_download() } else { s.generator.cancel_download() }}))
}

fn cancel_generation(s: &mut Session, _: &Value) -> Result<Value> {
    s.generator.cancel_generation();
    Ok(json!({"requested": true}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(query "generative.model.status", "Inpainting Model Status", [], None, "{} — native and Klein model availability, local folders, download progress and licences", always, status),
        cmd!(
            "generative.model.download",
            "Download Inpainting Model",
            [],
            None,
            "{model?: native|klein, acknowledged: true} — explicitly download or verify and repair pinned model files; native is the legacy default",
            always,
            download
        ),
        cmd!(
            "generative.model.cancel",
            "Cancel Inpainting Model Download",
            [],
            None,
            "{model?: native|klein} — stop the selected model download; keeps verified files and resumable parts",
            always,
            cancel_download
        ),
        cmd!(
            "photo.generativeCancel",
            "Cancel Generative Fill",
            [],
            None,
            "{} — request cancellation between model steps; the original photo is retained",
            always,
            cancel_generation
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "inpaint")]
    #[test]
    fn bundled_removal_availability_does_not_depend_on_fill_download() {
        let mut s = Session::with_demo();
        s.generator.dir = None;
        let path = std::env::temp_dir().join(format!("lc-removal-status-{}.safetensors", std::process::id()));
        std::fs::File::create(&path).unwrap().set_len(29_528_324).unwrap();
        s.generator.removal = Some(path.clone());
        let state = status(&mut s, &json!({})).unwrap();
        assert_eq!(state["installed"], false);
        assert_eq!(state["removalInstalled"], true);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(status(&mut s, &json!({})).unwrap()["removalInstalled"], false);
    }

    #[test]
    fn model_download_requires_explicit_acknowledgement() {
        let mut s = Session::with_demo();
        let n = s.catalog.photos().count();
        for p in [json!({}), json!({"acknowledged": false})] {
            let e = download(&mut s, &p).unwrap_err().to_string();
            assert!(e.contains("explicit acknowledgement"));
            assert!(!s.generator.download_status().running);
        }
        assert_eq!(n, s.catalog.photos().count());
        assert!(s.undo.is_empty());
    }

    #[test]
    fn native_setup_status_and_cancel_are_offline_queries() {
        let mut s = Session::with_demo();
        s.generator.dir = Some(std::env::temp_dir().join("lc-inpaint-absent-test"));
        let st = status(&mut s, &json!({})).unwrap();
        assert_eq!(st["installed"], false);
        assert_eq!(st["download"]["running"], false);
        assert_eq!(cancel_download(&mut s, &json!({})).unwrap()["cancelled"], false);
        let signals = s.generator.generation_signals();
        cancel_generation(&mut s, &json!({})).unwrap();
        assert!(signals.cancel.load(std::sync::atomic::Ordering::SeqCst));
        let signals = s.generator.generation_signals();
        assert!(!signals.cancel.load(std::sync::atomic::Ordering::SeqCst));
        signals.progress.update(7, 24);
        assert_eq!(s.generator.generation_progress(), (7, 24));
        signals.progress.set_phase("Denoise");
        assert_eq!(s.generator.generation_phase(), "Denoise");
        assert!(!s.generator.klein_download_status().running);
        assert_eq!(cancel_download(&mut s, &json!({"model":"klein"})).unwrap()["cancelled"], false);
        assert!(download(&mut s, &json!({"model":"klein"})).is_err());
        assert!(!s.generator.klein_download_status().running);
    }

    #[test]
    fn klein_setup_missing_files_does_not_start_network_or_touch_catalog() {
        let mut s = Session::with_demo();
        let original = s.catalog.photos().count();
        s.generator.klein_dir = Some(std::env::temp_dir().join(format!("lc-klein-missing-{}", std::process::id())));
        let state = status(&mut s, &json!({})).unwrap();
        assert_eq!(state["klein"]["installed"], false);
        assert_eq!(state["klein"]["download"]["running"], false);
        assert!(s.generator.klein_model_dir().is_err());
        assert!(download(&mut s, &json!({"model":"invalid","acknowledged":true})).is_err());
        assert!(!s.generator.klein_download_status().running);
        assert_eq!(s.catalog.photos().count(), original);
        assert!(s.undo.is_empty());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn klein_refuses_non_cuda_build_before_model_access() {
        let mut s = Session::with_demo();
        const { assert!(!Generator::KLEIN_AVAILABLE) };
        s.generator.klein_dir = None;
        assert!(s.generator.klein_model_dir().unwrap_err().contains("CUDA"));
        assert!(s.generator.start_klein_download().unwrap_err().contains("CUDA"));
        assert!(!s.generator.klein_download_status().running);
    }

    #[test]
    fn non_string_model_never_chooses_legacy_download() {
        let mut s = Session::with_demo();
        s.generator.dir = None;
        s.generator.klein_dir = None;
        for value in [Value::Null, json!(true), json!(123), json!([]), json!({})] {
            let request = json!({"model": value, "acknowledged": true});
            let error = download(&mut s, &request).unwrap_err().to_string();
            assert!(error.contains("model must"), "{error}");
            assert!(status(&mut s, &request).unwrap_err().to_string().contains("model must"));
            assert!(cancel_download(&mut s, &request).unwrap_err().to_string().contains("model must"));
            assert!(!s.generator.download_status().running);
            assert!(!s.generator.klein_download_status().running);
        }
    }
}
