//! Native inpainting setup is controlled through the same commands as its UI.
use super::{CommandSpec, always, bad, cmd};
use crate::{
    Result, Session,
    generative_model::{Generator, LICENSE_NAME, LICENSE_URL},
};
use serde_json::{Value, json};

fn status(s: &mut Session, _: &Value) -> Result<Value> {
    Ok(json!({
        "available": Generator::AVAILABLE,
        "installed": s.generator.installed(),
        "dir": s.generator.dir.as_ref().map(|d| d.to_string_lossy()),
        "download": s.generator.download_status(),
        "license": LICENSE_NAME,
        "licenseUrl": LICENSE_URL,
        "device": if cfg!(target_os="macos") { "Metal when available, otherwise CPU" } else { "CPU" },
    }))
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "generative.model.download";
    if p.get("acknowledged").and_then(Value::as_bool) != Some(true) {
        return Err(bad(C, "Downloading requires explicit acknowledgement of the model licence and download size"));
    }
    let started = s.generator.start_download().map_err(|e| bad(C, e))?;
    Ok(json!({"started": started}))
}

fn cancel_download(s: &mut Session, _: &Value) -> Result<Value> {
    Ok(json!({"cancelled": s.generator.cancel_download()}))
}

fn cancel_generation(s: &mut Session, _: &Value) -> Result<Value> {
    s.generator.cancel_generation();
    Ok(json!({"requested": true}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(query "generative.model.status", "Inpainting Model Status", [], None, "{} — native model availability, local folder, download progress and licence", always, status),
        cmd!(
            "generative.model.download",
            "Download Inpainting Model",
            [],
            None,
            "{acknowledged: true} — download the pinned SD1.5 inpainting files after accepting the named model licence; resumes partial downloads",
            always,
            download
        ),
        cmd!(
            "generative.model.cancel",
            "Cancel Inpainting Model Download",
            [],
            None,
            "{} — stop the model download; keeps verified files and resumable parts",
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
    }
}
