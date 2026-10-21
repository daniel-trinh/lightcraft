//! Separate setup commands for the ungated click-mask model.

use super::{CommandSpec, always, bad, cmd};
use serde_json::json;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(query "segment.sam1.status", "Object Mask Model Status", [], None,
        "{} — SAM 1 click-mask availability, local file and download progress (semantic SAM 3 presets are separate)", always, |s, _| {
            Ok(json!({"available": crate::segment::Segmenter::AVAILABLE,
                "installed": s.segmenter.sam1_installed(),
                    "busy": s.segmenter.busy(), "analyzing": s.segmenter.analyzing(),
                    "detailBusy": s.segmenter.detail_busy(),
                "file": s.segmenter.sam1_model.as_ref().map(|p| p.to_string_lossy()),
                "download": s.segmenter.sam1_download_status(),
                "sizeBytes": crate::segment::sam1_model::MODEL_BYTES,
                "license": "Apache-2.0", "licenseUrl": crate::segment::sam1_model::LICENSE_URL}))
        }),
        cmd!(query "segment.sam1.download", "Download Object Mask Model", [], None,
            "{acknowledged:true} — verify or download the pinned Apache-2.0 SAM 1 click-mask checkpoint (375 MB); repairs a damaged file, resumes partials and verifies SHA-256", always, |s, p| {
            if p.get("acknowledged").and_then(serde_json::Value::as_bool) != Some(true) {
                return Err(bad("segment.sam1.download", "Confirm the 375 MB Object model download before starting"));
            }
            let started = s.segmenter.start_sam1_download().map_err(|e| bad("segment.sam1.download", e))?;
            Ok(json!({"started": started}))
        }),
        cmd!(query "segment.sam1.cancel", "Cancel Object Model Download", [], None,
        "{} — stop the SAM 1 download, preserving its resumable partial file", always, |s, _| {
            Ok(json!({"cancelled":s.segmenter.cancel_sam1_download()}))
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_download_requires_explicit_acknowledgment() {
        let mut s = crate::Session::with_demo();
        let error = s.execute("segment.sam1.download", &json!({})).unwrap_err();
        assert!(error.to_string().contains("Confirm the 375 MB"));
        assert!(!s.segmenter.sam1_download_status().running);
        assert!(!s.segmenter.download_status().running);
    }
}
