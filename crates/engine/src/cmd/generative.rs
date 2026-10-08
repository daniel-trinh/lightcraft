//! Local generative fill from a selected mask, with native inference by default.

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};
use std::io::Read;

use super::{CommandSpec, bad, cmd, has_active};
use crate::{
    Result, Session,
    generative::{Backend, Options},
};

pub fn options(p: &Value) -> Result<Options> {
    const C: &str = "photo.generativeFill";
    let backend = match p.get("backend").and_then(Value::as_str) {
        Some("native") => Backend::Native,
        Some("comfy") => Backend::Comfy,
        Some(value) => return Err(bad(C, format!("unknown backend '{value}'; choose native or comfy"))),
        None if p.get("endpoint").is_some() || p.get("checkpoint").is_some() => Backend::Comfy,
        None => Backend::Native,
    };
    let o = Options {
        backend,
        endpoint: p.get("endpoint").and_then(Value::as_str).unwrap_or("http://127.0.0.1:8188").to_string(),
        checkpoint: p.get("checkpoint").and_then(Value::as_str).unwrap_or_default().to_string(),
        prompt: p.get("prompt").and_then(Value::as_str).unwrap_or_default().to_string(),
        negative: p.get("negative").and_then(Value::as_str).unwrap_or_default().to_string(),
        steps: p.get("steps").and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok()).unwrap_or(24),
        guidance: p.get("guidance").and_then(Value::as_f64).unwrap_or(7.5),
        denoise: p.get("denoise").and_then(Value::as_f64).unwrap_or(1.0) as f32,
        seed: p
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64),
        edge: p.get("edge").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()).unwrap_or(match backend {
            Backend::Native => 512,
            Backend::Comfy => 1024,
        }),
    };
    o.validate().map_err(|e| bad(C, e))?;
    Ok(o)
}

pub fn import_generated(s: &mut Session, original: PhotoId, path: &str) -> Result<Value> {
    const C: &str = "photo.generativeImport";
    if s.catalog.photo(original).is_none() {
        return Err(bad(C, "original photo is no longer in the library"));
    }
    if !path.ends_with(".png") {
        return Err(bad(C, "generated output must be a PNG"));
    }
    // Seed the import's probe cache: headless/demo sessions may have no FileProbe attached.
    // This also checks the durable output before a catalog record can refer to it.
    let file = std::fs::File::open(path).map_err(|e| bad(C, format!("{path}: {e}")))?;
    let mut bytes = Vec::new();
    file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|e| bad(C, format!("{path}: {e}")))?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(bad(C, "generated file is too large"));
    }
    let info = crate::files::probe_bytes(path, &bytes).map_err(|e| bad(C, format!("generated file is invalid: {e}")))?;
    s.import_probes.insert(path.to_string(), info);
    let undo_before = s.undo.len();
    let r = s.execute("library.import", &json!({"paths": [path]}))?;
    let id = r["imported"].get(0).and_then(Value::as_u64).ok_or_else(|| bad(C, format!("generated file was not imported: {r}")))?;
    let _ = s.execute("stack.group", &json!({"ids": [id, original.0], "top": id, "collapsed": false}));
    s.merge_undo(s.undo.len().saturating_sub(undo_before), "Generative Fill");
    s.selection = crate::Selection::single(PhotoId(id));
    Ok(json!({"id": id, "path": path, "original": original.0}))
}

fn run(s: &mut Session, p: &Value, remove: bool) -> Result<Value> {
    let c = if remove { "photo.generativeRemove" } else { "photo.generativeFill" };
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let mask = p.get("maskId").and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok()).ok_or_else(|| bad(c, "choose a maskId"))?;
    let mut opts = options(p)?;
    if remove {
        (opts.prompt, opts.negative) = crate::generative::removal_prompts(&opts.negative);
    }
    let job = s.plan_generative(id, mask, opts, p.get("dir").and_then(Value::as_str)).map_err(|e| bad(c, e))?;
    let path = job.run().map_err(|e| bad(c, e))?;
    import_generated(s, id, &path.to_string_lossy())
}

fn fill(s: &mut Session, p: &Value) -> Result<Value> {
    run(s, p, false)
}
fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    run(s, p, true)
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.generativeImport";
    let original = p.get("original").and_then(Value::as_u64).map(PhotoId).ok_or_else(|| bad(C, "missing original photo id"))?;
    let path = p.get("path").and_then(Value::as_str).ok_or_else(|| bad(C, "missing path"))?;
    import_generated(s, original, path)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "photo.generativeFill",
            "Generative Fill",
            [],
            None,
            "{maskId, backend?: native|comfy (native default), prompt?: text (empty removes), negative?, steps?: 24, guidance?: 7.5, seed?, edge?: 512, denoise?: 1 (native only supports 1), endpoint?: localhost ComfyUI URL, checkpoint?: installed checkpoint filename, dir?: demo destination} — generate selected mask as a new PNG and library photo; original is retained → {id, path, original}",
            has_active,
            fill
        ),
        cmd!(
            "photo.generativeRemove",
            "Generative Remove",
            [],
            None,
            "{maskId, backend?: native|comfy (native default), negative?, steps?: 24, guidance?: 7.5, seed?, edge?: 512, denoise?: 1 (native only supports 1), endpoint?: localhost ComfyUI URL, checkpoint?: installed checkpoint filename, dir?: demo destination} — remove the selected area using background-matching prompts, saving a separate PNG and library photo → {id, path, original}",
            has_active,
            remove
        ),
        cmd!(
            "photo.generativeImport",
            "Import Generated Edit",
            [],
            None,
            "{original, path} — add a completed local generated PNG as a separate photo (used by desktop background task)",
            super::always,
            import
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Explicit developer QA: downloads are separate from this test and all test pictures
    /// come from the procedural demo library. Normal CI does not need multi-GB model files.
    #[cfg(all(feature = "inpaint", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires LIGHTCRAFT_INPAINT_TEST_MODEL and LIGHTCRAFT_INPAINT_TEST_OUTPUT"]
    fn native_fill_and_remove_with_pinned_checkpoint() {
        let model = std::env::var_os("LIGHTCRAFT_INPAINT_TEST_MODEL").expect("model directory");
        let output = std::path::PathBuf::from(std::env::var_os("LIGHTCRAFT_INPAINT_TEST_OUTPUT").expect("output directory"));
        std::fs::create_dir_all(&output).unwrap();
        let mut s = Session::with_demo();
        s.generator.dir = Some(model.into());
        assert!(s.generator.installed());
        let source = s.active().unwrap();
        s.execute("crop.set", &json!({"rect":[0.1,0.1,0.9,0.9]})).unwrap();
        s.execute("mask.add", &json!({"kind":"radial","center":[0.5,0.6],"rx":0.18,"ry":0.16})).unwrap();
        let mid = s.active_mask.unwrap();
        let original_photo = s.catalog.photo(source).unwrap().clone();
        let expected = s.render_job(source, 256, 256, false, true).unwrap().run().rendered.unwrap().image;
        let mask = s
            .render_job(source, 256, 256, false, true)
            .unwrap()
            .with_overlay(lightcraft_pipeline::Overlay::Mask {
                id: mid as u16,
                view: lightcraft_pipeline::MaskView::WhiteOnBlack,
                color: [255; 3],
                opacity: 100,
            })
            .run()
            .rendered
            .unwrap()
            .image;
        for (name, image) in [("native-source.png", &expected), ("native-mask.png", &mask)] {
            let png =
                lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(image), &lightcraft_codecs::EncodeMeta::default()).unwrap();
            std::fs::write(output.join(name), png).unwrap();
        }
        for command in ["photo.generativeFill", "photo.generativeRemove"] {
            s.selection = crate::Selection::single(source);
            s.active_mask = Some(mid);
            let before = s.undo.len();
            let started = std::time::Instant::now();
            let r = s.execute(command,&json!({"maskId":mid,"edge":256,"steps":16,"seed":42,"prompt":"yellow flowers growing beside a mountain lake, natural photograph","negative":"red square, text, watermark","dir":output})).unwrap();
            eprintln!("{command}: {:.2}s -> {}", started.elapsed().as_secs_f64(), r["path"]);
            let path = r["path"].as_str().unwrap();
            let bytes = std::fs::read(path).unwrap();
            let generated = lightcraft_codecs::decode(&bytes, lightcraft_codecs::DecodeOptions::default()).unwrap().to_srgb8();
            assert_eq!((generated.width, generated.height), (expected.width, expected.height));
            let mut outside = 0;
            let mut changed = 0;
            for ((original, result), alpha) in expected.data.iter().zip(&generated.data).zip(&mask.data) {
                if alpha[0] == 0 {
                    assert_eq!(original, result);
                    outside += 1;
                } else if original != result {
                    changed += 1;
                }
            }
            assert!(outside > 0 && changed > 0, "generated content must change the selected area only");
            assert_eq!(s.catalog.photo(source).unwrap().develop, original_photo.develop);
            assert_eq!(s.undo.len(), before + 1);
            let generated_id = PhotoId(r["id"].as_u64().unwrap());
            s.execute("edit.undo", &json!({})).unwrap();
            assert!(s.catalog.photo(source).is_some());
            assert!(s.catalog.photo(generated_id).is_none());
            assert!(std::path::Path::new(path).is_file());
        }
    }

    #[test]
    fn cropped_generation_round_trip_preserves_unselected_pixels() {
        let mut s = Session::with_demo();
        let source = s.active().unwrap();
        s.execute("crop.set", &json!({"rect": [0.2, 0.2, 0.7, 0.6]})).unwrap();
        s.execute("mask.add", &json!({"kind": "radial", "center": [0.5, 0.5], "rx": 0.18, "ry": 0.18})).unwrap();
        let mask_id = s.active_mask.unwrap();
        let original_photo = s.catalog.photo(source).unwrap().clone();
        let expected = s.render_job(source, 64, 64, false, true).unwrap().run().rendered.unwrap().image;
        let mask = s
            .render_job(source, 64, 64, false, true)
            .unwrap()
            .with_overlay(lightcraft_pipeline::Overlay::Mask {
                id: mask_id as u16,
                view: lightcraft_pipeline::MaskView::WhiteOnBlack,
                color: [255; 3],
                opacity: 100,
            })
            .run()
            .rendered
            .unwrap()
            .image;
        assert_eq!((expected.width, expected.height), (mask.width, mask.height));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut dimensions = None;
            for expected_path in ["/upload/image", "/upload/image", "/prompt", "/history/test-id", "/view?"] {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut part = [0u8; 4096];
                    let n = stream.read(&mut part).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&part[..n]);
                    if let Some(i) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let headers = std::str::from_utf8(&request[..header_end]).unwrap().to_string();
                assert!(headers.contains(expected_path), "{headers}");
                let len = headers
                    .lines()
                    .find_map(|h| h.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
                    .unwrap();
                while request.len() < header_end + len {
                    let mut part = [0u8; 4096];
                    let n = stream.read(&mut part).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&part[..n]);
                }
                let body = if expected_path == "/upload/image" {
                    let png_at = request.windows(8).position(|w| w == b"\x89PNG\r\n\x1a\n").unwrap();
                    let w = u32::from_be_bytes(request[png_at + 16..png_at + 20].try_into().unwrap()) as usize;
                    let h = u32::from_be_bytes(request[png_at + 20..png_at + 24].try_into().unwrap()) as usize;
                    if let Some((pw, ph)) = dimensions {
                        assert_eq!((w, h), (pw, ph));
                    }
                    dimensions = Some((w, h));
                    let text = String::from_utf8_lossy(&request[..png_at]);
                    let name = text.split("filename=\"").nth(1).unwrap().split('"').next().unwrap();
                    serde_json::to_vec(&json!({"name": name})).unwrap()
                } else if expected_path == "/prompt" {
                    let v: Value = serde_json::from_slice(&request[header_end..]).unwrap();
                    assert_eq!(v["prompt"]["6"]["class_type"], "InpaintModelConditioning");
                    br#"{"prompt_id":"test-id"}"#.to_vec()
                } else if expected_path == "/history/test-id" {
                    br#"{"test-id":{"status":{"completed":true},"outputs":{"9":{"images":[{"filename":"generated.png","subfolder":"","type":"output"}]}}}}"#.to_vec()
                } else {
                    let (w, h) = dimensions.unwrap();
                    assert_eq!(w % 8, 0);
                    assert_eq!(h % 8, 0);
                    let img = lightcraft_raster::Rgba8::filled(w, h, [255, 0, 0, 255]);
                    lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap()
                };
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        let folder = std::env::temp_dir().join(format!("lightcraft-generative-e2e-{}", std::process::id()));
        let result = s
            .execute(
                "photo.generativeFill",
                &json!({
                    "maskId": mask_id, "endpoint": endpoint, "checkpoint": "inpaint.safetensors", "prompt": "red fill", "edge": 64, "dir": folder
                }),
            )
            .unwrap();
        server.join().unwrap();
        let source_after = s.catalog.photo(source).unwrap();
        assert_eq!(source_after.source, original_photo.source);
        assert_eq!(source_after.develop, original_photo.develop);
        let generated = s.catalog.photo(PhotoId(result["id"].as_u64().unwrap())).unwrap();
        assert!(matches!(&generated.source, lightcraft_catalog::Source::File { .. }));
        assert_eq!((generated.width as usize, generated.height as usize), (expected.width, expected.height));
        let bytes = std::fs::read(result["path"].as_str().unwrap()).unwrap();
        let output = lightcraft_codecs::decode(&bytes, lightcraft_codecs::DecodeOptions::default()).unwrap().to_srgb8();
        assert_eq!((output.width, output.height), (expected.width, expected.height));
        let mut outside = 0;
        let mut inside = 0;
        for ((source_px, result_px), alpha) in expected.data.iter().zip(&output.data).zip(&mask.data) {
            if alpha[0] == 0 {
                assert_eq!(source_px, result_px);
                outside += 1;
            } else if alpha[0] == 255 {
                assert_eq!(&[255, 0, 0, 255], result_px);
                inside += 1;
            }
        }
        assert!(outside > 0 && inside > 0);
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn generated_photo_import_is_one_undo_step_and_keeps_source() {
        let mut s = Session::with_demo();
        let source = s.active().unwrap();
        let folder = std::env::temp_dir().join(format!("lightcraft-generative-import-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("generated.png");
        let img = lightcraft_raster::Rgba8::filled(32, 32, [80, 120, 40, 255]);
        let bytes = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let undo_before = s.undo.len();
        let result = import_generated(&mut s, source, &path.to_string_lossy()).unwrap();
        let generated = PhotoId(result["id"].as_u64().unwrap());
        assert!(s.catalog.photo(source).is_some());
        assert!(s.catalog.photo(generated).is_some());
        assert_eq!(s.undo.len(), undo_before + 1);
        s.execute("edit.undo", &json!({})).unwrap();
        assert!(s.catalog.photo(generated).is_none());
        assert!(s.catalog.photo(source).is_some());
        assert!(path.exists(), "the generated file remains recoverable after undo");
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn unavailable_local_model_leaves_catalog_and_source_unchanged() {
        let mut s = Session::with_demo();
        let source = s.active().unwrap();
        s.execute("mask.add", &json!({"kind": "radial"})).unwrap();
        let before = s.catalog.photos().count();
        let undo_before = s.undo.len();
        let folder = std::env::temp_dir().join(format!("lightcraft-generative-fail-{}", std::process::id()));
        let e = s
            .execute(
                "photo.generativeRemove",
                &json!({
                    "maskId": s.active_mask.unwrap(), "endpoint": "http://127.0.0.1:0", "checkpoint": "inpaint.safetensors", "edge": 64, "dir": folder
                }),
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("connect"), "{e}");
        assert_eq!(s.catalog.photos().count(), before);
        assert_eq!(s.undo.len(), undo_before);
        assert!(s.catalog.photo(source).is_some());
        assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(folder);
    }
}
