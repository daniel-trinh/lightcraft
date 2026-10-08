//! Native local inpainting, with an optional advanced ComfyUI backend.
//! The model output is blended into a new PNG beside the source; the source is never changed.

use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use lightcraft_catalog::{PhotoId, Source};
use lightcraft_codecs::{EncodeImage, EncodeMeta};
use lightcraft_pipeline::{MaskView, Overlay};
use lightcraft_raster::Rgba8;
use serde_json::{Value, json};

use crate::{Session, media::RenderJob};

const MAX_RESPONSE: usize = 64 * 1024 * 1024;
const MAX_EDGE: usize = 2048;
const NATIVE_MAX_EDGE: usize = 512;
static NEXT_NAME: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct Options {
    pub backend: Backend,
    pub endpoint: String,
    pub checkpoint: String,
    pub prompt: String,
    pub negative: String,
    pub steps: u32,
    pub guidance: f64,
    pub denoise: f32,
    pub seed: u64,
    pub edge: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Native,
    Comfy,
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.prompt.len() > 4096 || self.negative.len() > 4096 {
            return Err("prompt is too long (maximum 4096 bytes)".into());
        }
        if !(1..=100).contains(&self.steps) {
            return Err("steps must be 1..100".into());
        }
        if !self.guidance.is_finite() || !(1.0..=20.0).contains(&self.guidance) {
            return Err("guidance must be 1..20".into());
        }
        match self.backend {
            Backend::Native => {
                if !self.denoise.is_finite() || self.denoise != 1.0 {
                    return Err("native inpainting currently requires denoise=1; partial denoising is only available with backend=comfy".into());
                }
                if !(64..=NATIVE_MAX_EDGE).contains(&self.edge) {
                    return Err("native inpainting supports image edges from 64 to 512 pixels".into());
                }
            }
            Backend::Comfy => {
                endpoint(&self.endpoint)?;
                if self.checkpoint.trim().is_empty() || self.checkpoint.chars().any(char::is_control) {
                    return Err("choose an installed ComfyUI checkpoint filename".into());
                }
                if !self.denoise.is_finite() || !(0.01..=1.0).contains(&self.denoise) {
                    return Err("denoise must be 0.01..1".into());
                }
                if !(64..=MAX_EDGE).contains(&self.edge) {
                    return Err("ComfyUI image edge must be 64..2048 pixels".into());
                }
            }
        }
        Ok(())
    }
}

/// Prompts used for contextual removal. The negative prompt supplied by the user is appended.
pub fn removal_prompts(negative: &str) -> (String, String) {
    let prompt = "empty natural background, matching the surrounding texture, lighting and perspective".to_string();
    let avoid = "subject, object, person, text, artifact";
    let negative = if negative.trim().is_empty() { avoid.to_string() } else { format!("{avoid}, {negative}") };
    (prompt, negative)
}

/// Detached work suitable for a UI background task.
pub struct Job {
    pub original: PhotoId,
    pub destination: PathBuf,
    options: Options,
    image: RenderJob,
    mask: RenderJob,
    model_dir: Option<PathBuf>,
}

impl Session {
    pub fn plan_generative(&mut self, id: PhotoId, mask_id: u32, options: Options, demo_dir: Option<&str>) -> Result<Job, String> {
        options.validate()?;
        let photo = self.catalog.photo(id).ok_or("no such photo")?;
        if !photo.develop.masks.iter().any(|m| m.id == mask_id && !m.components.is_empty()) {
            return Err("select a nonempty mask first".into());
        }
        let dir = match &photo.source {
            Source::File { path } => Path::new(path).parent().ok_or("source has no parent folder")?.to_path_buf(),
            Source::Demo { .. } => PathBuf::from(demo_dir.ok_or("demo photos require a destination folder")?),
        };
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let stem = Path::new(&photo.file_name).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "photo".into());
        let model_dir = match options.backend {
            Backend::Native => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Some(self.generator.model_dir()?)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err("native inpainting is only available in the desktop app".into());
                }
            }
            Backend::Comfy => None,
        };
        let image = self.render_job(id, options.edge, options.edge, false, true).ok_or("could not render photo")?;
        let mid = u16::try_from(mask_id).map_err(|_| "mask id is too large for rendering")?;
        let mask = self.render_job(id, options.edge, options.edge, false, true).ok_or("could not render mask")?.with_overlay(Overlay::Mask {
            id: mid,
            view: MaskView::WhiteOnBlack,
            color: [255; 3],
            opacity: 100,
        });
        let destination = unique_destination(&dir, &stem);
        Ok(Job { original: id, destination, options, image, mask, model_dir })
    }
}

fn unique_destination(dir: &Path, stem: &str) -> PathBuf {
    let first = dir.join(format!("{stem}-AI-Edit.png"));
    if !first.exists() {
        return first;
    }
    for n in 2..u32::MAX {
        let p = dir.join(format!("{stem}-AI-Edit-{n}.png"));
        if !p.exists() {
            return p;
        }
    }
    dir.join(format!("{stem}-AI-Edit-{}.png", std::process::id()))
}

impl Job {
    /// Run the local model and durably write a generated file. It becomes a catalog photo only
    /// after `photo.generativeImport`, so failures cannot replace the original or an earlier edit.
    pub fn run(self) -> Result<PathBuf, String> {
        self.run_with(&std::sync::atomic::AtomicBool::new(false), |_, _| {})
    }

    /// Run generation with cooperative cancellation and model progress reporting.
    pub fn run_with(self, cancel: &std::sync::atomic::AtomicBool, mut progress: impl FnMut(usize, usize)) -> Result<PathBuf, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("generation cancelled".into());
        }
        let mut image = self.image.run().rendered?.image;
        let mask = self.mask.run().rendered?.image;
        if (image.width, image.height) != (mask.width, mask.height) || image.width == 0 || image.height == 0 {
            return Err("mask and photo sizes do not match".into());
        }
        let minimum = if self.options.backend == Backend::Native { 64 } else { 8 };
        let (w, h) = (image.width.next_multiple_of(8).max(minimum), image.height.next_multiple_of(8).max(minimum));
        let mut mask_png = Rgba8::new(mask.width, mask.height);
        let mut selected = false;
        for (src, dst) in mask.data.iter().zip(&mut mask_png.data) {
            selected |= src[0] > 8;
            // ComfyUI's LoadImageMask inverts the PNG alpha channel.
            *dst = [0, 0, 0, 255 - src[0]];
        }
        if !selected {
            return Err("the selected mask has no visible area".into());
        }
        let padded_image = pad_edges(&image, w, h);
        progress(0, 1);
        let generated = match self.options.backend {
            Backend::Native => {
                #[cfg(all(feature = "inpaint", not(target_arch = "wasm32")))]
                {
                    let model_dir = self.model_dir.as_deref().ok_or("native model directory is unavailable")?;
                    if cancel.load(Ordering::Relaxed) {
                        return Err("generation cancelled".into());
                    }
                    let mut model = lightcraft_inpaint::Model::load(model_dir).map_err(|e| format!("could not load local inpainting model: {e}"))?;
                    if cancel.load(Ordering::Relaxed) {
                        return Err("generation cancelled".into());
                    }
                    let mut rgb = Vec::with_capacity(w * h * 3);
                    for px in &padded_image.data {
                        rgb.extend_from_slice(px.get(..3).ok_or("invalid rendered pixel")?);
                    }
                    let mask8 = native_mask8(&mask, w, h);
                    let native_options = lightcraft_inpaint::GenerateOptions {
                        prompt: if self.options.prompt.trim().is_empty() {
                            "clean natural background matching the surrounding image".into()
                        } else {
                            self.options.prompt.clone()
                        },
                        negative: self.options.negative.clone(),
                        seed: self.options.seed,
                        steps: self.options.steps as usize,
                        guidance: self.options.guidance,
                    };
                    let pixels = model
                        .generate(&rgb, &mask8, w, h, &native_options, cancel, &mut progress)
                        .map_err(|e| format!("native inpainting failed: {e}"))?;
                    if pixels.len() != w.saturating_mul(h).saturating_mul(3) {
                        return Err("native inpainting returned an invalid image size".into());
                    }
                    let mut generated = Rgba8::new(w, h);
                    for (dst, src) in generated.data.iter_mut().zip(pixels.as_chunks::<3>().0) {
                        let rgb = src.get(..3).ok_or("invalid generated pixel")?;
                        dst.get_mut(..3).ok_or("invalid output pixel")?.copy_from_slice(rgb);
                        if let Some(alpha) = dst.get_mut(3) {
                            *alpha = 255;
                        }
                    }
                    Rgba8::from_fn(image.width, image.height, |x, y| generated.get(x, y))
                }
                #[cfg(not(all(feature = "inpaint", not(target_arch = "wasm32"))))]
                {
                    let _ = self.model_dir.as_deref();
                    return Err("native inpainting is not enabled in this build".into());
                }
            }
            Backend::Comfy => {
                if cancel.load(Ordering::Relaxed) {
                    return Err("generation cancelled".into());
                }
                let input = png(&padded_image)?;
                let alpha = png(&pad_edges(&mask_png, w, h))?;
                let addr = endpoint(&self.options.endpoint)?;
                let nonce = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
                let base = format!(
                    "lightcraft-{}-{}-{nonce}",
                    std::process::id(),
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
                );
                let image_name = upload(addr, &format!("{base}-photo.png"), &input)?;
                let mask_name = upload(addr, &format!("{base}-mask.png"), &alpha)?;
                let prompt = workflow(&self.options, &image_name, &mask_name, &base);
                let response = http(
                    addr,
                    "POST",
                    "/prompt",
                    &serde_json::to_vec(&json!({"prompt": prompt, "client_id": "lightcraft"})).map_err(|e| e.to_string())?,
                    "application/json",
                    1024 * 1024,
                )?;
                let ticket: Value = serde_json::from_slice(&response).map_err(|e| format!("ComfyUI prompt response: {e}"))?;
                let id = ticket.get("prompt_id").and_then(Value::as_str).ok_or_else(|| format!("ComfyUI rejected the workflow: {ticket}"))?;
                if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                    return Err("invalid ComfyUI prompt id".into());
                }
                let output = poll_result(addr, id, cancel)?;
                let generated = lightcraft_codecs::decode(
                    &output,
                    lightcraft_codecs::DecodeOptions { max_pixels: (MAX_EDGE * MAX_EDGE) as u64, ..Default::default() },
                )
                .map_err(|e| format!("ComfyUI returned an invalid image: {e}"))?
                .to_srgb8();
                if (generated.width, generated.height) != (w, h) {
                    return Err(format!("ComfyUI output is {}×{}, expected {w}×{h}", generated.width, generated.height));
                }
                Rgba8::from_fn(image.width, image.height, |x, y| generated.get(x, y))
            }
        };
        if self.options.backend == Backend::Native {
            progress(self.options.steps as usize, self.options.steps as usize);
        } else {
            progress(1, 1);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("generation cancelled".into());
        }
        composite(&mut image, &generated, &mask);
        let output = png(&image)?;
        let path = lightcraft_catalog::safe_file::write_new_unique(
            &mut (0u32..).map(|n| {
                if n == 0 {
                    self.destination.clone()
                } else {
                    let stem = self.destination.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "AI-Edit".into());
                    self.destination.with_file_name(format!("{stem}-{n}.png"))
                }
            }),
            &output,
        )
        .map_err(|e| format!("could not save generated photo: {e}"))?;
        Ok(path)
    }
}

fn pad_edges(src: &Rgba8, w: usize, h: usize) -> Rgba8 {
    Rgba8::from_fn(w, h, |x, y| src.get(x.min(src.width - 1), y.min(src.height - 1)))
}

#[cfg(any(all(feature = "inpaint", not(target_arch = "wasm32")), test))]
fn native_mask8(mask: &Rgba8, w: usize, h: usize) -> Vec<u8> {
    // Diffusion conditioning is binary. Keep the original soft mask for final blending,
    // so low-opacity selections still request generation and retain their intended opacity.
    pad_edges(mask, w, h).data.into_iter().map(|px| if px[0] > 8 { 255 } else { 0 }).collect()
}

fn composite(image: &mut Rgba8, generated: &Rgba8, mask: &Rgba8) {
    for ((dst, generated_px), m) in image.data.iter_mut().zip(&generated.data).zip(&mask.data) {
        let a = m[0] as u16;
        for c in 0..3 {
            dst[c] = ((u16::from(dst[c]) * (255 - a) + u16::from(generated_px[c]) * a + 127) / 255) as u8;
        }
    }
}

fn png(img: &Rgba8) -> Result<Vec<u8>, String> {
    lightcraft_codecs::encode_png(&EncodeImage::rgba8(img), &EncodeMeta::default()).map_err(|e| e.to_string())
}

fn endpoint(value: &str) -> Result<SocketAddr, String> {
    let authority = value.strip_prefix("http://").ok_or("ComfyUI URL must start with http://")?;
    if authority.contains(['/', '?', '#', '@']) {
        return Err("ComfyUI URL must be an origin without a path or credentials".into());
    }
    let addr: SocketAddr = authority.parse().map_err(|_| "use a loopback IP and port, such as http://127.0.0.1:8188")?;
    if !matches!(addr.ip(), IpAddr::V4(a) if a.is_loopback()) && !matches!(addr.ip(), IpAddr::V6(a) if a.is_loopback()) {
        return Err("ComfyUI must be on this computer (loopback address)".into());
    }
    Ok(addr)
}

fn workflow(o: &Options, image: &str, mask: &str, prefix: &str) -> Value {
    let prompt = if o.prompt.trim().is_empty() { "clean natural background matching the surrounding image" } else { o.prompt.as_str() };
    json!({
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "alpha"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": o.checkpoint}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": o.negative}},
        "6": {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "pixels": ["1", 0], "mask": ["2", 0], "vae": ["3", 2], "noise_mask": true}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["6", 2], "seed": o.seed, "steps": o.steps, "cfg": o.guidance, "sampler_name": "euler", "scheduler": "normal", "denoise": o.denoise}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    })
}

fn upload(addr: SocketAddr, name: &str, bytes: &[u8]) -> Result<String, String> {
    let boundary = "LightcraftInpaintBoundary";
    let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n")
        .into_bytes();
    body.extend_from_slice(bytes);
    body.extend_from_slice(
        format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}--\r\n").as_bytes(),
    );
    let response = http(addr, "POST", "/upload/image", &body, &format!("multipart/form-data; boundary={boundary}"), 1024 * 1024)?;
    let info: Value = serde_json::from_slice(&response).map_err(|e| format!("ComfyUI upload response: {e}"))?;
    let returned = info.get("name").and_then(Value::as_str).ok_or_else(|| format!("ComfyUI rejected image upload: {info}"))?;
    if returned != name {
        return Err("ComfyUI renamed the uploaded image unexpectedly".into());
    }
    Ok(returned.to_string())
}

fn poll_result(addr: SocketAddr, id: &str, cancel: &std::sync::atomic::AtomicBool) -> Result<Vec<u8>, String> {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10 * 60) {
        if cancel.load(Ordering::Relaxed) {
            return Err("generation cancelled".into());
        }
        let bytes = http(addr, "GET", &format!("/history/{id}"), &[], "", 4 * 1024 * 1024)?;
        let history: Value = serde_json::from_slice(&bytes).map_err(|e| format!("ComfyUI history response: {e}"))?;
        if let Some(record) = history.get(id) {
            if record.pointer("/status/status_str").and_then(Value::as_str) == Some("error") {
                return Err(format!("ComfyUI failed: {}", record.get("status").unwrap_or(&Value::Null)));
            }
            if let Some(file) = record.pointer("/outputs/9/images/0") {
                let filename = file.get("filename").and_then(Value::as_str).ok_or("ComfyUI output has no filename")?;
                let subfolder = file.get("subfolder").and_then(Value::as_str).unwrap_or("");
                let kind = file.get("type").and_then(Value::as_str).unwrap_or("output");
                let path = format!("/view?filename={}&subfolder={}&type={}", url_encode(filename), url_encode(subfolder), url_encode(kind));
                return http(addr, "GET", &path, &[], "", MAX_RESPONSE);
            }
            if record.pointer("/status/completed").and_then(Value::as_bool) == Some(true) {
                return Err("ComfyUI finished without an image from SaveImage".into());
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err("ComfyUI did not finish within 10 minutes".into())
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn http(addr: SocketAddr, method: &str, path: &str, body: &[u8], content_type: &str, limit: usize) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| format!("could not connect to local ComfyUI: {e}"))?;
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\n\r\n",
        body.len()
    );
    write_deadline(&mut stream, head.as_bytes(), deadline)?;
    write_deadline(&mut stream, body, deadline)?;
    let mut response = Vec::new();
    let mut buffer = [0u8; 16384];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("ComfyUI response exceeded 60 seconds".into());
        }
        stream.set_read_timeout(Some(remaining.min(Duration::from_secs(30)))).map_err(|e| e.to_string())?;
        let n = stream.read(&mut buffer).map_err(|e| format!("ComfyUI response failed: {e}"))?;
        if n == 0 {
            break;
        }
        if response.len().saturating_add(n) > limit.saturating_add(65536) {
            return Err("ComfyUI response is too large".into());
        }
        response.extend_from_slice(&buffer[..n]);
    }
    parse_response(&response, limit)
}

fn write_deadline(stream: &mut TcpStream, data: &[u8], deadline: Instant) -> Result<(), String> {
    let mut offset = 0;
    while offset < data.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("ComfyUI request exceeded 60 seconds".into());
        }
        stream.set_write_timeout(Some(remaining.min(Duration::from_secs(30)))).map_err(|e| e.to_string())?;
        let n = stream.write(&data[offset..]).map_err(|e| format!("ComfyUI request failed: {e}"))?;
        if n == 0 {
            return Err("ComfyUI request stopped accepting data".into());
        }
        offset += n;
    }
    Ok(())
}

fn parse_response(response: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let split = response.windows(4).position(|w| w == b"\r\n\r\n").ok_or("ComfyUI sent an incomplete HTTP response")?;
    let header = std::str::from_utf8(response.get(..split).ok_or("bad HTTP header")?).map_err(|_| "bad HTTP header")?;
    let status = header.lines().next().and_then(|s| s.split_whitespace().nth(1)).and_then(|s| s.parse::<u16>().ok()).ok_or("bad HTTP status")?;
    let body = response.get(split + 4..).ok_or("bad HTTP body")?;
    if !(200..300).contains(&status) {
        return Err(format!("ComfyUI HTTP {status}: {}", String::from_utf8_lossy(body).chars().take(600).collect::<String>()));
    }
    if header.lines().any(|h| h.eq_ignore_ascii_case("transfer-encoding: chunked")) {
        return decode_chunked(body, limit);
    }
    if body.len() > limit {
        return Err("ComfyUI response is too large".into());
    }
    if let Some(n) = header.lines().find_map(|h| h.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
        && n != body.len()
    {
        return Err("ComfyUI response length does not match its header".into());
    }
    Ok(body.to_vec())
}

fn decode_chunked(body: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut pos = 0usize;
    let mut out = Vec::new();
    loop {
        let line = body.get(pos..).ok_or("invalid chunk")?;
        let end = line.windows(2).position(|w| w == b"\r\n").ok_or("invalid chunk size")?;
        let size = std::str::from_utf8(line.get(..end).ok_or("invalid chunk size")?).map_err(|_| "invalid chunk size")?;
        let n = usize::from_str_radix(size.split(';').next().unwrap_or(""), 16).map_err(|_| "invalid chunk size")?;
        pos = pos.checked_add(end + 2).ok_or("invalid chunk offset")?;
        if n == 0 {
            return Ok(out);
        }
        if out.len().saturating_add(n) > limit {
            return Err("ComfyUI response is too large".into());
        }
        out.extend_from_slice(body.get(pos..pos.saturating_add(n)).ok_or("truncated chunk")?);
        pos = pos.checked_add(n + 2).ok_or("invalid chunk offset")?;
        if body.get(pos - 2..pos) != Some(&b"\r\n"[..]) {
            return Err("invalid chunk ending".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    #[test]
    fn loopback_only_and_http_body_bounds() {
        assert!(endpoint("http://127.0.0.1:8188").is_ok());
        assert!(endpoint("http://[::1]:8188").is_ok());
        assert!(endpoint("http://192.168.1.2:8188").is_err());
        assert!(endpoint("https://127.0.0.1:8188").is_err());
        assert_eq!(parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 2).unwrap(), b"ok");
        assert!(parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nok", 2).is_err());
        assert_eq!(parse_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n", 2).unwrap(), b"ok");
    }
    #[test]
    fn stock_comfy_graph_has_inpaint_path() {
        let o = Options {
            backend: Backend::Comfy,
            endpoint: "http://127.0.0.1:8188".into(),
            checkpoint: "inpaint.safetensors".into(),
            prompt: "remove person".into(),
            negative: String::new(),
            steps: 20,
            guidance: 7.5,
            denoise: 1.0,
            seed: 1,
            edge: 1024,
        };
        o.validate().unwrap();
        let g = workflow(&o, "photo.png", "mask.png", "edit");
        assert_eq!(g["6"]["class_type"], "InpaintModelConditioning");
        assert_eq!(g["6"]["inputs"]["mask"], json!(["2", 0]));
        assert_eq!(g["9"]["inputs"]["images"], json!(["8", 0]));
    }
    #[test]
    fn generated_pixels_affect_only_selected_area() {
        let mut original = Rgba8 { width: 3, height: 1, data: vec![[10, 20, 30, 255]; 3] };
        let generated = Rgba8 { width: 3, height: 1, data: vec![[210, 220, 230, 255]; 3] };
        let mask = Rgba8 { width: 3, height: 1, data: vec![[0, 0, 0, 255], [128, 128, 128, 255], [255, 255, 255, 255]] };
        composite(&mut original, &generated, &mask);
        assert_eq!(original.data[0], [10, 20, 30, 255]);
        assert_eq!(original.data[1], [110, 120, 130, 255]);
        assert_eq!(original.data[2], [210, 220, 230, 255]);
    }

    #[test]
    fn native_mask_uses_white_selected_pixels_and_edge_pads() {
        let mask = Rgba8 { width: 2, height: 1, data: vec![[0, 0, 0, 255], [50, 0, 0, 255]] };
        assert_eq!(native_mask8(&mask, 4, 2), vec![0, 255, 255, 255, 0, 255, 255, 255]);
    }
    #[test]
    fn removal_uses_context_prompt_and_keeps_user_negative_terms() {
        let (prompt, negative) = removal_prompts("red car");
        assert!(prompt.contains("background"));
        assert!(negative.contains("object"));
        assert!(negative.contains("red car"));
    }

    #[test]
    fn local_comfy_transport_uploads_and_reads_history_image() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for expected in ["POST /upload/image", "POST /prompt", "GET /history/test-id", "GET /view?"] {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut part = [0u8; 1024];
                    let n = stream.read(&mut part).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&part[..n]);
                    if let Some(i) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let headers = std::str::from_utf8(&request[..header_end]).unwrap().to_string();
                assert!(headers.starts_with(expected), "{headers}");
                let length = headers
                    .lines()
                    .find_map(|h| h.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
                    .unwrap();
                while request.len() < header_end + length {
                    let mut part = [0u8; 1024];
                    let n = stream.read(&mut part).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&part[..n]);
                }
                let body = match expected {
                    "POST /upload/image" => {
                        assert!(request.windows(7).any(|w| w == b"PNGDATA"));
                        br#"{"name":"mask.png","subfolder":"","type":"input"}"#.to_vec()
                    }
                    "POST /prompt" => {
                        let v: Value = serde_json::from_slice(&request[header_end..]).unwrap();
                        assert!(v["prompt"].is_object());
                        br#"{"prompt_id":"test-id"}"#.to_vec()
                    }
                    "GET /history/test-id" => br#"{"test-id":{"status":{"completed":true},"outputs":{"9":{"images":[{"filename":"edit 1.png","subfolder":"","type":"output"}]}}}}"#.to_vec(),
                    _ => {
                        assert!(headers.contains("filename=edit%201.png"));
                        b"GENERATED".to_vec()
                    }
                };
                let reply = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                stream.write_all(reply.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        assert_eq!(upload(addr, "mask.png", b"PNGDATA").unwrap(), "mask.png");
        let response = http(addr, "POST", "/prompt", br#"{"prompt":{}}"#, "application/json", 1024).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&response).unwrap()["prompt_id"], "test-id");
        assert_eq!(poll_result(addr, "test-id", &std::sync::atomic::AtomicBool::new(false)).unwrap(), b"GENERATED");
        server.join().unwrap();
    }

    #[test]
    fn native_generation_options_reject_unsupported_partial_denoise() {
        let options = Options {
            backend: Backend::Native,
            endpoint: String::new(),
            checkpoint: String::new(),
            prompt: "fill the selected area".into(),
            negative: String::new(),
            steps: 24,
            guidance: 7.5,
            denoise: 1.0,
            seed: 4,
            edge: 512,
        };
        assert!(options.validate().is_ok());
        let mut partial = options.clone();
        partial.denoise = 0.5;
        assert!(partial.validate().unwrap_err().contains("denoise=1"));
        let mut too_large = options;
        too_large.edge = 1024;
        assert!(too_large.validate().unwrap_err().contains("512"));
    }
}
