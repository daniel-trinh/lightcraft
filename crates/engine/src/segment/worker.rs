//! The AI mask worker thread: it owns one SAM model and the last encoded photo, and runs every
//! request (load, render the model input, encode, clicks, descriptions, detail passes) in
//! turn. The session only sends jobs and receives outcomes over channels, so no thread ever
//! waits on a lock the model holds, and the UI thread never runs the model.
//!
//! Every job runs under `catch_unwind`: a panic inside candle becomes an error outcome, drops
//! the (possibly inconsistent) model and leaves the worker ready for the next job. The busy
//! counters are decremented by drop guards, so they can't stay stuck. The model is unloaded
//! after [`IDLE_UNLOAD`] without requests (it holds several GB).

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use lightcraft_develop::{ObjectModel, SegMask};
use lightcraft_segment::sam1::{self, Sam1};
use lightcraft_segment::{Encoded, Sam3};

use super::{Input, Prompt, Tag};
use crate::media::RenderJob;

/// Unload the model after this long without a request.
pub const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);
static NO_CANCEL: AtomicBool = AtomicBool::new(false);

/// What to compute.
pub enum Kind {
    /// Load the model and encode the photo (so the first click is fast).
    Prepare,
    Clicks(Vec<lightcraft_segment::Click>),
    Text(String),
    /// A zoomed-in pass over `region` (normalized) of the photo, rendered by the job's `render`.
    Detail {
        prompt: Prompt,
        region: [f64; 4],
    },
    /// Panics (tests of the worker's recovery).
    #[cfg(test)]
    Panic,
}

/// One request.
pub struct Job {
    /// SAM 3's model folder or SAM 1's checkpoint file.
    pub dir: PathBuf,
    pub model: ObjectModel,
    /// Which photo and look the request is about (the encode cache's key and stale-result guard).
    pub key: u64,
    /// Renders the model input when the cache doesn't hold `key` (always for a detail pass).
    pub render: Option<RenderJob>,
    /// The photo's file, when it is missing on disk (for a clearer error if rendering fails).
    pub missing: Option<String>,
    pub kind: Kind,
    pub tag: Tag,
    pub reply: mpsc::Sender<Outcome>,
}

/// A finished request: the segmentation (`None`: nothing selected / nothing to report).
pub struct Outcome {
    pub tag: Tag,
    /// Photo/content/develop key captured before rendering the model input.
    pub key: u64,
    pub result: Result<Option<SegMask>, String>,
    /// A newer request for the same selection came in; this one was skipped.
    pub superseded: bool,
}

/// Counters the session reads without locking.
#[derive(Default)]
pub struct Shared {
    /// Requests other than detail passes, queued or running.
    pub pending: AtomicUsize,
    /// Detail passes queued or running.
    pub detail: AtomicUsize,
    /// The model is in memory.
    pub loaded: AtomicBool,
    /// Loading the model or encoding a photo right now (the slow parts).
    pub encoding: AtomicBool,
}

/// Decrements a counter when dropped (also when unwinding).
struct Done<'a>(&'a AtomicUsize);

impl Drop for Done<'_> {
    fn drop(&mut self) {
        let _ = self.0.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| Some(n.saturating_sub(1)));
    }
}

/// Sets a flag while alive.
struct Flag<'a>(&'a AtomicBool);

impl<'a> Flag<'a> {
    fn raise(f: &'a AtomicBool) -> Self {
        f.store(true, Ordering::SeqCst);
        Flag(f)
    }
}

impl Drop for Flag<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// The session's handle on the worker thread (started on the first request, restarted if it
/// ever ends).
#[derive(Default)]
pub struct Worker {
    tx: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub shared: Arc<Shared>,
}

impl Worker {
    fn counter(&self, job: &Job) -> &AtomicUsize {
        if matches!(job.kind, Kind::Detail { .. }) { &self.shared.detail } else { &self.shared.pending }
    }

    /// Whether the thread has ended (it never should); its counters are then meaningless.
    fn dead(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| t.is_finished())
    }

    pub fn pending(&self) -> usize {
        if self.dead() { 0 } else { self.shared.pending.load(Ordering::SeqCst) }
    }

    pub fn detail(&self) -> usize {
        if self.dead() { 0 } else { self.shared.detail.load(Ordering::SeqCst) }
    }

    pub fn encoding(&self) -> bool {
        !self.dead() && self.shared.encoding.load(Ordering::SeqCst)
    }

    pub fn loaded(&self) -> bool {
        !self.dead() && self.shared.loaded.load(Ordering::SeqCst)
    }

    /// Queue `job`; returns at once.
    pub fn submit(&mut self, job: Job) -> Result<(), String> {
        if self.dead() {
            // a thread that ended took its counters' meaning with it
            self.tx = None;
            self.thread = None;
            self.shared = Arc::new(Shared::default());
        }
        self.counter(&job).fetch_add(1, Ordering::SeqCst);
        let job = match &self.tx {
            Some(tx) => match tx.send(job) {
                Ok(()) => return Ok(()),
                Err(mpsc::SendError(job)) => job,
            },
            None => job,
        };
        // (re)start the thread
        let (tx, rx) = mpsc::channel();
        let shared = self.shared.clone();
        match std::thread::Builder::new().name("ai-mask".into()).spawn(move || run(&rx, &shared, IDLE_UNLOAD)) {
            Ok(t) => {
                self.thread = Some(t);
                self.tx = Some(tx.clone());
                tx.send(job).map_err(|_| "the AI mask worker stopped".to_string())
            }
            Err(e) => {
                let _ = self.counter(&job).try_update(Ordering::SeqCst, Ordering::SeqCst, |n| Some(n.saturating_sub(1)));
                Err(format!("could not start the AI mask worker: {e}"))
            }
        }
    }
}

/// The model and the last encoded photo.
#[derive(Default)]
struct State {
    model: Option<(ObjectModel, PathBuf, Model)>,
    cache: Option<(u64, Encoding)>,
}

enum Model {
    Sam3(Box<Sam3>),
    Sam1(Box<Sam1>),
}

enum Encoding {
    Sam3(Encoded),
    Sam1(sam1::Encoded),
}

/// Whether `later` makes `job` pointless (it asks for the same selection again).
fn supersedes(later: &Job, job: &Job) -> bool {
    match (&later.tag, &job.tag) {
        (Tag::Prepare, Tag::Prepare) => later.key == job.key && later.model == job.model,
        (Tag::Clicks { photo: a, mask: am, comp: ac, model: ax, .. }, Tag::Clicks { photo: b, mask: bm, comp: bc, model: bx, .. }) => {
            (a, am, ac, ax) == (b, bm, bc, bx)
        }
        (Tag::Detail { photo: a, mask: am, comp: ac, .. }, Tag::Detail { photo: b, mask: bm, comp: bc, .. }) => {
            (a, am, ac, later.model) == (b, bm, bc, job.model)
        }
        _ => false,
    }
}

fn run(rx: &mpsc::Receiver<Job>, shared: &Shared, idle: Duration) {
    let mut state = State::default();
    let mut queue: VecDeque<Job> = VecDeque::new();
    loop {
        if queue.is_empty() {
            match rx.recv_timeout(idle) {
                Ok(j) => queue.push_back(j),
                Err(RecvTimeoutError::Timeout) => {
                    if state.model.is_some() {
                        state = State::default();
                        shared.loaded.store(false, Ordering::SeqCst);
                        log::info!("AI mask model unloaded after {} minutes without use", idle.as_secs() / 60);
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            queue.push_back(j);
        }
        let Some(mut job) = queue.pop_front() else { continue };
        let is_detail = matches!(job.kind, Kind::Detail { .. });
        let _done = Done(if is_detail { &shared.detail } else { &shared.pending });
        if queue.iter().any(|later| supersedes(later, &job)) {
            let _ = job.reply.send(Outcome { tag: job.tag, key: job.key, result: Ok(None), superseded: true });
            continue;
        }
        let result = match std::panic::catch_unwind(AssertUnwindSafe(|| run_job(&mut state, shared, &mut job))) {
            Ok(r) => r,
            Err(p) => {
                let why = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
                log::error!("AI mask model panicked: {why}");
                // don't trust what the model kept: load it again next time
                state = State::default();
                Err("the AI model failed unexpectedly (it will be reloaded on the next try)".to_string())
            }
        };
        if result.is_err() && !is_detail {
            state.cache = None;
        }
        shared.loaded.store(state.model.is_some(), Ordering::SeqCst);
        let _ = job.reply.send(Outcome { tag: job.tag, key: job.key, result, superseded: false });
    }
}

/// The model for `dir`, loading it when needed.
fn model<'a>(state: &'a mut State, shared: &Shared, backend: ObjectModel, dir: &std::path::Path) -> Result<&'a mut Model, String> {
    if state.model.as_ref().is_none_or(|(b, d, _)| *b != backend || d != dir) {
        state.model = None;
        state.cache = None;
        let _busy = Flag::raise(&shared.encoding);
        let t = web_time::Instant::now();
        let m = match backend {
            ObjectModel::Sam3 => Model::Sam3(Box::new(Sam3::load(dir).map_err(|e| e.to_string())?)),
            ObjectModel::Sam1 => {
                Model::Sam1(Box::new(Sam1::load(dir, lightcraft_segment::best_device().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?))
            }
        };
        let device = match &m {
            Model::Sam3(m) => m.device(),
            Model::Sam1(m) => m.device(),
        };
        let compute = if device.is_cuda() {
            "NVIDIA CUDA"
        } else if device.is_metal() {
            "Metal"
        } else {
            "CPU"
        };
        log::info!("{backend:?} loaded on {compute} in {:?}", t.elapsed());
        state.model = Some((backend, dir.to_path_buf(), m));
        shared.loaded.store(true, Ordering::SeqCst);
    }
    state.model.as_mut().map(|(_, _, m)| m).ok_or_else(|| "the model did not load".to_string())
}

/// Render the model input.
fn render(job: &mut Job) -> Result<Input, String> {
    let r = job.render.take().ok_or("the photo is not available")?.run();
    let img = match r.rendered {
        Ok(r) => r.image,
        Err(e) => {
            return Err(match &job.missing {
                Some(path) => format!(
                    "This photo's file is missing (moved or deleted), so it can't be analyzed for AI masks: {path}. Library ▸ Find Missing Photos can relink it."
                ),
                None => format!("couldn't render the photo for AI masks: {e}"),
            });
        }
    };
    let rgb = img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    Ok(Input { rgb, w: img.width, h: img.height })
}

/// The model and the encoded photo `job.key` (rendering and encoding it when needed).
fn encoded<'a>(state: &'a mut State, shared: &Shared, job: &mut Job) -> Result<(&'a mut Model, &'a mut Encoding), String> {
    model(state, shared, job.model, &job.dir)?;
    if state.cache.as_ref().is_none_or(|(k, _)| *k != job.key) {
        state.cache = None;
        let input = render(job)?;
        let _busy = Flag::raise(&shared.encoding);
        let m = state.model.as_mut().map(|(_, _, m)| m).ok_or("the model did not load")?;
        let t = web_time::Instant::now();
        let enc = match m {
            Model::Sam3(m) => {
                let mut enc = m.encode(&input.rgb, input.w, input.h).map_err(|e| e.to_string())?;
                m.prepare_clicks(&mut enc).map_err(|e| e.to_string())?;
                Encoding::Sam3(enc)
            }
            Model::Sam1(m) => Encoding::Sam1(m.encode(&input.rgb, input.w, input.h, &NO_CANCEL).map_err(|e| e.to_string())?),
        };
        log::info!("{:?} encoded {}×{} in {:?}", job.model, input.w, input.h, t.elapsed());
        state.cache = Some((job.key, enc));
    }
    let State { model, cache } = state;
    match (model.as_mut(), cache.as_mut()) {
        (Some((_, _, m)), Some((_, e))) => Ok((m, e)),
        _ => Err("the model is not ready".into()),
    }
}

/// Every phrase of `text` ("car, road"), merged (the per-pixel maximum); `None` when nothing
/// matches.
fn text_logits(model: &mut Sam3, enc: &mut Encoded, text: &str) -> Result<Option<Vec<f32>>, String> {
    let mut merged: Option<Vec<f32>> = None;
    for phrase in text.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
        if let Some(p) = model.segment_text(enc, phrase, 0.5).map_err(|e| e.to_string())? {
            merged = Some(match merged {
                None => p.logits,
                Some(m) => m.iter().zip(&p.logits).map(|(a, b)| a.max(*b)).collect(),
            });
        }
    }
    Ok(merged)
}

fn click_logits(model: &mut Model, enc: &mut Encoding, clicks: &[lightcraft_segment::Click]) -> Result<(usize, Vec<f32>), String> {
    match (model, enc) {
        (Model::Sam3(m), Encoding::Sam3(e)) => Ok((lightcraft_segment::MASK_SIDE, m.segment_clicks(e, clicks).map_err(|e| e.to_string())?.logits)),
        (Model::Sam1(m), Encoding::Sam1(e)) => {
            let clicks: Vec<_> = clicks.iter().map(|c| sam1::Click { x: f64::from(c.x), y: f64::from(c.y), positive: c.positive }).collect();
            Ok((sam1::MASK_SIDE, m.segment_clicks(e, &clicks, &NO_CANCEL).map_err(|e| e.to_string())?.logits))
        }
        _ => Err("AI mask model and encoded image do not match".into()),
    }
}

fn run_job(state: &mut State, shared: &Shared, job: &mut Job) -> Result<Option<SegMask>, String> {
    match std::mem::replace(&mut job.kind, Kind::Prepare) {
        Kind::Prepare => {
            encoded(state, shared, job)?;
            Ok(None)
        }
        Kind::Clicks(clicks) => {
            let (m, enc) = encoded(state, shared, job)?;
            let (side, logits) = click_logits(m, enc, &clicks)?;
            Ok(Some(SegMask::from_logits(side, &logits)))
        }
        Kind::Text(text) => {
            if job.model != ObjectModel::Sam3 {
                return Err("text masks require SAM 3".into());
            }
            let (m, enc) = encoded(state, shared, job)?;
            let (Model::Sam3(m), Encoding::Sam3(enc)) = (m, enc) else { return Err("text masks require SAM 3".into()) };
            Ok(text_logits(m, enc, &text)?.map(|l| SegMask::from_logits(lightcraft_segment::MASK_SIDE, &l)))
        }
        #[cfg(test)]
        #[allow(clippy::panic)]
        Kind::Panic => panic!("a test panic"),
        Kind::Detail { prompt, region } => {
            let input = render(job)?;
            let m = model(state, shared, job.model, &job.dir)?;
            let t = web_time::Instant::now();
            let mut enc = match m {
                Model::Sam3(m) => Encoding::Sam3(m.encode(&input.rgb, input.w, input.h).map_err(|e| e.to_string())?),
                Model::Sam1(m) => Encoding::Sam1(m.encode(&input.rgb, input.w, input.h, &NO_CANCEL).map_err(|e| e.to_string())?),
            };
            let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
            if !(rw > 0.0 && rh > 0.0) {
                return Ok(None);
            }
            let logits = match prompt {
                Prompt::Clicks(clicks) => {
                    let inside: Vec<lightcraft_segment::Click> = clicks
                        .iter()
                        .map(|c| ((c.at.x - region[0]) / rw, (c.at.y - region[1]) / rh, c.include))
                        .filter(|(x, y, _)| (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y))
                        .map(|(x, y, positive)| lightcraft_segment::Click { x: x as f32, y: y as f32, positive })
                        .collect();
                    if !inside.iter().any(|c| c.positive) {
                        return Ok(None);
                    }
                    Some(click_logits(m, &mut enc, &inside)?)
                }
                Prompt::Text(text) => {
                    let (Model::Sam3(m), Encoding::Sam3(enc)) = (m, &mut enc) else { return Err("text masks require SAM 3".into()) };
                    text_logits(m, enc, &text)?.map(|l| (lightcraft_segment::MASK_SIDE, l))
                }
            };
            log::info!("{:?} detail pass ({}×{}) in {:?}", job.model, input.w, input.h, t.elapsed());
            Ok(logits.map(|(side, l)| SegMask::from_logits_in(side, &l, region)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(tag: Tag, reply: &mpsc::Sender<Outcome>) -> Job {
        // an empty folder: loading fails with an error (no model files)
        let dir = std::env::temp_dir().join(format!("lc-sam3-worker-none-{}", std::process::id()));
        Job { dir, model: ObjectModel::Sam3, key: 1, render: None, missing: None, kind: Kind::Prepare, tag, reply: reply.clone() }
    }

    #[test]
    fn errors_come_back_and_the_counters_return_to_zero() {
        let mut w = Worker::default();
        let (tx, rx) = mpsc::channel();
        for _ in 0..3 {
            w.submit(job(Tag::Wait, &tx)).unwrap();
        }
        for _ in 0..3 {
            let o = rx.recv_timeout(Duration::from_secs(30)).unwrap();
            assert!(o.result.unwrap_err().contains("not found"));
        }
        let t = web_time::Instant::now();
        while w.pending() > 0 && t.elapsed() < Duration::from_secs(5) {
            std::thread::yield_now();
        }
        assert_eq!(w.pending(), 0);
        assert!(!w.encoding() && !w.loaded());
    }

    #[test]
    fn a_panic_in_a_job_is_an_error_and_the_worker_carries_on() {
        let shared = Shared::default();
        let (tx, rx) = mpsc::channel();
        let (jtx, jrx) = mpsc::channel();
        let mut j = job(Tag::Wait, &tx);
        j.kind = Kind::Panic;
        jtx.send(j).unwrap();
        jtx.send(job(Tag::Wait, &tx)).unwrap();
        shared.pending.store(2, Ordering::SeqCst);
        drop(jtx);
        run(&jrx, &shared, Duration::from_secs(60));
        let first = rx.recv().unwrap();
        assert!(first.result.unwrap_err().contains("failed unexpectedly"));
        let second = rx.recv().unwrap();
        assert!(second.result.is_err(), "the next job still runs");
        assert_eq!(shared.pending.load(Ordering::SeqCst), 0, "counters reset");
        assert!(!shared.encoding.load(Ordering::SeqCst));
    }

    #[test]
    fn a_newer_request_for_the_same_selection_supersedes_a_queued_one() {
        let (tx, rx) = mpsc::channel();
        let (jtx, jrx) = mpsc::channel();
        let clicks = |n: usize| Tag::Clicks {
            photo: lightcraft_catalog::PhotoId(1),
            mask: 1,
            comp: 0,
            model: ObjectModel::Sam3,
            hint: vec![lightcraft_geom::Point::new(0.5, 0.5); n],
            exclude: vec![],
        };
        for n in 1..=3 {
            jtx.send(job(clicks(n), &tx)).unwrap();
        }
        drop(jtx);
        let shared = Shared::default();
        shared.pending.store(3, Ordering::SeqCst);
        run(&jrx, &shared, Duration::from_secs(60));
        let out: Vec<Outcome> = rx.try_iter().collect();
        assert_eq!(out.len(), 3);
        assert!(out[0].superseded && out[1].superseded && !out[2].superseded);
        assert_eq!(shared.pending.load(Ordering::SeqCst), 0);
    }
}
