//! A failed save is an error, not a silent success: a command whose journal append fails returns
//! [`EngineError::NotSaved`], its change stays applied and queued, and the next save writes it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lightcraft_catalog::{Flag, MemStore, SnapshotPolicy, Store};
use serde_json::json;

use crate::library::LibraryStores;
use crate::{EngineError, Session};

/// A store whose writes fail while `fail` is set (a full or unplugged disk).
#[derive(Clone, Default)]
struct Flaky {
    files: MemStore,
    fail: Arc<AtomicBool>,
    /// Only snapshots fail (appends work).
    snapshots_only: bool,
    /// Only appends fail (a dead log handle; fresh files can still be written).
    appends_only: bool,
    /// Only settings files fail (the catalog remains writable).
    settings_only: bool,
}

impl Flaky {
    fn failing(&self) -> bool {
        self.fail.load(Ordering::SeqCst)
    }
    fn set_failing(&self, on: bool) {
        self.fail.store(on, Ordering::SeqCst);
    }
    /// The files as a crash at this moment would leave them.
    fn crash_copy(&self) -> Flaky {
        let m = MemStore::new();
        *m.files.lock().unwrap() = self.files.files.lock().unwrap().clone();
        Flaky { files: m, ..Default::default() }
    }
}

impl Store for Flaky {
    fn read(&mut self, name: &str) -> std::io::Result<Option<Vec<u8>>> {
        self.files.read(name)
    }
    fn write_atomic(&mut self, name: &str, data: &[u8]) -> std::io::Result<()> {
        if self.failing()
            && (self.settings_only && matches!(name, "presets.json" | "prefs.json" | "view.json") || !self.appends_only && !self.settings_only)
        {
            return Err(std::io::Error::other("disk full"));
        }
        self.files.write_atomic(name, data)
    }
    fn append(&mut self, name: &str, data: &[u8]) -> std::io::Result<()> {
        if self.failing() && !self.snapshots_only && !self.settings_only {
            return Err(std::io::Error::other("disk full"));
        }
        self.files.append(name, data)
    }
    fn truncate(&mut self, name: &str, len: u64) -> std::io::Result<()> {
        self.files.truncate(name, len)
    }
    fn describe(&self) -> String {
        "flaky".into()
    }
}

fn open(store: &Flaky, seed: bool) -> Session {
    let mut s = Session::new();
    let stores = LibraryStores { dir: "flaky".into(), catalog: Box::new(store.clone()), files: Box::new(MemStore::new()), on_disk: false };
    s.open_library_in(stores, seed).unwrap();
    s
}

#[test]
fn failed_append_fails_the_command_and_is_retried() {
    let store = Flaky::default();
    let mut s = open(&store, true);
    let id = s.selection.active.unwrap();
    s.execute("photo.rate", &json!({"rating": 1})).unwrap();
    store.set_failing(true);

    let e = s.execute("photo.rate", &json!({"rating": 4})).unwrap_err();
    assert!(matches!(e, EngineError::NotSaved(_)), "{e:?}");
    let msg = e.to_string();
    assert!(
        msg.starts_with("saved in memory but not written to disk: ") && msg.contains("disk full") && msg.ends_with("LightCraft will retry"),
        "{msg}"
    );
    // applied in memory, undoable, queued
    assert_eq!(s.catalog.photo(id).unwrap().rating, 4);
    assert!(!s.undo.is_empty());
    assert_eq!(s.unsaved().map(|u| u.0), Some(1));
    // queries and commands that change nothing are unaffected (they retry the queue, but report
    // their own result)
    let info = s.execute("library.info", &json!({})).unwrap();
    assert_eq!(info["unsavedOps"], 1, "{info}");
    assert!(info["unsavedError"].as_str().unwrap().contains("disk full"), "{info}");
    s.execute("library.select", &json!({"ids": [id.0]})).unwrap();
    // a second change fails too; both stay queued, in order
    let e = s.execute("photo.flag", &json!({"flag": "pick"})).unwrap_err();
    assert!(matches!(e, EngineError::NotSaved(_)));
    assert_eq!(s.unsaved().map(|u| u.0), Some(2));
    // the frame loop backs off instead of retrying every frame
    s.persist_if_dirty();
    assert_eq!(s.unsaved().map(|u| u.0), Some(2));
    // a crash now loses only the unsaved changes; the earlier ones are on disk
    assert_eq!(open(&store.crash_copy(), true).catalog.photo(id).unwrap().rating, 1);

    // the disk comes back: the next save writes the whole queue
    store.set_failing(false);
    s.execute("library.info", &json!({})).unwrap();
    assert_eq!(s.unsaved(), None);
    assert!(s.library.as_ref().unwrap().last_error.is_none());
    let expect = s.catalog.to_snapshot();
    drop(s); // no close: like a crash
    let s2 = open(&store, true);
    assert_eq!(s2.catalog.to_snapshot(), expect);
    let p = s2.catalog.photo(id).unwrap();
    assert_eq!((p.rating, p.flag), (4, Flag::Pick));
}

/// Undo is a command like any other: a failed write of its op is reported, and retried.
#[test]
fn failed_append_of_undo_is_reported_and_retried() {
    let store = Flaky::default();
    let mut s = open(&store, true);
    let id = s.selection.active.unwrap();
    let before = s.catalog.photo(id).unwrap().rating;
    s.execute("photo.rate", &json!({"rating": 3})).unwrap();
    store.set_failing(true);
    assert!(matches!(s.execute("edit.undo", &json!({})), Err(EngineError::NotSaved(_))));
    assert_eq!(s.catalog.photo(id).unwrap().rating, before);
    store.set_failing(false);
    s.persist().unwrap();
    drop(s);
    assert_eq!(open(&store, true).catalog.photo(id).unwrap().rating, before);
}

/// A failed compaction is not a failed command: the command's own records were appended.
#[test]
fn failed_snapshot_does_not_fail_the_command() {
    let store = Flaky { snapshots_only: true, ..Default::default() };
    let mut s = open(&store, false);
    s.library.as_mut().unwrap().journal_mut().policy = SnapshotPolicy { max_records: 1, max_bytes: u64::MAX };
    store.set_failing(true);
    s.execute("album.create", &json!({"name": "A"})).unwrap();
    s.execute("album.create", &json!({"name": "B"})).unwrap();
    assert_eq!(s.unsaved(), None);
    assert!(s.library.as_ref().unwrap().last_error.is_some(), "the compaction failure is reported");
    drop(s);
    store.set_failing(false);
    assert_eq!(open(&store, false).catalog.albums().count(), 2);
}

/// Issue #101: quitting while the log can't be appended to still saves the queued changes in the
/// closing snapshot (a fresh file), and reports success; they are not appended again later.
#[test]
fn close_saves_queued_changes_in_the_snapshot_when_appends_fail() {
    let store = Flaky { appends_only: true, ..Default::default() };
    let mut s = open(&store, true);
    let id = s.selection.active.unwrap();
    store.set_failing(true);
    assert!(matches!(s.execute("photo.rate", &json!({"rating": 4})), Err(EngineError::NotSaved(_))));
    assert!(matches!(s.execute("photo.flag", &json!({"flag": "pick"})), Err(EngineError::NotSaved(_))));
    assert_eq!(s.unsaved().map(|u| u.0), Some(2));
    s.close_library().unwrap();
    assert_eq!(s.unsaved(), None);
    assert!(s.pending_log.is_empty());
    let expect = s.catalog.to_snapshot();
    drop(s);
    store.set_failing(false);
    let mut s2 = open(&store, true);
    assert_eq!(s2.catalog.to_snapshot(), expect);
    let p = s2.catalog.photo(id).unwrap();
    assert_eq!((p.rating, p.flag), (4, Flag::Pick));
    // later edits continue after the absorbed ops
    s2.execute("photo.rate", &json!({"rating": 2})).unwrap();
    let expect = s2.catalog.to_snapshot();
    drop(s2);
    assert_eq!(open(&store, true).catalog.to_snapshot(), expect);
}

/// When neither the log nor the snapshot can be written, closing fails with `NotSaved` and the
/// changes stay queued (a later retry can still save them).
#[test]
fn close_fails_when_nothing_can_be_saved() {
    let store = Flaky::default();
    let mut s = open(&store, true);
    store.set_failing(true);
    assert!(s.execute("photo.rate", &json!({"rating": 4})).is_err());
    let e = s.close_library().unwrap_err();
    assert!(matches!(e, EngineError::NotSaved(_)), "{e:?}");
    assert_eq!(s.unsaved().map(|u| u.0), Some(1));
    store.set_failing(false);
    s.close_library().unwrap();
    assert_eq!(s.unsaved(), None);
}

/// Regression proof: a failed switch must leave the prior library connected to its own state.
#[test]
fn failed_demo_open_does_not_replace_the_previous_catalog() {
    let a = Flaky::default();
    let mut s = open(&a, true);
    s.execute("album.create", &json!({"name": "Keep in A"})).unwrap();
    let expected_albums = s.catalog.albums().count();

    let b = Flaky::default();
    b.set_failing(true); // Journal::open continues, but seeding B's demo snapshot fails.
    let stores = LibraryStores { dir: "broken-b".into(), catalog: Box::new(b), files: Box::new(MemStore::new()), on_disk: false };
    assert!(s.open_library_in(stores, true).is_err());

    // This close currently snapshots B's in-memory demo into A's still-open journal.
    s.close_library().unwrap();
    drop(s);
    assert_eq!(open(&a, false).catalog.albums().count(), expected_albums, "A's saved album was overwritten");
}

#[test]
fn failed_preset_save_is_reported_retried_and_blocks_close() {
    let catalog = MemStore::new();
    let files = Flaky { settings_only: true, ..Default::default() };
    let stores = LibraryStores { dir: "settings-failure".into(), catalog: Box::new(catalog.clone()), files: Box::new(files.clone()), on_disk: false };
    let mut s = Session::new();
    s.open_library_in(stores, true).unwrap();
    files.set_failing(true);
    assert!(matches!(s.execute("preset.create", &json!({"name": "Keep Me"})), Err(EngineError::NotSaved(_))));
    assert!(s.unsaved().is_some());
    assert!(matches!(s.close_library(), Err(EngineError::NotSaved(_))));
    files.set_failing(false);
    s.close_library().unwrap();
    drop(s);
    let stores = LibraryStores { dir: "settings-failure".into(), catalog: Box::new(catalog), files: Box::new(files), on_disk: false };
    let mut fresh = Session::new();
    fresh.open_library_in(stores, false).unwrap();
    assert!(fresh.presets.iter().any(|p| p.name == "Keep Me"));
}

#[test]
fn failed_preferences_save_is_retained_until_retry() {
    let catalog = MemStore::new();
    let files = Flaky { settings_only: true, ..Default::default() };
    let stores =
        LibraryStores { dir: "preferences-failure".into(), catalog: Box::new(catalog.clone()), files: Box::new(files.clone()), on_disk: false };
    let mut s = Session::new();
    s.open_library_in(stores, false).unwrap();
    files.set_failing(true);
    assert!(matches!(s.execute("library.xmpPreferences", &json!({"autoWrite": true})), Err(EngineError::NotSaved(_))));
    assert!(s.xmp.auto_write);
    assert!(s.unsaved().is_some());
    assert!(matches!(s.close_library(), Err(EngineError::NotSaved(_))));
    files.set_failing(false);
    s.close_library().unwrap();
    drop(s);
    let stores = LibraryStores { dir: "preferences-failure".into(), catalog: Box::new(catalog), files: Box::new(files), on_disk: false };
    let mut fresh = Session::new();
    fresh.open_library_in(stores, false).unwrap();
    assert!(fresh.xmp.auto_write);
}

#[test]
fn switching_libraries_refuses_to_drop_unsaved_catalog_ops() {
    let a = Flaky::default();
    let mut s = open(&a, true);
    let id = s.active().unwrap();
    a.set_failing(true);
    assert!(matches!(s.execute("photo.rate", &json!({"rating": 4})), Err(EngineError::NotSaved(_))));
    let b = MemStore::new();
    let stores = LibraryStores { dir: "other".into(), catalog: Box::new(b), files: Box::new(MemStore::new()), on_disk: false };
    assert!(matches!(s.open_library_in(stores, false), Err(EngineError::NotSaved(_))));
    assert_eq!(s.catalog.photo(id).unwrap().rating, 4);
    assert_eq!(s.unsaved().map(|u| u.0), Some(1));
    a.set_failing(false);
    s.close_library().unwrap();
    drop(s);
    assert_eq!(open(&a, false).catalog.photo(id).unwrap().rating, 4);
}

#[test]
fn snapshot_rescue_does_not_hide_a_failed_preset_save() {
    let catalog = Flaky { appends_only: true, ..Default::default() };
    let files = Flaky { settings_only: true, ..Default::default() };
    let stores = LibraryStores { dir: "combined-failure".into(), catalog: Box::new(catalog.clone()), files: Box::new(files.clone()), on_disk: false };
    let mut s = Session::new();
    s.open_library_in(stores, true).unwrap();
    let id = s.active().unwrap();

    files.set_failing(true);
    assert!(matches!(s.execute("preset.create", &json!({"name": "Keep Me"})), Err(EngineError::NotSaved(_))));
    catalog.set_failing(true);
    assert!(matches!(s.execute("photo.rate", &json!({"rating": 4})), Err(EngineError::NotSaved(_))));

    // The snapshot can save the rating despite the broken log. The preset remains unsaved.
    assert!(matches!(s.close_library(), Err(EngineError::NotSaved(_))));
    assert_eq!(s.unsaved().map(|u| u.0), Some(1));
    files.set_failing(false);
    s.close_library().unwrap();
    drop(s);

    let stores = LibraryStores { dir: "combined-failure".into(), catalog: Box::new(catalog), files: Box::new(files), on_disk: false };
    let mut fresh = Session::new();
    fresh.open_library_in(stores, false).unwrap();
    assert_eq!(fresh.catalog.photo(id).unwrap().rating, 4);
    assert!(fresh.presets.iter().any(|p| p.name == "Keep Me"));
}
