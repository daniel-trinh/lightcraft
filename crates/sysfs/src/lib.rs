//! Atomic no-replace rename for a file or directory.
//!
//! `std::fs::rename` replaces a destination on Unix. Checking first is racy, and publishing
//! through a hard link before deleting the old name leaves two names if the process stops.
//! This helper asks the operating system to move the name only if the destination is absent.
//! Unsupported platforms and file systems return an error without a check-then-rename fallback.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::io;
use std::path::Path;

/// Move `from` to `to` atomically, refusing to replace an existing file, directory or link.
/// Both paths must be on a file system that supports the operation; cross-volume moves fail.
pub fn rename_no_replace(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        windows::rename_no_replace(from, to)
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        use rustix::fs::{CWD, RenameFlags, renameat_with};
        renameat_with(CWD, from, CWD, to, RenameFlags::NOREPLACE).map_err(|e| {
            if e == rustix::io::Errno::NOSYS || e == rustix::io::Errno::INVAL {
                io::Error::new(io::ErrorKind::Unsupported, "atomic no-replace rename is unavailable on this file system")
            } else {
                e.into()
            }
        })
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = (from, to);
        Err(io::Error::new(io::ErrorKind::Unsupported, "atomic no-replace rename is unavailable on this platform"))
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    use std::ffi::OsStr;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_WRITE_THROUGH, MoveFileExW};

    fn wide(path: &OsStr) -> io::Result<Vec<u16>> {
        let wide: Vec<u16> = path.encode_wide().collect();
        if wide.contains(&0) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL character"));
        }
        Ok(wide.into_iter().chain(std::iter::once(0)).collect())
    }

    pub(super) fn rename_no_replace(from: &Path, to: &Path) -> io::Result<()> {
        let from = wide(from.as_os_str())?;
        let to = wide(to.as_os_str())?;
        // SAFETY: both arguments are NUL-terminated UTF-16 buffers kept alive for this call.
        // MoveFileExW receives no REPLACE_EXISTING or COPY_ALLOWED flag: it atomically renames
        // only when the destination is absent on the same volume, otherwise it fails.
        let moved = unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_WRITE_THROUGH) };
        if moved == 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch() -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!("lc-sysfs-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn existing_file_is_preserved_with_the_source() {
        let dir = scratch();
        let (from, to) = (dir.join("from.txt"), dir.join("to.txt"));
        std::fs::write(&from, b"source").unwrap();
        std::fs::write(&to, b"destination").unwrap();
        assert!(rename_no_replace(&from, &to).is_err());
        assert_eq!(std::fs::read(&from).unwrap(), b"source");
        assert_eq!(std::fs::read(&to).unwrap(), b"destination");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn existing_directory_is_preserved_with_the_source() {
        let dir = scratch();
        let (from, to) = (dir.join("from"), dir.join("to"));
        std::fs::create_dir(&from).unwrap();
        std::fs::create_dir(&to).unwrap();
        std::fs::write(from.join("source.txt"), b"source").unwrap();
        std::fs::write(to.join("destination.txt"), b"destination").unwrap();
        assert!(rename_no_replace(&from, &to).is_err());
        assert_eq!(std::fs::read(from.join("source.txt")).unwrap(), b"source");
        assert_eq!(std::fs::read(to.join("destination.txt")).unwrap(), b"destination");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_source_does_not_create_destination() {
        let dir = scratch();
        let to = dir.join("to.txt");
        assert!(rename_no_replace(&dir.join("missing.txt"), &to).is_err());
        assert!(!to.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
    fn moves_file_and_directory_without_leaving_old_name() {
        let dir = scratch();
        let (file_from, file_to) = (dir.join("file.txt"), dir.join("renamed.txt"));
        std::fs::write(&file_from, b"content").unwrap();
        rename_no_replace(&file_from, &file_to).unwrap();
        assert!(!file_from.exists());
        assert_eq!(std::fs::read(&file_to).unwrap(), b"content");

        let (folder_from, folder_to) = (dir.join("folder"), dir.join("renamed-folder"));
        std::fs::create_dir(&folder_from).unwrap();
        std::fs::write(folder_from.join("child.txt"), b"child").unwrap();
        rename_no_replace(&folder_from, &folder_to).unwrap();
        assert!(!folder_from.exists());
        assert_eq!(std::fs::read(folder_to.join("child.txt")).unwrap(), b"child");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
    fn concurrent_writers_cannot_replace_the_winning_file() {
        let dir = scratch();
        let (a, b, to) = (dir.join("a"), dir.join("b"), dir.join("output"));
        std::fs::write(&a, b"first original").unwrap();
        std::fs::write(&b, b"second original").unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let gate = barrier.clone();
        let from = a.clone();
        let dest = to.clone();
        let worker = std::thread::spawn(move || {
            gate.wait();
            rename_no_replace(&from, &dest)
        });
        barrier.wait();
        let second = rename_no_replace(&b, &to);
        let first = worker.join().unwrap();
        assert_ne!(first.is_ok(), second.is_ok(), "exactly one name is moved");
        if first.is_ok() {
            assert_eq!(std::fs::read(&to).unwrap(), b"first original");
            assert_eq!(std::fs::read(&b).unwrap(), b"second original");
        } else {
            assert_eq!(std::fs::read(&to).unwrap(), b"second original");
            assert_eq!(std::fs::read(&a).unwrap(), b"first original");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
