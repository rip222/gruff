//! Per-workspace filter persistence stored under `~/.gruff/workspaces/`.
//!
//! Each opened folder maps to a truncated-SHA256 of its canonical absolute
//! path (lowercased on macOS to handle case-insensitive filesystems). The
//! resulting `<hash>.toml` carries the hide-set that the sidebar's file
//! checkboxes drive, so mid-session curation survives relaunches — on a
//! monorepo with 20 packages that has been narrowed to 3, the narrowing
//! doesn't have to be redone every launch.
//!
//! Only the hide-set is per-workspace. View toggles like `include_tests` and
//! `collapse_barrels` remain global (see `config.rs`). Malformed or missing
//! files fall back to [`WorkspaceState::default()`] silently, matching the
//! existing `config.rs` pattern — the app still launches cleanly even when
//! somebody hand-edits a workspace file into a broken state.
//!
//! The module is deliberately graph-agnostic: it serialises whatever string
//! ids the caller hands it, and the caller is responsible for pruning stale
//! ids at load time. See [`crate::app::GruffApp::load_folder`] for that flow.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::graph::NodeId;

/// Length of the hex hash used to name per-workspace files. 16 hex chars =
/// 64 bits of entropy, well clear of the collision floor for the dozens of
/// repos a single user opens, and short enough to keep directory listings
/// legible.
const HASH_LEN: usize = 16;

/// Serialised per-workspace state. Wraps the hide-set so future per-workspace
/// fields (e.g. a per-workspace layout seed) can land without a schema break —
/// `#[serde(default)]` on the field keeps older TOMLs loadable.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct WorkspaceState {
    /// Node ids currently hidden from the sidebar / canvas. Serialised as a
    /// sorted vector so TOML output is deterministic across runs, which makes
    /// manual inspection (and any future diff-based sync) easier. Deserialised
    /// back into a [`HashSet`] because that's the shape `FilterState`
    /// consumes internally.
    #[serde(default, with = "hide_set_as_vec")]
    pub hidden: HashSet<NodeId>,
}

/// Compute the stable, hex-encoded hash used to name a workspace's state
/// file. Input is expected to be an already-canonicalized absolute path —
/// this function doesn't canonicalize on the caller's behalf because the
/// app's `Indexer::build` path already produces a canonical root and
/// re-canonicalizing in two places would just invite drift.
///
/// On macOS the filesystem is case-insensitive by default, so `/tmp/Repo`
/// and `/tmp/repo` must hash identically; we fold to ASCII lowercase before
/// hashing. On Linux the filesystem is case-sensitive, so we preserve case.
pub fn hash_workspace(canonical: &Path) -> String {
    let key = path_hash_key(canonical);
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    // 8 bytes → 16 hex chars (the constant above).
    let mut hex = String::with_capacity(HASH_LEN);
    for byte in digest.iter().take(HASH_LEN / 2) {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(target_os = "macos")]
fn path_hash_key(path: &Path) -> String {
    path.to_string_lossy().to_ascii_lowercase()
}

#[cfg(not(target_os = "macos"))]
fn path_hash_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// `~/.gruff/workspaces/<hash>.toml`. `None` only when `$HOME` is unset,
/// which shouldn't happen on macOS but we handle gracefully rather than
/// panic — mirrors [`crate::config::config_path`].
pub fn workspace_path(root: &Path) -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join(".gruff")
            .join("workspaces")
            .join(format!("{}.toml", hash_workspace(root)))
    })
}

/// Load the workspace state for `root`. Missing, unreadable, or malformed
/// files fall back to [`WorkspaceState::default()`] silently so the app
/// never refuses to launch on a bad workspace file.
pub fn load(root: &Path) -> WorkspaceState {
    workspace_path(root)
        .map(|p| load_from(&p))
        .unwrap_or_default()
}

/// Path-addressed load. Extracted so unit tests can point at a tempdir
/// without having to mutate `$HOME`.
pub fn load_from(path: &Path) -> WorkspaceState {
    let Ok(contents) = fs::read_to_string(path) else {
        return WorkspaceState::default();
    };
    toml::from_str(&contents).unwrap_or_default()
}

/// Persist `state` for `root`. Creates `~/.gruff/workspaces/` if missing.
/// Callers should treat an error as non-fatal — the worst case is the
/// user's curation doesn't survive a relaunch; the app itself keeps working.
pub fn save(root: &Path, state: &WorkspaceState) -> io::Result<()> {
    let Some(path) = workspace_path(root) else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "HOME not set; cannot locate ~/.gruff/workspaces/",
        ));
    };
    save_to(&path, state)
}

/// Path-addressed save. Extracted for unit tests, same shape as
/// [`crate::config::save_to`].
pub fn save_to(path: &Path, state: &WorkspaceState) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body =
        toml::to_string_pretty(state).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, body)
}

/// Serde adapter: write the hide-set as a sorted vector and read it back
/// into a `HashSet`. Sorting keeps TOML output stable so the file only
/// churns when the hide-set actually changes.
mod hide_set_as_vec {
    use std::collections::HashSet;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::graph::NodeId;

    pub fn serialize<S: Serializer>(set: &HashSet<NodeId>, s: S) -> Result<S::Ok, S::Error> {
        let mut sorted: Vec<&NodeId> = set.iter().collect();
        sorted.sort();
        sorted.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<HashSet<NodeId>, D::Error> {
        let v: Vec<NodeId> = Vec::deserialize(d)?;
        Ok(v.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_state() -> WorkspaceState {
        let mut hidden = HashSet::new();
        hidden.insert("pkg/a/src/foo.ts".to_string());
        hidden.insert("pkg/b/src/bar.ts".to_string());
        WorkspaceState { hidden }
    }

    #[test]
    fn hash_is_stable_for_same_path() {
        // Sanity: hashing is deterministic across calls. If this ever breaks
        // every existing workspace file on disk becomes orphaned.
        let p = Path::new("/tmp/some/repo");
        assert_eq!(hash_workspace(p), hash_workspace(p));
    }

    #[test]
    fn hash_has_expected_length() {
        // 16 hex chars — the constant the decision doc pins.
        assert_eq!(hash_workspace(Path::new("/tmp/x")).len(), 16);
    }

    #[test]
    fn hash_differs_across_distinct_paths() {
        // Otherwise two unrelated repos would collide on the same TOML
        // and stomp each other's hide-sets.
        assert_ne!(
            hash_workspace(Path::new("/tmp/a")),
            hash_workspace(Path::new("/tmp/b"))
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hash_folds_case_on_macos() {
        // macOS filesystems are case-insensitive by default, so
        // `/tmp/Repo` and `/tmp/repo` refer to the same directory — the
        // hash must agree so the hide-set survives the case drift a user
        // might see when typing paths by hand vs. dragging a folder in.
        assert_eq!(
            hash_workspace(Path::new("/tmp/Repo")),
            hash_workspace(Path::new("/tmp/repo")),
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn hash_preserves_case_off_macos() {
        // Linux filesystems are case-sensitive, so `/tmp/Repo` and
        // `/tmp/repo` are distinct directories; their hashes must differ.
        assert_ne!(
            hash_workspace(Path::new("/tmp/Repo")),
            hash_workspace(Path::new("/tmp/repo")),
        );
    }

    #[test]
    fn load_from_missing_returns_default() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("absent.toml");
        assert_eq!(load_from(&path), WorkspaceState::default());
    }

    #[test]
    fn load_from_malformed_returns_default() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("broken.toml");
        fs::write(&path, "this is not toml ]]]").unwrap();
        assert_eq!(load_from(&path), WorkspaceState::default());
    }

    #[test]
    fn round_trip_preserves_hide_set() {
        // Save → load equality is the core correctness guarantee: the
        // hide-set is exactly what we put in, no duplicates, no drops.
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested").join("state.toml");
        let state = sample_state();
        save_to(&path, &state).unwrap();
        let reloaded = load_from(&path);
        assert_eq!(reloaded, state);
    }

    #[test]
    fn empty_hide_set_round_trips() {
        // The default — nothing hidden — must round-trip cleanly so
        // we never lose state by persisting "no curation yet".
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.toml");
        let state = WorkspaceState::default();
        save_to(&path, &state).unwrap();
        assert_eq!(load_from(&path), state);
    }

    #[test]
    fn serialized_hide_set_is_sorted() {
        // Deterministic output means touching a workspace's curation in
        // the same order two runs in a row produces byte-identical files.
        // Insert in reverse order and assert the TOML text is sorted.
        let dir = tempdir().unwrap();
        let path = dir.path().join("sorted.toml");
        let mut hidden = HashSet::new();
        hidden.insert("z".to_string());
        hidden.insert("a".to_string());
        hidden.insert("m".to_string());
        let state = WorkspaceState { hidden };
        save_to(&path, &state).unwrap();
        let body = fs::read_to_string(&path).unwrap();
        let pos_a = body.find("\"a\"").expect("a must appear");
        let pos_m = body.find("\"m\"").expect("m must appear");
        let pos_z = body.find("\"z\"").expect("z must appear");
        assert!(pos_a < pos_m && pos_m < pos_z, "body was: {body}");
    }

    #[test]
    fn workspace_path_is_under_gruff_workspaces_dir() {
        // Acceptance criterion from the PRD: everything lives under
        // `~/.gruff/workspaces/`. Mutate `$HOME` locally via the test
        // fixture and assert the resolved path's parent is the expected
        // directory. `HOME_GUARD` serialises this against other
        // HOME-mutating tests so `cargo test` stays stable at the
        // default thread count.
        let _guard = crate::test_support::HOME_GUARD
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let home = tempdir().unwrap();
        // SAFETY: `set_var` is not thread-safe per the Rust 2024
        // edition contract. The `HOME_GUARD` above serialises every
        // HOME-mutating test in the crate, and we don't spawn
        // background threads that could observe a torn read.
        unsafe {
            std::env::set_var("HOME", home.path());
        }
        let root = Path::new("/tmp/some/repo");
        let path = workspace_path(root).expect("HOME is set above");
        assert_eq!(
            path.parent().unwrap(),
            home.path().join(".gruff").join("workspaces"),
        );
        assert_eq!(
            path.file_name().unwrap().to_string_lossy(),
            format!("{}.toml", hash_workspace(root)),
        );
    }

    #[test]
    fn save_creates_parent_directory() {
        // `~/.gruff/workspaces/` likely doesn't exist on first save after
        // install. `save_to` must create it rather than surface an error —
        // matches `config::save_to`'s behavior on a fresh `$HOME`.
        let dir = tempdir().unwrap();
        let path = dir.path().join("deep").join("nested").join("state.toml");
        assert!(!path.parent().unwrap().exists());
        save_to(&path, &sample_state()).unwrap();
        assert!(path.exists());
    }
}
