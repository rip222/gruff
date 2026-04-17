//! Persistent user settings loaded from `~/.gruff/config.toml`.
//!
//! The file is the only settings surface in v1 — there's no GUI for editing it.
//! A missing or malformed file silently falls back to defaults so the app can
//! still launch; callers are expected to write a fresh file on first save.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::recents::RecentList;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Config {
    #[serde(default)]
    pub editor: EditorConfig,
    #[serde(default)]
    pub watch: WatchConfig,
    /// Path of the most recently opened repo. Re-opened on next launch if
    /// the path still exists and is readable. Layout, selection, and filter
    /// state are deliberately *not* persisted — each session starts fresh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_repo: Option<PathBuf>,
    /// MRU list of recently opened folders, surfaced by the
    /// `File → Open Recent ▸` submenu. Stored under `[recent]` in
    /// `config.toml`; an absent table deserializes to an empty list via the
    /// `#[serde(default)]` on both this field and `RecentList::paths`.
    #[serde(default)]
    pub recent: RecentList,
    /// Optional glob list the user can populate in `config.toml` to mark
    /// additional files as entry points for the dead-code detector (#36).
    /// Patterns are matched against workspace-relative paths; an empty list
    /// means "rely only on auto-discovered entry points." `#[serde(default)]`
    /// keeps the field absent-able — older configs load without this key.
    #[serde(default)]
    pub entry_points: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct EditorConfig {
    /// Editor command name — e.g. `code`, `vim`, `subl`. Empty means the user
    /// has not picked one yet; the app prompts on first open-in-editor attempt.
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct WatchConfig {
    /// Debounce window for the filesystem watcher (consumed in slice #10).
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: default_debounce_ms(),
        }
    }
}

fn default_debounce_ms() -> u64 {
    500
}

/// `~/.gruff/config.toml`. `None` only when `$HOME` is unset, which shouldn't
/// happen on macOS but we handle gracefully rather than panic.
pub fn config_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".gruff").join("config.toml"))
}

pub fn load() -> Config {
    config_path().map(|p| load_from(&p)).unwrap_or_default()
}

pub fn load_from(path: &Path) -> Config {
    let Ok(contents) = fs::read_to_string(path) else {
        return Config::default();
    };
    toml::from_str(&contents).unwrap_or_default()
}

pub fn save(cfg: &Config) -> io::Result<PathBuf> {
    let Some(path) = config_path() else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "HOME not set; cannot locate ~/.gruff/config.toml",
        ));
    };
    save_to(&path, cfg)?;
    Ok(path)
}

pub fn save_to(path: &Path, cfg: &Config) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = toml::to_string_pretty(cfg).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    fs::write(path, body)
}

/// Ensure the config file exists on disk, creating it with defaults if not.
/// Returns the path so callers can feed it to the editor.
pub fn ensure_exists() -> io::Result<PathBuf> {
    let Some(path) = config_path() else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "HOME not set; cannot locate ~/.gruff/config.toml",
        ));
    };
    if !path.exists() {
        save_to(&path, &Config::default())?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn load_from_missing_returns_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = load_from(&path);
        assert_eq!(cfg, Config::default());
        assert_eq!(cfg.watch.debounce_ms, 500);
        assert_eq!(cfg.editor.name, "");
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");
        let cfg = Config {
            editor: EditorConfig {
                name: "nvim".to_string(),
            },
            watch: WatchConfig { debounce_ms: 250 },
            last_repo: None,
            recent: RecentList::default(),
            entry_points: Vec::new(),
        };
        save_to(&path, &cfg).unwrap();
        let reloaded = load_from(&path);
        assert_eq!(cfg, reloaded);
    }

    #[test]
    fn last_repo_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config {
            last_repo: Some(PathBuf::from("/tmp/some/repo")),
            ..Config::default()
        };
        save_to(&path, &cfg).unwrap();
        let reloaded = load_from(&path);
        assert_eq!(reloaded.last_repo, Some(PathBuf::from("/tmp/some/repo")));
    }

    #[test]
    fn last_repo_absent_when_not_set() {
        // The `skip_serializing_if` attribute keeps the config file clean when
        // no repo has been opened yet — no stray `last_repo = ""` line.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        save_to(&path, &Config::default()).unwrap();
        let body = fs::read_to_string(&path).unwrap();
        assert!(!body.contains("last_repo"), "body was: {body}");
    }

    #[test]
    fn hand_edited_file_is_honored() {
        // Matches the "config file round-trips: edits made by hand are honored
        // on next launch" acceptance criterion.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
[editor]
name = "code"

[watch]
debounce_ms = 750
"#,
        )
        .unwrap();
        let cfg = load_from(&path);
        assert_eq!(cfg.editor.name, "code");
        assert_eq!(cfg.watch.debounce_ms, 750);
    }

    #[test]
    fn partial_file_fills_missing_fields_with_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[editor]\nname = \"vim\"\n").unwrap();
        let cfg = load_from(&path);
        assert_eq!(cfg.editor.name, "vim");
        assert_eq!(cfg.watch.debounce_ms, 500);
    }

    #[test]
    fn malformed_file_falls_back_to_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "this is not toml ]]]").unwrap();
        assert_eq!(load_from(&path), Config::default());
    }

    #[test]
    fn missing_recent_table_loads_empty_list() {
        // An older config.toml written before the `[recent]` table existed
        // must still load without error — `#[serde(default)]` fills the
        // field with an empty `RecentList`.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
[editor]
name = "code"

[watch]
debounce_ms = 500
"#,
        )
        .unwrap();
        let cfg = load_from(&path);
        assert!(cfg.recent.is_empty(), "recent should default to empty");
    }

    #[test]
    fn recent_round_trips_preserving_order_and_cap() {
        // Save → load → save equality: order, content, and the cap all
        // survive a TOML round-trip.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut cfg = Config::default();
        // Push more than the cap to force eviction before saving.
        for i in 0..7 {
            cfg.recent.push(PathBuf::from(format!("/tmp/r{i}")));
        }
        assert_eq!(cfg.recent.len(), crate::recents::MAX_RECENTS);
        save_to(&path, &cfg).unwrap();
        let reloaded = load_from(&path);
        assert_eq!(
            reloaded, cfg,
            "TOML round-trip must preserve recent list exactly"
        );
        // Sanity: top-of-list is the most-recent push.
        let first: Vec<_> = reloaded.recent.iter_excluding(None).collect();
        assert_eq!(first[0], PathBuf::from("/tmp/r6").as_path());
    }

    #[test]
    fn entry_points_default_to_empty_list() {
        // Absent `entry_points` key in the config loads as an empty vec, the
        // same way every other opt-in list in this file behaves. Older
        // configs written before the dead-code detector existed must keep
        // loading without error.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
[editor]
name = "code"
"#,
        )
        .unwrap();
        let cfg = load_from(&path);
        assert!(
            cfg.entry_points.is_empty(),
            "entry_points should default to empty list when the key is absent",
        );
    }

    #[test]
    fn populated_entry_points_round_trip() {
        // Save → load → equality on a realistic glob list. Preserves order,
        // content, and the fact that entries are plain strings (no URL
        // escaping, no path normalisation, no surprise unification).
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config {
            entry_points: vec![
                "scripts/**/*.ts".to_string(),
                "tools/**/main.ts".to_string(),
            ],
            ..Config::default()
        };
        save_to(&path, &cfg).unwrap();
        let reloaded = load_from(&path);
        assert_eq!(cfg, reloaded);
        assert_eq!(
            reloaded.entry_points,
            vec![
                "scripts/**/*.ts".to_string(),
                "tools/**/main.ts".to_string(),
            ],
            "round-trip must preserve both order and content",
        );
    }

    #[test]
    fn malformed_entry_points_falls_back_silently() {
        // A hand-edited `entry_points` value of the wrong TOML type (e.g. a
        // string instead of an array) must fall back to the whole-config
        // default rather than panic — same policy as the other sections.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
entry_points = "not-an-array"
"#,
        )
        .unwrap();
        let cfg = load_from(&path);
        assert_eq!(cfg, Config::default());
        assert!(cfg.entry_points.is_empty());
    }

    #[test]
    fn malformed_recent_falls_back_silently() {
        // A hand-edited file with a garbled `[recent]` value must fall back
        // to the whole-config default rather than panic. Mirrors the
        // existing "malformed → defaults" policy for other sections.
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r#"
[recent]
paths = "not-an-array-of-paths"
"#,
        )
        .unwrap();
        let cfg = load_from(&path);
        assert_eq!(cfg, Config::default());
        assert!(cfg.recent.is_empty());
    }
}
