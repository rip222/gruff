//! Persistent user settings loaded from `~/.gruff/config.toml`.
//!
//! The file is the only settings surface in v1 — there's no GUI for editing it.
//! A missing or malformed file silently falls back to defaults so the app can
//! still launch; callers are expected to write a fresh file on first save.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Config {
    #[serde(default)]
    pub editor: EditorConfig,
    #[serde(default)]
    pub watch: WatchConfig,
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
        };
        save_to(&path, &cfg).unwrap();
        let reloaded = load_from(&path);
        assert_eq!(cfg, reloaded);
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
}
