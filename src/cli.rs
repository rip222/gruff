//! Pure command-line argument parsing for `gruff <path>`.
//!
//! No flags, one optional positional. Relative paths resolve against the
//! caller-supplied `cwd` so this module stays free of process-global state and
//! is fully testable without a real working directory. Invalid paths never
//! fall back to `last_repo` — they surface via [`CliOutcome::InvalidPath`] so
//! the UI can land on onboarding with the error in the status bar.

use std::path::{Path, PathBuf};

/// What `main.rs` should do after parsing argv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliOutcome {
    /// No path argument — resume last-opened repo via the existing config flow.
    Autoload,
    /// User passed a path that exists and is a directory; open it directly.
    /// Always canonicalized so `.` / `..` / symlinks collapse before the rest
    /// of the app sees the path.
    OpenPath(PathBuf),
    /// Path was provided but couldn't be opened as a directory. The raw input
    /// is preserved so the status-bar message can quote exactly what the user
    /// typed; `reason` is a short human-readable explanation.
    InvalidPath { raw: String, reason: String },
}

/// Parse `argv` (the full process args, `argv[0]` is the program name) against
/// `cwd`. Pure — no filesystem writes, no env reads; the caller owns both
/// inputs. Returns exactly one outcome per invocation.
///
/// Contract:
/// - `argv.len() <= 1` → [`CliOutcome::Autoload`] (bare `gruff`).
/// - empty-string path → [`CliOutcome::InvalidPath`] (no silent fallback).
/// - path that canonicalizes and is a directory → [`CliOutcome::OpenPath`].
/// - anything else (missing, file, unreadable) → [`CliOutcome::InvalidPath`].
pub fn parse_args(argv: &[String], cwd: &Path) -> CliOutcome {
    let Some(raw) = argv.get(1) else {
        return CliOutcome::Autoload;
    };

    if raw.is_empty() {
        return CliOutcome::InvalidPath {
            raw: raw.clone(),
            reason: "path is empty".to_string(),
        };
    }

    let candidate = Path::new(raw);
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        cwd.join(candidate)
    };

    // `canonicalize` requires the target to exist, so a failure here doubles as
    // our "nonexistent or unreadable" detector per the decision doc.
    let resolved = match std::fs::canonicalize(&absolute) {
        Ok(p) => p,
        Err(e) => {
            return CliOutcome::InvalidPath {
                raw: raw.clone(),
                reason: e.to_string(),
            };
        }
    };

    if !resolved.is_dir() {
        return CliOutcome::InvalidPath {
            raw: raw.clone(),
            reason: "not a directory".to_string(),
        };
    }

    CliOutcome::OpenPath(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Wrap a list of string literals in the `[String]` shape `parse_args`
    /// expects, with a dummy `argv[0]` so the caller's intent matches the
    /// real `std::env::args()` layout.
    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("gruff")
            .chain(args.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn no_arg_requests_autoload() {
        let cwd = tempdir().unwrap();
        assert_eq!(
            parse_args(&argv(&[]), cwd.path()),
            CliOutcome::Autoload,
            "bare `gruff` must fall through to the existing autoload flow"
        );
    }

    #[test]
    fn absolute_existing_dir_is_opened_canonicalized() {
        let dir = tempdir().unwrap();
        let abs = dir.path().to_path_buf();
        let cwd = tempdir().unwrap();

        let outcome = parse_args(&argv(&[abs.to_str().unwrap()]), cwd.path());
        match outcome {
            CliOutcome::OpenPath(p) => {
                // Canonicalize both sides — on macOS /tmp is a symlink to
                // /private/tmp, so the direct equality check would fail.
                assert_eq!(p, fs::canonicalize(&abs).unwrap());
            }
            other => panic!("expected OpenPath, got {other:?}"),
        }
    }

    #[test]
    fn relative_path_resolves_against_passed_cwd() {
        let cwd = tempdir().unwrap();
        let sub = cwd.path().join("sub");
        fs::create_dir(&sub).unwrap();

        let outcome = parse_args(&argv(&["sub"]), cwd.path());
        match outcome {
            CliOutcome::OpenPath(p) => {
                assert_eq!(p, fs::canonicalize(&sub).unwrap());
            }
            other => panic!("expected OpenPath, got {other:?}"),
        }
    }

    #[test]
    fn dot_resolves_to_cwd() {
        let cwd = tempdir().unwrap();
        let outcome = parse_args(&argv(&["."]), cwd.path());
        match outcome {
            CliOutcome::OpenPath(p) => {
                assert_eq!(p, fs::canonicalize(cwd.path()).unwrap());
            }
            other => panic!("expected OpenPath, got {other:?}"),
        }
    }

    #[test]
    fn dotdot_resolves_to_parent_of_cwd() {
        let parent = tempdir().unwrap();
        let child = parent.path().join("child");
        fs::create_dir(&child).unwrap();

        let outcome = parse_args(&argv(&[".."]), &child);
        match outcome {
            CliOutcome::OpenPath(p) => {
                assert_eq!(p, fs::canonicalize(parent.path()).unwrap());
            }
            other => panic!("expected OpenPath, got {other:?}"),
        }
    }

    #[test]
    fn nonexistent_path_is_invalid() {
        let cwd = tempdir().unwrap();
        let missing = cwd.path().join("does-not-exist");
        let outcome = parse_args(&argv(&[missing.to_str().unwrap()]), cwd.path());
        match outcome {
            CliOutcome::InvalidPath { raw, reason } => {
                assert_eq!(raw, missing.to_str().unwrap());
                assert!(!reason.is_empty(), "reason must not be empty");
            }
            other => panic!("expected InvalidPath, got {other:?}"),
        }
    }

    #[test]
    fn file_path_is_invalid() {
        let cwd = tempdir().unwrap();
        let file = cwd.path().join("a.txt");
        fs::write(&file, "hi").unwrap();

        let outcome = parse_args(&argv(&[file.to_str().unwrap()]), cwd.path());
        match outcome {
            CliOutcome::InvalidPath { raw, reason } => {
                assert_eq!(raw, file.to_str().unwrap());
                assert!(
                    reason.contains("not a directory"),
                    "expected directory-check failure, got {reason:?}"
                );
            }
            other => panic!("expected InvalidPath, got {other:?}"),
        }
    }

    #[test]
    fn empty_string_arg_is_invalid() {
        let cwd = tempdir().unwrap();
        let outcome = parse_args(&argv(&[""]), cwd.path());
        match outcome {
            CliOutcome::InvalidPath { raw, reason } => {
                assert_eq!(raw, "");
                assert!(!reason.is_empty());
            }
            other => panic!("expected InvalidPath, got {other:?}"),
        }
    }
}
