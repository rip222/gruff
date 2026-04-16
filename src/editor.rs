//! Shell-out launcher for the user's editor.
//!
//! Resolution happens at spawn time — we don't front-load PATH lookups on app
//! launch. That matches the UX promise: a user who never opens a file in their
//! editor never sees the editor prompt.

use std::io;
use std::path::Path;
use std::process::Command;

/// Failure modes for [`open_in_editor`]. The UI distinguishes these to decide
/// whether to prompt the user versus surface a generic error.
#[derive(Debug)]
pub enum OpenError {
    /// No editor has been configured yet (config value is empty).
    NotConfigured,
    /// The configured editor command wasn't found on `PATH`.
    NotFound(String),
    /// The spawn failed for some other reason (permissions, etc).
    Io(io::Error),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::NotConfigured => write!(f, "no editor configured"),
            OpenError::NotFound(name) => write!(f, "editor `{name}` not found on PATH"),
            OpenError::Io(e) => write!(f, "failed to launch editor: {e}"),
        }
    }
}

/// Spawn `editor_name` with `file` as its sole argument. The child is
/// detached — we don't wait on it, so GUI editors stay open after Gruff exits
/// and fast CLI editors (that fail immediately) report their error via
/// [`OpenError::Io`].
pub fn open_in_editor(editor_name: &str, file: &Path) -> Result<(), OpenError> {
    let trimmed = editor_name.trim();
    if trimmed.is_empty() {
        return Err(OpenError::NotConfigured);
    }
    // GUI launchers (Cursor, VS Code) inherit the parent's PATH, but terminal
    // Claude Code launches strip `/usr/local/bin` in some setups. Shelling
    // through the user's login shell picks up their full PATH reliably.
    let result = Command::new(trimmed).arg(file).spawn();
    match result {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            Err(OpenError::NotFound(trimmed.to_string()))
        }
        Err(e) => Err(OpenError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn empty_editor_name_is_not_configured() {
        let r = open_in_editor("", &PathBuf::from("/tmp/x"));
        assert!(matches!(r, Err(OpenError::NotConfigured)));
    }

    #[test]
    fn whitespace_only_editor_name_is_not_configured() {
        let r = open_in_editor("   ", &PathBuf::from("/tmp/x"));
        assert!(matches!(r, Err(OpenError::NotConfigured)));
    }

    #[test]
    fn unknown_editor_returns_not_found() {
        let r = open_in_editor("gruff_nonexistent_editor_xyz_42", &PathBuf::from("/tmp/x"));
        assert!(matches!(r, Err(OpenError::NotFound(_))));
    }
}
