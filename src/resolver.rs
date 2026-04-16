use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Extensions probed when resolving a relative import that has no extension on disk.
pub const CANDIDATE_EXTS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];

/// Resolve a relative import (`./foo`, `../bar`, `./foo/index`) from `from_file`
/// to a concrete file path, probing the standard JS/TS extensions and the
/// `index.*` convention for directory imports.
///
/// Returns `None` for non-relative specifiers (bare packages, absolute paths,
/// tsconfig aliases) — those are out of scope for this slice.
pub fn resolve_relative(from_file: &Path, import_source: &str) -> Option<PathBuf> {
    if !is_relative_specifier(import_source) {
        return None;
    }

    let from_dir = from_file.parent()?;
    let target = from_dir.join(import_source);

    if let Some(resolved) = probe_file(&target) {
        return Some(normalize(resolved));
    }

    if target.is_dir() {
        for ext in CANDIDATE_EXTS {
            let index = target.join(format!("index.{ext}"));
            if index.is_file() {
                return Some(normalize(index));
            }
        }
    }

    None
}

/// Lexically collapse `.` and `..` components without touching the filesystem.
fn normalize(p: PathBuf) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c.as_os_str()),
        }
    }
    out
}

fn is_relative_specifier(s: &str) -> bool {
    s == "." || s == ".." || s.starts_with("./") || s.starts_with("../")
}

fn probe_file(target: &Path) -> Option<PathBuf> {
    if target.is_file() {
        return Some(target.to_path_buf());
    }
    for ext in CANDIDATE_EXTS {
        let with_ext = append_ext(target, ext);
        if with_ext.is_file() {
            return Some(with_ext);
        }
    }
    None
}

fn append_ext(p: &Path, ext: &str) -> PathBuf {
    let mut os: OsString = p.as_os_str().into();
    os.push(".");
    os.push(ext);
    PathBuf::from(os)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn touch(dir: &Path, rel: &str) -> PathBuf {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, "").unwrap();
        path
    }

    #[test]
    fn returns_none_for_bare_specifier() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        assert!(resolve_relative(&from, "lodash").is_none());
        assert!(resolve_relative(&from, "@scope/pkg").is_none());
    }

    #[test]
    fn resolves_sibling_ts_file() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        let b = touch(dir.path(), "src/b.ts");
        assert_eq!(resolve_relative(&from, "./b"), Some(b));
    }

    #[test]
    fn resolves_sibling_tsx_file() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/App.tsx");
        let comp = touch(dir.path(), "src/Button.tsx");
        assert_eq!(resolve_relative(&from, "./Button"), Some(comp));
    }

    #[test]
    fn extension_priority_ts_over_js() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        let ts = touch(dir.path(), "src/shared.ts");
        touch(dir.path(), "src/shared.js");
        assert_eq!(resolve_relative(&from, "./shared"), Some(ts));
    }

    #[test]
    fn resolves_parent_directory_import() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/nested/a.ts");
        let target = touch(dir.path(), "src/util.ts");
        assert_eq!(resolve_relative(&from, "../util"), Some(target));
    }

    #[test]
    fn resolves_directory_index() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        let index = touch(dir.path(), "src/utils/index.ts");
        assert_eq!(resolve_relative(&from, "./utils"), Some(index));
    }

    #[test]
    fn resolves_literal_extension_if_file_exists() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        let target = touch(dir.path(), "src/data.json");
        assert_eq!(resolve_relative(&from, "./data.json"), Some(target));
    }

    #[test]
    fn returns_none_when_nothing_matches() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        assert!(resolve_relative(&from, "./missing").is_none());
    }
}
