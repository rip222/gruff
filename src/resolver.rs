use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Extensions probed when resolving a relative import that has no extension on disk.
pub const CANDIDATE_EXTS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];

/// Outcome of resolving an import specifier.
///
/// `WorkspaceFile` is the in-repo target a relative import landed on.
/// `External` is the bare specifier's owning `node_modules` package (e.g.
/// `lodash/fp` → `"lodash"`, `@scope/pkg/sub` → `"@scope/pkg"`), reported by
/// package name only — the indexer doesn't walk into `node_modules`.
/// `Unresolved` is for specifiers we can't place (unreadable path, non-existent
/// relative target, empty specifier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedImport {
    WorkspaceFile(PathBuf),
    External(String),
    Unresolved,
}

/// Resolve an import specifier from `from_file` into a [`ResolvedImport`].
///
/// Relative specifiers (`./foo`, `../bar`) go through [`resolve_relative`];
/// bare specifiers (`react`, `@scope/pkg`, `lodash/fp`) are reported as
/// `External(<package-name>)` without touching the filesystem. Absolute paths
/// and empty strings are currently reported as `Unresolved`.
pub fn resolve(from_file: &Path, import_source: &str) -> ResolvedImport {
    if import_source.is_empty() {
        return ResolvedImport::Unresolved;
    }
    if is_relative_specifier(import_source) {
        return match resolve_relative(from_file, import_source) {
            Some(p) => ResolvedImport::WorkspaceFile(p),
            None => ResolvedImport::Unresolved,
        };
    }
    match bare_specifier_package(import_source) {
        Some(name) => ResolvedImport::External(name),
        None => ResolvedImport::Unresolved,
    }
}

/// Extract the `node_modules` package name from a bare import specifier.
///
/// - `react` → `"react"`
/// - `lodash/fp` → `"lodash"` (subpath stripped)
/// - `@scope/pkg` → `"@scope/pkg"` (scope kept with the next segment)
/// - `@scope/pkg/deep/sub` → `"@scope/pkg"`
///
/// Returns `None` for anything that isn't a bare specifier (absolute paths,
/// protocol URLs, empty strings, malformed `@scope` specifiers).
pub fn bare_specifier_package(s: &str) -> Option<String> {
    if s.is_empty() || is_relative_specifier(s) {
        return None;
    }
    // Protocol URLs (`node:fs`, `http://...`) and Windows-style absolute paths
    // aren't package specifiers — surface them as unresolved rather than
    // pretending the prefix before `:` is a package.
    if s.contains(':') || s.starts_with('/') || s.starts_with('\\') {
        return None;
    }

    if let Some(rest) = s.strip_prefix('@') {
        // Scoped: `@scope/name[/sub...]`. Must have at least scope + name.
        let mut parts = rest.splitn(3, '/');
        let scope = parts.next()?;
        let name = parts.next()?;
        if scope.is_empty() || name.is_empty() {
            return None;
        }
        return Some(format!("@{scope}/{name}"));
    }

    // Unscoped: take the first path segment as the package name.
    let name = s.split('/').next()?;
    if name.is_empty() {
        return None;
    }
    Some(name.to_string())
}

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

    // --- bare-specifier extraction -----------------------------------------

    #[test]
    fn bare_unscoped_package_returns_name_as_is() {
        assert_eq!(bare_specifier_package("react"), Some("react".to_string()));
        assert_eq!(
            bare_specifier_package("lodash-es"),
            Some("lodash-es".to_string())
        );
    }

    #[test]
    fn bare_unscoped_subpath_trimmed_to_package_name() {
        assert_eq!(bare_specifier_package("lodash/fp"), Some("lodash".to_string()));
        assert_eq!(
            bare_specifier_package("lodash/fp/get"),
            Some("lodash".to_string())
        );
    }

    #[test]
    fn bare_scoped_package_keeps_scope() {
        assert_eq!(
            bare_specifier_package("@scope/pkg"),
            Some("@scope/pkg".to_string())
        );
        assert_eq!(
            bare_specifier_package("@myorg/foo"),
            Some("@myorg/foo".to_string())
        );
    }

    #[test]
    fn bare_scoped_subpath_trimmed_to_scope_and_name() {
        assert_eq!(
            bare_specifier_package("@scope/pkg/deep"),
            Some("@scope/pkg".to_string())
        );
        assert_eq!(
            bare_specifier_package("@scope/pkg/deep/more"),
            Some("@scope/pkg".to_string())
        );
    }

    #[test]
    fn bare_rejects_relative_and_malformed_specifiers() {
        assert_eq!(bare_specifier_package(""), None);
        assert_eq!(bare_specifier_package("./foo"), None);
        assert_eq!(bare_specifier_package("../bar"), None);
        // Missing name after scope — not a legal package specifier.
        assert_eq!(bare_specifier_package("@scope"), None);
        assert_eq!(bare_specifier_package("@scope/"), None);
        // `node:` protocol (Node builtins) — not a node_modules package.
        assert_eq!(bare_specifier_package("node:fs"), None);
        // Absolute filesystem paths — never a package specifier.
        assert_eq!(bare_specifier_package("/etc/passwd"), None);
    }

    // --- full resolve() routing -------------------------------------------

    #[test]
    fn resolve_routes_relative_to_workspace_file() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        let b = touch(dir.path(), "src/b.ts");
        assert_eq!(resolve(&from, "./b"), ResolvedImport::WorkspaceFile(b));
    }

    #[test]
    fn resolve_routes_bare_to_external() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        assert_eq!(resolve(&from, "react"), ResolvedImport::External("react".to_string()));
        assert_eq!(
            resolve(&from, "@scope/pkg/deep"),
            ResolvedImport::External("@scope/pkg".to_string())
        );
        assert_eq!(
            resolve(&from, "lodash/fp"),
            ResolvedImport::External("lodash".to_string())
        );
    }

    #[test]
    fn resolve_unresolved_for_missing_relative() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        assert_eq!(resolve(&from, "./missing"), ResolvedImport::Unresolved);
    }

    #[test]
    fn resolve_unresolved_for_protocol_or_empty() {
        let dir = tempdir().unwrap();
        let from = touch(dir.path(), "src/a.ts");
        assert_eq!(resolve(&from, ""), ResolvedImport::Unresolved);
        assert_eq!(resolve(&from, "node:fs"), ResolvedImport::Unresolved);
    }
}
