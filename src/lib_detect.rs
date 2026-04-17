//! Language-agnostic lib-root detection.
//!
//! A "lib" is a sub-unit of a workspace package that has its own modularity
//! boundary — for TypeScript, any folder with its own `tsconfig.json`. The
//! [`LibDetector`] trait surfaces the set of lib roots in a workspace; a node
//! is assigned to the *deepest* lib root that is an ancestor of its path.
//!
//! This is the lib equivalent of [`crate::aggregation::NodeAggregator`]:
//! ecosystems plug in their own detector (Rust → `Cargo.toml`, Python →
//! `pyproject.toml`, etc.) without touching color resolution or layout.
//!
//! The trait itself doesn't touch the filesystem — the TypeScript impl
//! consumes the tsconfig paths already collected by [`Workspace::discover`].

use std::path::{Path, PathBuf};

use crate::workspace::Workspace;

/// One detected lib root. `path` is the folder containing the lib signal (for
/// TS, the folder holding the `tsconfig.json`). `package` is the name of the
/// workspace package that owns the folder, or `None` if the folder lies
/// outside every declared package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibRoot {
    pub path: PathBuf,
    pub package: Option<String>,
}

/// Discover a workspace's lib roots. Implementations are language-specific:
/// [`TsLibDetector`] treats any folder with its own `tsconfig.json` as a lib.
pub trait LibDetector {
    fn detect(&self, ws: &Workspace) -> Vec<LibRoot>;
}

/// TypeScript lib detector: every `tsconfig.json` location in the workspace
/// is a lib root.
#[derive(Debug, Default)]
pub struct TsLibDetector;

impl TsLibDetector {
    pub fn new() -> Self {
        Self
    }
}

impl LibDetector for TsLibDetector {
    fn detect(&self, ws: &Workspace) -> Vec<LibRoot> {
        let mut out: Vec<LibRoot> = Vec::new();
        for ts in &ws.tsconfigs {
            let Some(dir) = ts.parent() else { continue };
            let dir = dir.to_path_buf();
            let package = ws
                .packages
                .iter()
                .filter(|p| dir.starts_with(&p.root))
                .max_by_key(|p| p.root.components().count())
                .map(|p| p.name.clone());
            out.push(LibRoot {
                path: dir,
                package,
            });
        }
        // Deepest-first ordering makes `deepest_lib_for` a straight linear
        // scan, and keeps the output order stable for tests.
        out.sort_by(|a, b| {
            b.path
                .components()
                .count()
                .cmp(&a.path.components().count())
                .then_with(|| a.path.cmp(&b.path))
        });
        out
    }
}

/// Deepest lib root that is an ancestor of — or equal to — `path`. Returns the
/// lib's index in `libs` so the caller can key color assignment off it.
///
/// `path` may be a file (the lib root's folder is an ancestor) or a folder
/// (a barrel display node's folder path). Equality counts as ancestor so the
/// lib root itself maps to its own lib.
pub fn deepest_lib_for(path: &Path, libs: &[LibRoot]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (i, lib) in libs.iter().enumerate() {
        if path.starts_with(&lib.path) {
            let depth = lib.path.components().count();
            if best.map(|(_, d)| depth > d).unwrap_or(true) {
                best = Some((i, depth));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Group lib indices by their owning package. The returned map has one entry
/// per package name that has at least one lib, with the value being the
/// ordered list of lib indices into `libs` that belong to that package. Used
/// by color assignment to turn a global lib index into a
/// `(lib_index_within_package, lib_count_in_package)` pair.
pub fn libs_by_package(libs: &[LibRoot]) -> std::collections::HashMap<String, Vec<usize>> {
    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    // Order libs within a package by path so the lib-index-within-package is
    // stable across runs (the detector itself returns deepest-first, which
    // mixes different packages).
    let mut indexed: Vec<(usize, &LibRoot)> = libs.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| a.path.cmp(&b.path));
    for (i, lib) in indexed {
        if let Some(pkg) = &lib.package {
            map.entry(pkg.clone()).or_default().push(i);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(dir: &Path, rel: &str, contents: &str) -> PathBuf {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn detects_root_tsconfig_as_sole_lib() {
        // Bare repo with a single tsconfig — one lib root at the root, owning
        // package is whatever the root package.json declares.
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"solo"}"#);
        write(dir.path(), "tsconfig.json", "{}");

        let ws = Workspace::discover(dir.path());
        let libs = TsLibDetector::new().detect(&ws);
        assert_eq!(libs.len(), 1);
        assert_eq!(libs[0].package.as_deref(), Some("solo"));
    }

    #[test]
    fn detects_nested_tsconfigs_as_distinct_libs() {
        // Package `a` has a root tsconfig plus a nested one inside
        // `a/internal/` — both are lib roots.
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        write(dir.path(), "packages/a/tsconfig.json", "{}");
        write(dir.path(), "packages/a/internal/tsconfig.json", "{}");

        let ws = Workspace::discover(dir.path());
        let libs = TsLibDetector::new().detect(&ws);

        // Two libs belong to package `a`.
        let a_libs: Vec<_> = libs.iter().filter(|l| l.package.as_deref() == Some("a")).collect();
        assert_eq!(a_libs.len(), 2);
    }

    #[test]
    fn file_maps_to_deepest_enclosing_lib() {
        // File at `packages/a/internal/x.ts` must map to the `internal`
        // lib, not the outer package-root lib.
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        write(dir.path(), "packages/a/tsconfig.json", "{}");
        write(dir.path(), "packages/a/internal/tsconfig.json", "{}");
        let file = write(dir.path(), "packages/a/internal/x.ts", "");

        let ws = Workspace::discover(dir.path());
        let libs = TsLibDetector::new().detect(&ws);
        let canonical = file.canonicalize().unwrap();
        let idx = deepest_lib_for(&canonical, &libs).expect("file should map to a lib");
        assert!(
            libs[idx].path.ends_with("internal"),
            "expected internal lib, got {:?}",
            libs[idx].path,
        );
    }

    #[test]
    fn file_outside_any_tsconfig_has_no_lib() {
        // Loose file under `packages/a/` with no tsconfig anywhere — no lib
        // assignment, caller falls back to package_color.
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        let file = write(dir.path(), "packages/a/src/x.ts", "");

        let ws = Workspace::discover(dir.path());
        let libs = TsLibDetector::new().detect(&ws);
        let canonical = file.canonicalize().unwrap();
        assert!(deepest_lib_for(&canonical, &libs).is_none());
    }

    #[test]
    fn libs_by_package_groups_and_orders_within_package() {
        // Two libs under package `a` plus one under `b`. `libs_by_package`
        // must return a stable per-package order so the "lib index within
        // package" we feed into the color palette is deterministic.
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        write(dir.path(), "packages/a/tsconfig.json", "{}");
        write(dir.path(), "packages/a/sub/tsconfig.json", "{}");
        write(dir.path(), "packages/b/package.json", r#"{"name":"b"}"#);
        write(dir.path(), "packages/b/tsconfig.json", "{}");

        let ws = Workspace::discover(dir.path());
        let libs = TsLibDetector::new().detect(&ws);
        let grouped = libs_by_package(&libs);
        assert_eq!(grouped.get("a").map(|v| v.len()), Some(2));
        assert_eq!(grouped.get("b").map(|v| v.len()), Some(1));

        // Stability: lib order within `a` must be by path (ascending), so
        // `packages/a` comes before `packages/a/sub`.
        let a_indices = grouped.get("a").unwrap();
        let a_paths: Vec<&std::path::Path> =
            a_indices.iter().map(|&i| libs[i].path.as_path()).collect();
        assert!(a_paths[0].components().count() <= a_paths[1].components().count());
    }

    #[test]
    fn pre_existing_package_tsconfig_surfaces_as_first_class_lib() {
        // Guard against regressing the "workspace already collected this
        // tsconfig at the package root" behavior — the detector must still
        // return that tsconfig's folder as a lib root.
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"solo"}"#);
        write(dir.path(), "tsconfig.json", "{}");

        let ws = Workspace::discover(dir.path());
        assert!(!ws.tsconfigs.is_empty(), "precondition: tsconfig recorded");
        let libs = TsLibDetector::new().detect(&ws);
        assert_eq!(libs.len(), 1);
        assert_eq!(libs[0].package.as_deref(), Some("solo"));
    }
}
