//! Entry-point discovery for the dead-code detector.
//!
//! An "entry point" is a file the workspace considers a root of reachability —
//! if unreachable from any entry point, a file is a candidate for being dead
//! code. Discovery is pure: it takes a [`Workspace`] + [`Config`] and returns
//! the set of graph [`NodeId`]s (canonical, workspace-relative paths) that
//! qualify.
//!
//! Four sources feed the result, unioned:
//!
//! 1. **`package.json`** — each workspace package contributes its `main`,
//!    `module`, `bin` (string or map), and `exports` (walked recursively) to
//!    the entry set. Paths are resolved relative to the package root, then
//!    canonicalised and converted to the same workspace-relative id the
//!    indexer uses.
//! 2. **Test files** — any file matching `**/*.test.{ts,tsx,js,jsx}`,
//!    `**/*.spec.*`, or `**/__tests__/**` under any package root.
//! 3. **Top-level config files** — build/linter/TS config files at the
//!    package root itself (e.g. `vite.config.ts`, `tsconfig.json`, dotfiles
//!    like `.eslintrc.*`).
//! 4. **User-declared globs** — [`Config::entry_points`] is a list of glob
//!    patterns applied against the workspace root.
//!
//! Declared entries (from `package.json` or user globs) that don't resolve to
//! any live node id are silently dropped — renames / deletes since the config
//! was written shouldn't manifest as ghost entries.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use glob::Pattern;
use serde_json::Value;

use crate::config::Config;
use crate::graph::NodeId;
use crate::workspace::Workspace;

/// Globs matched against each source file's workspace-relative path to decide
/// whether it's a test. Matches the PRD list verbatim: `**/*.test.{ts,tsx,js,jsx}`,
/// `**/*.spec.*`, and `**/__tests__/**`. Glob's pattern type doesn't do
/// brace-expansion, so brace groups are enumerated here.
const TEST_GLOBS: &[&str] = &[
    "**/*.test.ts",
    "**/*.test.tsx",
    "**/*.test.js",
    "**/*.test.jsx",
    "**/*.spec.ts",
    "**/*.spec.tsx",
    "**/*.spec.js",
    "**/*.spec.jsx",
    "**/__tests__/**",
];

/// Globs matched at the *package root* (not recursively) to decide whether a
/// file is a top-level config file. The PRD calls out "vite / webpack /
/// rollup / eslint / tsconfig / dotfiles"; this list covers the common shapes
/// without trying to enumerate every possible bundler config name.
const CONFIG_GLOBS: &[&str] = &[
    "vite.config.js",
    "vite.config.ts",
    "vite.config.mjs",
    "vite.config.cjs",
    "webpack.config.js",
    "webpack.config.ts",
    "webpack.config.mjs",
    "webpack.config.cjs",
    "rollup.config.js",
    "rollup.config.ts",
    "rollup.config.mjs",
    "rollup.config.cjs",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.mjs",
    ".eslintrc.json",
    ".prettierrc.js",
    ".prettierrc.cjs",
    ".prettierrc.json",
    "tsconfig.json",
    "tsconfig.*.json",
    "jest.config.js",
    "jest.config.ts",
    "jest.config.mjs",
    "jest.config.cjs",
    "babel.config.js",
    "babel.config.ts",
    "babel.config.mjs",
    "babel.config.cjs",
    ".babelrc.js",
    ".babelrc.json",
];

/// Discover every entry-point [`NodeId`] for the given workspace + user config.
///
/// Returns the union of:
/// * every declared entry in each package's `package.json` that resolves to
///   an on-disk file under the workspace (silently drops the rest);
/// * every file under any package root that matches [`TEST_GLOBS`];
/// * every file at a package root that matches [`CONFIG_GLOBS`];
/// * every file under the workspace root matching a
///   [`Config::entry_points`] glob.
///
/// The returned ids are in the same form as the indexer's
/// `relative_id(root, canonical)` — workspace-relative, forward-slashed — so
/// callers can directly pass the set to reachability / orphan-detection
/// routines whose key space is the graph's node ids.
pub fn discover(workspace: &Workspace, config: &Config) -> HashSet<NodeId> {
    let mut out: HashSet<NodeId> = HashSet::new();

    for pkg in &workspace.packages {
        // 1. package.json declared entries.
        for path in declared_entries_in(&pkg.manifest, &pkg.root) {
            if let Some(id) = to_workspace_id(&workspace.root, &path) {
                out.insert(id);
            }
        }

        // 2. Test files under the package root.
        collect_matching_under(&pkg.root, &workspace.root, TEST_GLOBS, &mut out);

        // 3. Top-level config files at the package root itself.
        collect_top_level_matching(&pkg.root, &workspace.root, CONFIG_GLOBS, &mut out);
    }

    // 4. User-declared globs, rooted at the workspace root.
    if !config.entry_points.is_empty() {
        let patterns: Vec<String> = config.entry_points.clone();
        let pattern_refs: Vec<&str> = patterns.iter().map(String::as_str).collect();
        collect_matching_under(&workspace.root, &workspace.root, &pattern_refs, &mut out);
    }

    out
}

/// Parse a single `package.json` and return every declared entry-point path
/// it references (as absolute paths under `pkg_root`). Missing fields are
/// treated as an empty contribution — this function never errors.
///
/// Handles the four shapes called out in the PRD:
/// * `main` — string
/// * `module` — string
/// * `bin` — string or `{ subcmd: path }` map
/// * `exports` — string, object with subpath keys, or nested condition keys;
///   walked recursively and every string leaf counts as an entry.
fn declared_entries_in(manifest: &Path, pkg_root: &Path) -> Vec<PathBuf> {
    let Ok(src) = std::fs::read_to_string(manifest) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<Value>(&src) else {
        return Vec::new();
    };

    let mut out: Vec<PathBuf> = Vec::new();

    if let Some(s) = v.get("main").and_then(Value::as_str) {
        out.push(pkg_root.join(s));
    }
    if let Some(s) = v.get("module").and_then(Value::as_str) {
        out.push(pkg_root.join(s));
    }
    if let Some(bin) = v.get("bin") {
        match bin {
            Value::String(s) => out.push(pkg_root.join(s)),
            Value::Object(map) => {
                for leaf in map.values() {
                    if let Some(s) = leaf.as_str() {
                        out.push(pkg_root.join(s));
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(exports) = v.get("exports") {
        walk_exports_strings(exports, pkg_root, &mut out);
    }

    out
}

/// Recursively collect every string leaf beneath an `exports` value,
/// resolving each against `pkg_root`. Treats any non-string, non-object value
/// as opaque. Handles:
/// * `"exports": "./index.js"`
/// * `"exports": { ".": "./index.js" }`
/// * `"exports": { ".": { "import": "./esm.js", "require": "./cjs.js" } }`
fn walk_exports_strings(v: &Value, pkg_root: &Path, out: &mut Vec<PathBuf>) {
    match v {
        Value::String(s) => out.push(pkg_root.join(s)),
        Value::Object(map) => {
            for leaf in map.values() {
                walk_exports_strings(leaf, pkg_root, out);
            }
        }
        _ => {}
    }
}

/// Walk every file under `scan_root` (skipping `node_modules`) and insert
/// workspace-relative ids for files whose path, relative to `scan_root`,
/// matches any of `patterns`.
fn collect_matching_under(
    scan_root: &Path,
    workspace_root: &Path,
    patterns: &[&str],
    out: &mut HashSet<NodeId>,
) {
    let compiled: Vec<Pattern> = patterns
        .iter()
        .filter_map(|p| Pattern::new(p).ok())
        .collect();
    if compiled.is_empty() {
        return;
    }
    let walker = ignore::WalkBuilder::new(scan_root)
        .follow_links(false)
        .require_git(false)
        .filter_entry(|e| e.file_name() != "node_modules")
        .build();
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = match path.strip_prefix(scan_root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if compiled
            .iter()
            .any(|p| p.matches(&rel_str) || p.matches_path(rel))
            && let Some(id) = to_workspace_id(workspace_root, path)
        {
            out.insert(id);
        }
    }
}

/// Match a set of patterns at the top level of `dir` only — no recursion, no
/// subdirectory traversal. Used for the config-file list which the PRD says
/// is "top-level only."
fn collect_top_level_matching(
    dir: &Path,
    workspace_root: &Path,
    patterns: &[&str],
    out: &mut HashSet<NodeId>,
) {
    let compiled: Vec<Pattern> = patterns
        .iter()
        .filter_map(|p| Pattern::new(p).ok())
        .collect();
    if compiled.is_empty() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if compiled.iter().any(|p| p.matches(name))
            && let Some(id) = to_workspace_id(workspace_root, &path)
        {
            out.insert(id);
        }
    }
}

/// Convert an absolute path to the indexer's node-id form — canonical, then
/// stripped of the workspace root, forward-slashed. Returns `None` when the
/// path doesn't exist on disk or can't be expressed relative to the root
/// (e.g. a declared entry pointing outside the package root).
///
/// Canonicalisation matches the indexer's `canonicalize_or` fallback: when a
/// path doesn't canonicalise directly we try canonicalising its parent so
/// still-reachable-but-just-deleted paths behave consistently. In practice
/// entry-point discovery runs after a full indexer scan, so the "parent
/// canonicalises but file is gone" case is rare.
fn to_workspace_id(workspace_root: &Path, path: &Path) -> Option<NodeId> {
    let canonical = canonicalize_or(path);
    // Drop declared entries that don't resolve to an existing file — the PRD
    // calls this out explicitly ("non-existent declared entries are silently
    // dropped"). We check after canonicalisation so a symlinked workspace
    // still resolves correctly.
    if !canonical.is_file() {
        return None;
    }
    let ws_canon = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let rel = canonical.strip_prefix(&ws_canon).ok()?;
    Some(rel.to_string_lossy().replace('\\', "/"))
}

/// Mirror of `indexer::canonicalize_or`. Kept private here rather than
/// re-exported to avoid widening the indexer's surface for one helper.
fn canonicalize_or(p: &Path) -> PathBuf {
    if let Ok(c) = p.canonicalize() {
        return c;
    }
    if let (Some(parent), Some(name)) = (p.parent(), p.file_name())
        && let Ok(cp) = parent.canonicalize()
    {
        return cp.join(name);
    }
    p.to_path_buf()
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

    /// A workspace with a single package at root, exposing only `main`. The
    /// declared main must surface as an entry point.
    #[test]
    fn single_package_with_main_only() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"app","main":"./src/index.ts"}"#);
        write(dir.path(), "src/index.ts", "export const x = 1;");

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        assert!(
            entries.contains("src/index.ts"),
            "declared main must be an entry, got {entries:?}"
        );
    }

    /// `module` + nested `exports` subpaths. Both direct string values and
    /// condition-keyed objects contribute their string leaves.
    #[test]
    fn package_with_module_and_exports_subpaths() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{
                "name": "app",
                "module": "./src/mod.ts",
                "exports": {
                    ".": { "import": "./src/esm.ts", "require": "./src/cjs.ts" },
                    "./sub": "./src/sub.ts"
                }
            }"#,
        );
        write(dir.path(), "src/mod.ts", "");
        write(dir.path(), "src/esm.ts", "");
        write(dir.path(), "src/cjs.ts", "");
        write(dir.path(), "src/sub.ts", "");

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        for id in ["src/mod.ts", "src/esm.ts", "src/cjs.ts", "src/sub.ts"] {
            assert!(entries.contains(id), "missing {id} in {entries:?}");
        }
    }

    /// `bin` as a bare string. The string path itself is the entry.
    #[test]
    fn bin_as_string() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{"name":"app","bin":"./cli.js"}"#,
        );
        write(dir.path(), "cli.js", "");

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        assert!(entries.contains("cli.js"));
    }

    /// `bin` as `{ subcmd: "./path" }` — every map value that's a string is an
    /// entry.
    #[test]
    fn bin_as_map() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{"name":"app","bin":{"app":"./cli/a.js","helper":"./cli/b.js"}}"#,
        );
        write(dir.path(), "cli/a.js", "");
        write(dir.path(), "cli/b.js", "");

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        assert!(entries.contains("cli/a.js"));
        assert!(entries.contains("cli/b.js"));
    }

    /// Test files caught by the three PRD globs surface as entries.
    #[test]
    fn test_files_are_entry_points() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"app"}"#);
        write(dir.path(), "src/a.test.ts", "");
        write(dir.path(), "src/b.spec.tsx", "");
        write(dir.path(), "src/__tests__/c.ts", "");
        // Non-test file under the same package — must not be an entry via
        // this rule.
        write(dir.path(), "src/lib.ts", "");

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        for id in ["src/a.test.ts", "src/b.spec.tsx", "src/__tests__/c.ts"] {
            assert!(entries.contains(id), "missing {id} in {entries:?}");
        }
        assert!(
            !entries.contains("src/lib.ts"),
            "non-test file must not be an entry via test globs",
        );
    }

    /// Top-level config files (and config-like dotfiles) match. A deeply
    /// nested `tsconfig.json` does NOT match as a config entry — the PRD
    /// scopes the glob to "top level."
    #[test]
    fn top_level_config_files_match() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"app"}"#);
        write(dir.path(), "vite.config.ts", "");
        write(dir.path(), "tsconfig.json", "{}");
        write(dir.path(), ".eslintrc.js", "");
        // Nested tsconfig — must not be swept up by the config rule.
        write(dir.path(), "nested/tsconfig.json", "{}");

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        for id in ["vite.config.ts", "tsconfig.json", ".eslintrc.js"] {
            assert!(entries.contains(id), "missing {id} in {entries:?}");
        }
        assert!(
            !entries.contains("nested/tsconfig.json"),
            "nested config file must not match the top-level config rule",
        );
    }

    /// User-declared globs from [`Config::entry_points`] surface matching
    /// files as entries.
    #[test]
    fn user_entry_points_globs_apply() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"app"}"#);
        write(dir.path(), "scripts/deploy.ts", "");
        write(dir.path(), "scripts/build.ts", "");
        write(dir.path(), "src/lib.ts", "");

        let ws = Workspace::discover(dir.path());
        let config = Config {
            entry_points: vec!["scripts/**/*.ts".to_string()],
            ..Config::default()
        };
        let entries = discover(&ws, &config);
        assert!(entries.contains("scripts/deploy.ts"));
        assert!(entries.contains("scripts/build.ts"));
        assert!(
            !entries.contains("src/lib.ts"),
            "files not covered by the user glob must not leak in",
        );
    }

    /// A declared entry pointing at a non-existent file is silently dropped
    /// instead of producing a ghost entry. Mirrors the PRD decision.
    #[test]
    fn nonexistent_declared_entry_is_dropped() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{"name":"app","main":"./does-not-exist.ts"}"#,
        );

        let ws = Workspace::discover(dir.path());
        let entries = discover(&ws, &Config::default());
        assert!(
            entries.iter().all(|id| id != "does-not-exist.ts"),
            "declared entry pointing nowhere must be silently dropped, got {entries:?}",
        );
    }
}
