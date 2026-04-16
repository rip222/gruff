use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::parser::{ImportKind, ImportStatement};
use crate::workspace::Workspace;

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
/// Bare-bones helper used by call sites that don't care about tsconfig paths
/// or workspace aliases. The indexer goes through [`resolve_import`] instead,
/// which threads workspace context for the richer resolution modes.
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

/// Parsed `tsconfig.json` slice the resolver consults for `paths` aliases.
///
/// Only the fields we resolve against are kept (`baseUrl` and `paths`); the
/// rest of the file is ignored. Comments are stripped before parsing because
/// `tsconfig.json` is JSONC.
#[derive(Debug, Clone)]
pub struct Tsconfig {
    /// Directory containing the `tsconfig.json`.
    pub dir: PathBuf,
    /// Resolved `compilerOptions.baseUrl` (defaults to `dir`).
    pub base_url: PathBuf,
    /// `(pattern, substitutions)` pairs from `compilerOptions.paths`. Both
    /// the pattern and each substitution may contain a single `*` wildcard.
    pub paths: Vec<(String, Vec<String>)>,
}

/// Per-workspace lookup tables built once per index, threaded through every
/// per-import [`resolve_import`] call.
///
/// Holds tsconfig parses keyed by their on-disk path, plus precomputed
/// workspace-package roots and entry files so the resolver can answer
/// "does `@org/shared` mean a workspace package?" in O(1).
#[derive(Debug, Default)]
pub struct ResolverContext {
    /// Parsed tsconfigs keyed by the path the workspace recorded for them.
    pub tsconfigs: HashMap<PathBuf, Tsconfig>,
    /// `package_name → entry file on disk`. Only populated for packages whose
    /// entry resolved successfully — packages with no usable entry can still
    /// service subpath imports through `package_roots`.
    pub package_entries: HashMap<String, PathBuf>,
    /// `package_name → package root directory`. Used for subpath imports like
    /// `@org/shared/utils` (split into pkg + subpath, resolved under root).
    pub package_roots: HashMap<String, PathBuf>,
}

impl ResolverContext {
    /// Build a context from a discovered workspace: parse every tsconfig the
    /// workspace surfaced and pre-resolve each package's entry file.
    pub fn build(ws: &Workspace) -> Self {
        let mut tsconfigs = HashMap::new();
        for tsc_path in &ws.tsconfigs {
            if let Some(cfg) = parse_tsconfig(tsc_path) {
                tsconfigs.insert(tsc_path.clone(), cfg);
            }
        }

        let mut package_entries = HashMap::new();
        let mut package_roots = HashMap::new();
        for pkg in &ws.packages {
            package_roots.insert(pkg.name.clone(), pkg.root.clone());
            if let Some(entry) = find_package_entry(&pkg.root, &pkg.manifest) {
                package_entries.insert(pkg.name.clone(), entry);
            }
        }

        Self {
            tsconfigs,
            package_entries,
            package_roots,
        }
    }
}

/// Resolve a single [`ImportStatement`] into zero or more concrete targets.
///
/// Handles every resolution mode the parser can produce: relative paths,
/// tsconfig `paths` aliases, workspace package aliases (by `package.json`
/// `name`), bare `node_modules` specifiers, and template-prefix dynamic
/// imports (which expand to a directory glob → many edges).
///
/// Returns an empty vec when the import is fully unresolvable (e.g.
/// `import(modName)` with `modName` only known at runtime). The caller uses
/// "empty for a `Dynamic` kind" as the trigger for the unresolved-dynamic
/// status-bar count.
pub fn resolve_import(
    ctx: &ResolverContext,
    ws: &Workspace,
    from_file: &Path,
    imp: &ImportStatement,
) -> Vec<ResolvedImport> {
    if imp.is_unresolvable {
        return Vec::new();
    }

    if imp.kind == ImportKind::Dynamic && imp.is_template {
        return resolve_template_prefix(from_file, &imp.source);
    }

    let source = imp.source.as_str();
    if source.is_empty() {
        return vec![ResolvedImport::Unresolved];
    }

    if is_relative_specifier(source) {
        return match resolve_relative(from_file, source) {
            Some(p) => vec![ResolvedImport::WorkspaceFile(p)],
            None => vec![ResolvedImport::Unresolved],
        };
    }

    if let Some(p) = resolve_workspace_alias(ctx, source) {
        return vec![ResolvedImport::WorkspaceFile(p)];
    }

    if let Some(p) = resolve_tsconfig_path(ctx, ws, from_file, source) {
        return vec![ResolvedImport::WorkspaceFile(p)];
    }

    match bare_specifier_package(source) {
        Some(name) => vec![ResolvedImport::External(name)],
        None => vec![ResolvedImport::Unresolved],
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

// --- tsconfig parsing -------------------------------------------------------

/// Parse a `tsconfig.json` into the [`Tsconfig`] subset we resolve against.
///
/// Strips `//` and `/* */` comments first because `tsconfig.json` is JSONC,
/// not strict JSON — Node's TypeScript tooling permits both.
pub fn parse_tsconfig(path: &Path) -> Option<Tsconfig> {
    let src = fs::read_to_string(path).ok()?;
    let cleaned = strip_jsonc_comments(&src);
    let v: Value = serde_json::from_str(&cleaned).ok()?;
    let dir = path.parent()?.to_path_buf();
    let co = v.get("compilerOptions");
    let base_url = co
        .and_then(|c| c.get("baseUrl"))
        .and_then(|x| x.as_str())
        .map(|s| dir.join(s))
        .unwrap_or_else(|| dir.clone());
    let mut paths = Vec::new();
    if let Some(obj) = co.and_then(|c| c.get("paths")).and_then(|p| p.as_object()) {
        for (pattern, value) in obj {
            if let Some(arr) = value.as_array() {
                let subs: Vec<String> = arr
                    .iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect();
                paths.push((pattern.clone(), subs));
            }
        }
        // Longest patterns first so `@org/shared/utils` beats `@org/shared/*`
        // when both could match.
        paths.sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));
    }
    Some(Tsconfig {
        dir,
        base_url,
        paths,
    })
}

/// Strip line and block comments from a JSONC string so `serde_json` can
/// parse it. Doesn't track string boundaries; for `tsconfig.json` this is
/// acceptable because legitimate `//` and `/*` inside string values are
/// vanishingly rare in real configs.
fn strip_jsonc_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b as char);
            if b == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push('"');
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                // Line comment — skip to end of line.
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if bytes[i + 1] == b'*' {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
                continue;
            }
        }
        out.push(b as char);
        i += 1;
    }
    out
}

// --- tsconfig paths / workspace alias resolution ----------------------------

/// Walk up the directory tree from `from_file` looking for the nearest
/// `tsconfig.json`, then try to resolve `source` against its `paths` aliases.
///
/// Stops at the first tsconfig found — TypeScript's "nearest tsconfig wins"
/// rule. If that tsconfig has no matching alias, returns `None` (we don't
/// fall through to a parent tsconfig). Stops at the workspace root so we
/// don't escape into the user's home directory.
fn resolve_tsconfig_path(
    ctx: &ResolverContext,
    ws: &Workspace,
    from_file: &Path,
    source: &str,
) -> Option<PathBuf> {
    let mut current = from_file.parent()?.to_path_buf();
    let ws_root = ws.root.canonicalize().unwrap_or_else(|_| ws.root.clone());
    loop {
        let candidate = current.join("tsconfig.json");
        if candidate.is_file() {
            let canon = candidate.canonicalize().unwrap_or_else(|_| candidate.clone());
            // Prefer a parse cached at workspace-discovery time; fall back to a
            // fresh parse for tsconfigs that live outside any discovered
            // package root (the spec only requires "nearest tsconfig", not
            // "registered tsconfig").
            let parsed_owned: Tsconfig;
            let tsc = match ctx.tsconfigs.get(&canon) {
                Some(t) => t,
                None => match parse_tsconfig(&canon) {
                    Some(t) => {
                        parsed_owned = t;
                        &parsed_owned
                    }
                    None => return None,
                },
            };
            return try_resolve_with_tsconfig(tsc, source);
        }

        // Stop at workspace root so we never read tsconfigs outside the repo.
        let canon_current = current.canonicalize().unwrap_or_else(|_| current.clone());
        if canon_current == ws_root {
            return None;
        }
        match current.parent() {
            Some(p) => current = p.to_path_buf(),
            None => return None,
        }
    }
}

fn try_resolve_with_tsconfig(tsc: &Tsconfig, source: &str) -> Option<PathBuf> {
    for (pattern, subs) in &tsc.paths {
        let Some(matched) = match_paths_pattern(pattern, source) else {
            continue;
        };
        for sub in subs {
            let resolved_sub = sub.replacen('*', matched.as_str(), 1);
            let target = tsc.base_url.join(&resolved_sub);
            if let Some(p) = probe_file(&target) {
                return Some(normalize(p));
            }
            if target.is_dir() {
                for ext in CANDIDATE_EXTS {
                    let i = target.join(format!("index.{ext}"));
                    if i.is_file() {
                        return Some(normalize(i));
                    }
                }
            }
        }
    }
    None
}

/// Match a tsconfig `paths` pattern against an import source.
///
/// Returns `Some(matched_substring)` when `source` matches `pattern` —
/// the matched substring is what the `*` placeholder captured (or empty
/// for exact patterns). Returns `None` if the pattern doesn't match.
fn match_paths_pattern(pattern: &str, source: &str) -> Option<String> {
    if let Some(star) = pattern.find('*') {
        let prefix = &pattern[..star];
        let suffix = &pattern[star + 1..];
        if source.len() < prefix.len() + suffix.len() {
            return None;
        }
        if !source.starts_with(prefix) || !source.ends_with(suffix) {
            return None;
        }
        let middle = &source[prefix.len()..source.len() - suffix.len()];
        Some(middle.to_string())
    } else if source == pattern {
        Some(String::new())
    } else {
        None
    }
}

fn resolve_workspace_alias(ctx: &ResolverContext, source: &str) -> Option<PathBuf> {
    // Exact package-name match → entry file (if we found one).
    if let Some(entry) = ctx.package_entries.get(source) {
        return Some(entry.clone());
    }
    // Subpath: `@org/shared/utils` → resolve `utils` under the package root.
    // Iterate roots so unscoped (`shared/utils`) and scoped (`@org/shared/utils`)
    // both land here without a separate code path.
    for (name, root) in &ctx.package_roots {
        let Some(rest) = source.strip_prefix(name).and_then(|r| r.strip_prefix('/')) else {
            continue;
        };
        let target = root.join(rest);
        if let Some(p) = probe_file(&target) {
            return Some(normalize(p));
        }
        if target.is_dir() {
            for ext in CANDIDATE_EXTS {
                let i = target.join(format!("index.{ext}"));
                if i.is_file() {
                    return Some(normalize(i));
                }
            }
        }
    }
    None
}

/// Pick a package's entry file from its `package.json`, walking the standard
/// fields in priority order (`source` for monorepo unbundled tooling, then
/// `module`, `main`, and the type-only fields). Falls back to `index.*` /
/// `src/index.*` when the manifest is silent.
fn find_package_entry(root: &Path, manifest: &Path) -> Option<PathBuf> {
    let mut declared: Option<String> = None;
    if let Ok(src) = fs::read_to_string(manifest) {
        if let Ok(v) = serde_json::from_str::<Value>(&src) {
            for field in ["source", "module", "main", "types", "typings"] {
                if let Some(s) = v.get(field).and_then(|x| x.as_str()) {
                    declared = Some(s.to_string());
                    break;
                }
            }
        }
    }
    if let Some(rel) = declared {
        let target = root.join(&rel);
        if let Some(p) = probe_file(&target) {
            return Some(normalize(p));
        }
        if target.is_dir() {
            for ext in CANDIDATE_EXTS {
                let i = target.join(format!("index.{ext}"));
                if i.is_file() {
                    return Some(normalize(i));
                }
            }
        }
    }
    for sub in ["", "src"] {
        let base = if sub.is_empty() {
            root.to_path_buf()
        } else {
            root.join(sub)
        };
        for ext in CANDIDATE_EXTS {
            let i = base.join(format!("index.{ext}"));
            if i.is_file() {
                return Some(normalize(i));
            }
        }
    }
    None
}

// --- template-prefix dynamic import -----------------------------------------

/// Expand a template-prefix dynamic import like `` import(`./locales/${l}`) ``
/// into edges to every candidate file under the literal prefix.
///
/// The prefix is split into a directory part + filename prefix at the last
/// `/` — so `./locales/` includes every `*.{ts,…}` in `./locales`, while
/// `./pages/admin-` includes only files in `./pages` whose name starts with
/// `admin-`. Files outside the candidate extensions and the importing file
/// itself are skipped.
fn resolve_template_prefix(from_file: &Path, prefix: &str) -> Vec<ResolvedImport> {
    if prefix.is_empty() || !is_relative_specifier(prefix) {
        return vec![ResolvedImport::Unresolved];
    }
    let Some(from_dir) = from_file.parent() else {
        return vec![ResolvedImport::Unresolved];
    };
    let (dir_part, file_prefix) = match prefix.rfind('/') {
        Some(i) => (&prefix[..=i], &prefix[i + 1..]),
        None => (prefix, ""),
    };
    let target_dir = normalize(from_dir.join(dir_part));
    if !target_dir.is_dir() {
        return vec![ResolvedImport::Unresolved];
    }
    let entries = match fs::read_dir(&target_dir) {
        Ok(e) => e,
        Err(_) => return vec![ResolvedImport::Unresolved],
    };
    let from_canon = from_file.canonicalize().ok();
    let mut results: Vec<ResolvedImport> = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !file_prefix.is_empty() && !name.starts_with(file_prefix) {
            continue;
        }
        let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !CANDIDATE_EXTS.contains(&ext) {
            continue;
        }
        // Don't add a self-loop when the directory contains the importing file.
        if let Some(canon_from) = &from_canon {
            if p.canonicalize().ok().as_deref() == Some(canon_from.as_path()) {
                continue;
            }
        }
        results.push(ResolvedImport::WorkspaceFile(normalize(p)));
    }
    if results.is_empty() {
        results.push(ResolvedImport::Unresolved);
    }
    // Sort for deterministic output — `read_dir` order is OS-dependent.
    results.sort_by(|a, b| match (a, b) {
        (ResolvedImport::WorkspaceFile(x), ResolvedImport::WorkspaceFile(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    });
    results
}

// --- shared helpers ---------------------------------------------------------

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

    fn write(dir: &Path, rel: &str, contents: &str) -> PathBuf {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
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

    // --- pattern matching -------------------------------------------------

    #[test]
    fn paths_pattern_matches_star_capture() {
        assert_eq!(
            match_paths_pattern("@org/shared/*", "@org/shared/utils"),
            Some("utils".to_string())
        );
        assert_eq!(
            match_paths_pattern("@org/shared/*", "@org/shared/utils/index"),
            Some("utils/index".to_string())
        );
    }

    #[test]
    fn paths_pattern_matches_exact() {
        assert_eq!(match_paths_pattern("foo", "foo"), Some(String::new()));
        assert_eq!(match_paths_pattern("foo", "bar"), None);
    }

    #[test]
    fn paths_pattern_rejects_non_matching_prefix() {
        assert_eq!(
            match_paths_pattern("@org/shared/*", "@org/other/utils"),
            None
        );
    }

    // --- tsconfig parsing -------------------------------------------------

    #[test]
    fn tsconfig_parses_paths_and_baseurl() {
        let dir = tempdir().unwrap();
        let tsc = write(
            dir.path(),
            "tsconfig.json",
            r#"{
                // top-level comment
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {
                        "@org/shared/*": ["packages/shared/src/*"]
                    }
                }
            }"#,
        );
        let parsed = parse_tsconfig(&tsc).expect("parse");
        assert!(!parsed.paths.is_empty());
        assert_eq!(parsed.paths[0].0, "@org/shared/*");
        assert_eq!(parsed.paths[0].1, vec!["packages/shared/src/*".to_string()]);
    }

    #[test]
    fn tsconfig_strips_block_comments() {
        let dir = tempdir().unwrap();
        let tsc = write(
            dir.path(),
            "tsconfig.json",
            r#"{
                /* block
                   comment */
                "compilerOptions": { "paths": { "x": ["./y"] } }
            }"#,
        );
        let parsed = parse_tsconfig(&tsc).expect("parse");
        assert_eq!(parsed.paths.len(), 1);
    }

    // --- resolve_import end-to-end ----------------------------------------

    fn ws_with(root: &Path) -> Workspace {
        Workspace::discover(root)
    }

    #[test]
    fn resolves_tsconfig_paths_alias_walking_up_tree() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        write(
            dir.path(),
            "tsconfig.json",
            r#"{
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": { "@myorg/shared/*": ["packages/shared/src/*"] }
                }
            }"#,
        );
        let target = write(dir.path(), "packages/shared/src/utils.ts", "");
        let from = write(dir.path(), "apps/web/src/index.ts", "");

        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement::literal("@myorg/shared/utils", ImportKind::Static);
        let results = resolve_import(&ctx, &ws, &from, &imp);
        assert_eq!(
            results,
            vec![ResolvedImport::WorkspaceFile(
                target.canonicalize().unwrap_or(target)
            )]
        );
    }

    #[test]
    fn nested_tsconfig_overrides_root_for_paths() {
        // Nearest tsconfig wins. Root maps `@x/*` → root_pkg, nested overrides
        // the same alias to point at a different file.
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        write(
            dir.path(),
            "tsconfig.json",
            r#"{
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": { "@x/*": ["root_pkg/*"] }
                }
            }"#,
        );
        write(dir.path(), "root_pkg/foo.ts", "");
        write(
            dir.path(),
            "packages/a/tsconfig.json",
            r#"{
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": { "@x/*": ["nested/*"] }
                }
            }"#,
        );
        let nested = write(dir.path(), "packages/a/nested/foo.ts", "");
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        let from = write(dir.path(), "packages/a/src/index.ts", "");

        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement::literal("@x/foo", ImportKind::Static);
        let results = resolve_import(&ctx, &ws, &from, &imp);
        assert_eq!(
            results,
            vec![ResolvedImport::WorkspaceFile(
                nested.canonicalize().unwrap_or(nested)
            )]
        );
    }

    #[test]
    fn resolves_workspace_alias_to_entry_file() {
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(
            dir.path(),
            "packages/shared/package.json",
            r#"{"name":"@org/shared","main":"src/index.ts"}"#,
        );
        let entry = write(dir.path(), "packages/shared/src/index.ts", "");
        write(
            dir.path(),
            "packages/web/package.json",
            r#"{"name":"@org/web"}"#,
        );
        let from = write(dir.path(), "packages/web/src/main.ts", "");

        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement::literal("@org/shared", ImportKind::Static);
        let results = resolve_import(&ctx, &ws, &from, &imp);
        assert_eq!(
            results,
            vec![ResolvedImport::WorkspaceFile(
                entry.canonicalize().unwrap_or(entry)
            )]
        );
    }

    #[test]
    fn resolves_workspace_alias_subpath() {
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(
            dir.path(),
            "packages/shared/package.json",
            r#"{"name":"@org/shared"}"#,
        );
        let target = write(dir.path(), "packages/shared/utils.ts", "");
        write(
            dir.path(),
            "packages/web/package.json",
            r#"{"name":"@org/web"}"#,
        );
        let from = write(dir.path(), "packages/web/src/main.ts", "");

        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement::literal("@org/shared/utils", ImportKind::Static);
        let results = resolve_import(&ctx, &ws, &from, &imp);
        assert_eq!(
            results,
            vec![ResolvedImport::WorkspaceFile(
                target.canonicalize().unwrap_or(target)
            )]
        );
    }

    #[test]
    fn template_prefix_directory_glob_emits_edge_per_file() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let from = write(dir.path(), "src/loader.ts", "");
        let en = write(dir.path(), "src/locales/en.ts", "");
        let fr = write(dir.path(), "src/locales/fr.ts", "");
        let de = write(dir.path(), "src/locales/de.ts", "");
        // Non-matching extension — must NOT show up.
        write(dir.path(), "src/locales/readme.md", "");

        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement {
            source: "./locales/".to_string(),
            kind: ImportKind::Dynamic,
            is_template: true,
            is_unresolvable: false,
        };
        let mut got: Vec<PathBuf> = resolve_import(&ctx, &ws, &from, &imp)
            .into_iter()
            .filter_map(|r| match r {
                ResolvedImport::WorkspaceFile(p) => Some(p),
                _ => None,
            })
            .collect();
        got.sort();
        let mut expected = vec![en, fr, de];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn template_prefix_with_filename_prefix_filters() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let from = write(dir.path(), "src/loader.ts", "");
        let admin_users = write(dir.path(), "src/pages/admin-users.ts", "");
        let admin_logs = write(dir.path(), "src/pages/admin-logs.ts", "");
        // Non-matching name prefix — must NOT show up.
        write(dir.path(), "src/pages/public.ts", "");

        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement {
            source: "./pages/admin-".to_string(),
            kind: ImportKind::Dynamic,
            is_template: true,
            is_unresolvable: false,
        };
        let mut got: Vec<PathBuf> = resolve_import(&ctx, &ws, &from, &imp)
            .into_iter()
            .filter_map(|r| match r {
                ResolvedImport::WorkspaceFile(p) => Some(p),
                _ => None,
            })
            .collect();
        got.sort();
        let mut expected = vec![admin_logs, admin_users];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn truly_unresolvable_dynamic_returns_empty() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let from = write(dir.path(), "src/a.ts", "");
        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement {
            source: String::new(),
            kind: ImportKind::Dynamic,
            is_template: false,
            is_unresolvable: true,
        };
        let results = resolve_import(&ctx, &ws, &from, &imp);
        assert!(results.is_empty(), "expected empty, got {results:?}");
    }

    #[test]
    fn dynamic_string_import_resolves_like_static() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let from = write(dir.path(), "src/a.ts", "");
        let target = write(dir.path(), "src/b.ts", "");
        let ws = ws_with(dir.path());
        let ctx = ResolverContext::build(&ws);
        let imp = ImportStatement::literal("./b", ImportKind::Dynamic);
        let results = resolve_import(&ctx, &ws, &from, &imp);
        assert_eq!(results, vec![ResolvedImport::WorkspaceFile(target)]);
    }
}
