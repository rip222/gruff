use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageManager {
    Pnpm,
    Yarn,
    Npm,
    /// No recognized lockfile — treat as a plain directory.
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Package {
    /// Name from `package.json`, or a fallback derived from the directory name
    /// when the manifest has no `name` field.
    pub name: String,
    /// Canonicalized directory containing the `package.json`.
    pub root: PathBuf,
    /// Canonicalized path to the `package.json` itself.
    pub manifest: PathBuf,
}

/// Monorepo handle produced by [`Workspace::discover`]. Holds every discovered
/// package and the per-package `tsconfig.json` locations the resolver needs.
/// Path filtering (gitignore, hidden files, `node_modules`) is handled by the
/// indexer's `ignore::WalkBuilder` rather than being plumbed through here.
pub struct Workspace {
    pub root: PathBuf,
    pub package_manager: PackageManager,
    /// The raw workspace globs declared by the root (e.g. `packages/*`). Empty
    /// for single-package repos.
    pub workspace_globs: Vec<String>,
    pub packages: Vec<Package>,
    pub tsconfigs: Vec<PathBuf>,
}

impl Workspace {
    /// Scan `root` for monorepo structure: detect the package manager from the
    /// lockfile, expand `workspaces` globs, and locate nested `package.json`
    /// and `tsconfig.json` files.
    ///
    /// Works identically for a single-package repo: the returned `Workspace`
    /// has one [`Package`] and an empty `workspace_globs`.
    pub fn discover(root: &Path) -> Self {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let package_manager = detect_pm(&root);
        let workspace_globs = read_workspace_globs(&root, package_manager);

        let mut packages: Vec<Package> = Vec::new();
        let mut seen: HashSet<PathBuf> = HashSet::new();

        // Always include the root itself if it has a manifest — the root is
        // a package in npm/yarn workspaces and commonly a "tools" package in
        // pnpm monorepos.
        if let Some(pkg) = read_package(&root) {
            if seen.insert(pkg.root.clone()) {
                packages.push(pkg);
            }
        }

        for pat in &workspace_globs {
            for candidate in expand_glob(&root, pat) {
                if !candidate.is_dir() {
                    continue;
                }
                if let Some(pkg) = read_package(&candidate) {
                    if seen.insert(pkg.root.clone()) {
                        packages.push(pkg);
                    }
                }
            }
        }

        // Fallback: a folder with no package.json at all still yields a single
        // Package so downstream code doesn't have to special-case the "not a
        // project" shape. Uses the directory name as a stand-in identity.
        if packages.is_empty() {
            let name = root
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".to_string());
            packages.push(Package {
                name,
                root: root.clone(),
                manifest: root.join("package.json"),
            });
        }

        let mut tsconfigs = Vec::new();
        for pkg in &packages {
            let ts = pkg.root.join("tsconfig.json");
            if ts.is_file() {
                tsconfigs.push(ts.canonicalize().unwrap_or(ts));
            }
        }

        Self {
            root,
            package_manager,
            workspace_globs,
            packages,
            tsconfigs,
        }
    }

    /// The package that owns `file`, chosen by deepest matching root. Returns
    /// `None` only if `file` lives outside every package — which shouldn't
    /// happen inside a well-formed workspace but is handled defensively.
    pub fn owning_package(&self, file: &Path) -> Option<&Package> {
        let canon = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
        let mut best: Option<&Package> = None;
        let mut best_depth = 0usize;
        for pkg in &self.packages {
            if canon.starts_with(&pkg.root) {
                let depth = pkg.root.components().count();
                if best.is_none() || depth > best_depth {
                    best_depth = depth;
                    best = Some(pkg);
                }
            }
        }
        best
    }

    /// Nearest `tsconfig.json` above `file`, walking up the tree. Consumed by
    /// the resolver in slice #6 — surfaced here so that wiring is in place.
    pub fn tsconfig_for(&self, file: &Path) -> Option<&Path> {
        let canon = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
        let mut best: Option<&PathBuf> = None;
        let mut best_depth = 0usize;
        for ts in &self.tsconfigs {
            let Some(dir) = ts.parent() else { continue };
            if canon.starts_with(dir) {
                let depth = dir.components().count();
                if best.is_none() || depth > best_depth {
                    best_depth = depth;
                    best = Some(ts);
                }
            }
        }
        best.map(|p| p.as_path())
    }
}

// --- lockfile-based package manager detection -------------------------------

fn detect_pm(root: &Path) -> PackageManager {
    // Priority: pnpm > yarn > npm. First match wins.
    if root.join("pnpm-lock.yaml").is_file() {
        return PackageManager::Pnpm;
    }
    if root.join("yarn.lock").is_file() {
        return PackageManager::Yarn;
    }
    if root.join("package-lock.json").is_file() {
        return PackageManager::Npm;
    }
    PackageManager::Unknown
}

// --- workspace glob extraction ---------------------------------------------

fn read_workspace_globs(root: &Path, pm: PackageManager) -> Vec<String> {
    let mut globs = Vec::new();

    if pm == PackageManager::Pnpm {
        let pnpm_path = root.join("pnpm-workspace.yaml");
        if let Ok(src) = fs::read_to_string(&pnpm_path) {
            globs.extend(parse_pnpm_workspace_yaml(&src));
        }
    }

    // npm / yarn keep workspace globs in the root package.json. pnpm usually
    // doesn't, but accept it for repos that mix the two conventions.
    let manifest = root.join("package.json");
    if let Ok(src) = fs::read_to_string(&manifest) {
        if let Ok(v) = serde_json::from_str::<Value>(&src) {
            globs.extend(extract_package_json_workspaces(&v));
        }
    }

    globs
}

fn extract_package_json_workspaces(v: &Value) -> Vec<String> {
    let Some(ws) = v.get("workspaces") else {
        return Vec::new();
    };
    // Two supported shapes:
    //   "workspaces": ["packages/*"]
    //   "workspaces": { "packages": ["packages/*"] }   (yarn classic nohoist)
    if let Some(arr) = ws.as_array() {
        return arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
    }
    if let Some(obj) = ws.as_object() {
        if let Some(pkgs) = obj.get("packages").and_then(|p| p.as_array()) {
            return pkgs
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
        }
    }
    Vec::new()
}

/// Parse the subset of `pnpm-workspace.yaml` we care about:
///
/// ```yaml
/// packages:
///   - "apps/*"
///   - 'packages/*'
///   - tools
/// ```
///
/// Anything else (catalog, overrides, etc.) is ignored. A hand-rolled parser
/// avoids pulling in a full YAML dependency for what is effectively a
/// list-of-strings.
fn parse_pnpm_workspace_yaml(src: &str) -> Vec<String> {
    let mut globs = Vec::new();
    let mut in_packages = false;
    let mut list_indent: Option<usize> = None;

    for raw_line in src.lines() {
        // Strip inline comments outside of quoted strings. The inputs we care
        // about don't use `#` inside values, so a simple strip is fine.
        let line = match raw_line.find('#') {
            Some(i) => &raw_line[..i],
            None => raw_line,
        };
        if line.trim().is_empty() {
            continue;
        }

        let indent = line.chars().take_while(|c| *c == ' ').count();
        let trimmed = line.trim_start();

        if indent == 0 {
            in_packages = trimmed.starts_with("packages:");
            list_indent = None;
            continue;
        }

        if !in_packages {
            continue;
        }

        // List item under `packages:` — `- value`.
        if let Some(rest) = trimmed.strip_prefix("- ") {
            // Lock in the indent of the first item; any line at lesser
            // indent means we've left the packages block.
            if list_indent.map(|l| indent < l).unwrap_or(false) {
                in_packages = false;
                continue;
            }
            list_indent.get_or_insert(indent);
            let value = strip_quotes(rest.trim());
            if !value.is_empty() {
                globs.push(value.to_string());
            }
        } else if list_indent.map(|l| indent <= l).unwrap_or(false) {
            // Something at list-level that isn't a list item → end of block.
            in_packages = false;
        }
    }

    globs
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

// --- glob expansion ---------------------------------------------------------

fn expand_glob(root: &Path, pattern: &str) -> Vec<PathBuf> {
    // Negative patterns (`!apps/legacy`) aren't handled in this slice — they're
    // vanishingly rare in real monorepos and yarn/pnpm only treat them as
    // exclusion hints for tooling, not filesystem matching.
    if pattern.starts_with('!') {
        return Vec::new();
    }
    let joined = root.join(pattern);
    let Some(joined_str) = joined.to_str() else {
        return Vec::new();
    };
    let Ok(iter) = glob::glob(joined_str) else {
        return Vec::new();
    };
    iter.filter_map(Result::ok)
        .filter_map(|p| p.canonicalize().ok().or(Some(p)))
        .collect()
}

// --- package.json reading ---------------------------------------------------

fn read_package(dir: &Path) -> Option<Package> {
    let manifest = dir.join("package.json");
    if !manifest.is_file() {
        return None;
    }
    let src = fs::read_to_string(&manifest).ok()?;
    let v: Value = serde_json::from_str(&src).ok()?;
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .map(String::from)
        .unwrap_or_else(|| {
            dir.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| ".".to_string())
        });
    let root = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let manifest = manifest.canonicalize().unwrap_or(manifest);
    Some(Package {
        name,
        root,
        manifest,
    })
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
    fn detects_pnpm_from_lockfile() {
        let dir = tempdir().unwrap();
        write(dir.path(), "pnpm-lock.yaml", "");
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let ws = Workspace::discover(dir.path());
        assert_eq!(ws.package_manager, PackageManager::Pnpm);
    }

    #[test]
    fn detects_yarn_from_lockfile() {
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let ws = Workspace::discover(dir.path());
        assert_eq!(ws.package_manager, PackageManager::Yarn);
    }

    #[test]
    fn detects_npm_from_lockfile() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package-lock.json", "");
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let ws = Workspace::discover(dir.path());
        assert_eq!(ws.package_manager, PackageManager::Npm);
    }

    #[test]
    fn pnpm_takes_priority_over_yarn_and_npm() {
        // If multiple lockfiles exist (e.g. a half-migrated repo), pnpm wins.
        let dir = tempdir().unwrap();
        write(dir.path(), "pnpm-lock.yaml", "");
        write(dir.path(), "yarn.lock", "");
        write(dir.path(), "package-lock.json", "");
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        let ws = Workspace::discover(dir.path());
        assert_eq!(ws.package_manager, PackageManager::Pnpm);
    }

    #[test]
    fn enumerates_pnpm_workspace_packages() {
        let dir = tempdir().unwrap();
        write(dir.path(), "pnpm-lock.yaml", "");
        write(
            dir.path(),
            "pnpm-workspace.yaml",
            "packages:\n  - \"apps/*\"\n  - 'packages/*'\n",
        );
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        write(
            dir.path(),
            "apps/web/package.json",
            r#"{"name":"@org/web"}"#,
        );
        write(
            dir.path(),
            "apps/api/package.json",
            r#"{"name":"@org/api"}"#,
        );
        write(
            dir.path(),
            "packages/shared/package.json",
            r#"{"name":"@org/shared"}"#,
        );

        let ws = Workspace::discover(dir.path());
        let names: HashSet<_> = ws.packages.iter().map(|p| p.name.clone()).collect();
        assert!(names.contains("root"));
        assert!(names.contains("@org/web"));
        assert!(names.contains("@org/api"));
        assert!(names.contains("@org/shared"));
    }

    #[test]
    fn enumerates_yarn_workspace_packages() {
        let dir = tempdir().unwrap();
        write(dir.path(), "yarn.lock", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        write(dir.path(), "packages/b/package.json", r#"{"name":"b"}"#);

        let ws = Workspace::discover(dir.path());
        let names: HashSet<_> = ws.packages.iter().map(|p| p.name.clone()).collect();
        assert!(names.contains("root"));
        assert!(names.contains("a"));
        assert!(names.contains("b"));
    }

    #[test]
    fn enumerates_npm_workspace_packages() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package-lock.json", "");
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "packages/one/package.json", r#"{"name":"one"}"#);

        let ws = Workspace::discover(dir.path());
        let names: HashSet<_> = ws.packages.iter().map(|p| p.name.clone()).collect();
        assert!(names.contains("root"));
        assert!(names.contains("one"));
    }

    #[test]
    fn single_package_repo_has_one_package_and_empty_globs() {
        let dir = tempdir().unwrap();
        write(dir.path(), "package.json", r#"{"name":"solo"}"#);
        let ws = Workspace::discover(dir.path());
        assert!(ws.workspace_globs.is_empty());
        assert_eq!(ws.packages.len(), 1);
        assert_eq!(ws.packages[0].name, "solo");
    }

    #[test]
    fn folder_without_package_json_still_yields_single_package() {
        // Dropping a bare folder that isn't a project should still produce a
        // Workspace — the indexer must not have to special-case this.
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        let ws = Workspace::discover(dir.path());
        assert_eq!(ws.packages.len(), 1);
        assert_eq!(ws.package_manager, PackageManager::Unknown);
    }

    #[test]
    fn owning_package_picks_deepest_match() {
        let dir = tempdir().unwrap();
        write(
            dir.path(),
            "package.json",
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        );
        write(dir.path(), "yarn.lock", "");
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        let file = write(dir.path(), "packages/a/src/index.ts", "");

        let ws = Workspace::discover(dir.path());
        let pkg = ws
            .owning_package(&file)
            .expect("should find owning package");
        // Deeper package root must win over root package.
        assert_eq!(pkg.name, "a");
    }

    #[test]
    fn exposes_nested_tsconfigs() {
        let dir = tempdir().unwrap();
        write(dir.path(), "pnpm-lock.yaml", "");
        write(
            dir.path(),
            "pnpm-workspace.yaml",
            "packages:\n  - packages/*\n",
        );
        write(dir.path(), "package.json", r#"{"name":"root"}"#);
        write(dir.path(), "tsconfig.json", r#"{}"#);
        write(dir.path(), "packages/a/package.json", r#"{"name":"a"}"#);
        write(dir.path(), "packages/a/tsconfig.json", r#"{}"#);
        let file = write(dir.path(), "packages/a/src/x.ts", "");

        let ws = Workspace::discover(dir.path());
        assert_eq!(ws.tsconfigs.len(), 2);
        // Nearest tsconfig is the one inside package `a`, not the root one.
        let near = ws.tsconfig_for(&file).expect("should resolve tsconfig");
        assert!(near.to_string_lossy().contains("packages/a"));
    }
}
