use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

use crate::error::GruffError;
use crate::filters::{is_config_file, is_test_file};
use crate::graph::{Edge, Graph, GraphDiff, Node, NodeId, NodeKind};
use crate::parser::{ImportKind, parse_file_imports};
use crate::resolver::{ResolvedImport, ResolverContext, resolve_import};
use crate::workspace::Workspace;

const SOURCE_EXTS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];

/// Indexer knobs the UI can flip at runtime. Kept small — new filters belong
/// here so the "what does the default graph show?" answer stays in one place.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexerOptions {
    /// When false (default), test files (`*.test.*`, `*.spec.*`) are dropped
    /// from the graph. Toggle the menu item or call
    /// [`Indexer::set_include_tests`] to flip.
    pub include_tests: bool,
}

/// Node id prefix for synthetic external-package leaves. Picked so it can't
/// collide with a relative-path file id (paths have no `:` in them).
const EXTERNAL_ID_PREFIX: &str = "external:";
/// Node id prefix for synthetic workspace-package aggregators — the source
/// endpoint of every edge into an external leaf.
const WORKSPACE_PKG_ID_PREFIX: &str = "package:";

/// Build the synthetic id for an external package node.
pub fn external_node_id(name: &str) -> String {
    format!("{EXTERNAL_ID_PREFIX}{name}")
}

/// Build the synthetic id for a workspace package aggregator node. Uses a
/// distinct prefix so it can't collide with either files (relative paths) or
/// externals.
pub fn workspace_package_node_id(name: &str) -> String {
    format!("{WORKSPACE_PKG_ID_PREFIX}{name}")
}

/// True if `path` has a JS/TS source extension the indexer cares about.
/// Surfaced publicly so the watcher can filter events before they reach the
/// incremental-update path.
pub fn is_source_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SOURCE_EXTS.contains(&ext))
        .unwrap_or(false)
}

/// Output of an indexing pass: the file-level dependency graph plus index-time
/// metadata the UI needs (currently the count of fully-unresolvable dynamic
/// imports, surfaced in the status bar).
#[derive(Debug, Default)]
pub struct IndexResult {
    pub graph: Graph,
    /// Number of `import(...)` calls whose source couldn't be statically
    /// determined (e.g. `import(modName)`). Reported in the status bar so the
    /// user knows the graph isn't showing those edges.
    pub unresolved_dynamic: usize,
    /// Current per-file parse/read failures. Surfaced by the status bar while
    /// the affected file remains broken.
    pub errors: Vec<GruffError>,
}

/// Stateful indexer that keeps enough context alive between calls to service
/// incremental updates. Owns the workspace handle, resolver context, graph,
/// and the per-file import bookkeeping needed to diff a changed file without
/// rescanning the whole repo.
pub struct Indexer {
    pub ws: Workspace,
    pub ctx: ResolverContext,
    pub graph: Graph,
    pub unresolved_dynamic: usize,
    pub options: IndexerOptions,
    /// `canonical_path → graph node id`. Populated for every file node; used
    /// by `update_file` to locate the existing node without rescanning.
    id_by_path: HashMap<PathBuf, NodeId>,
    /// `canonical_path → owning package name` (`None` for unattributed files).
    /// Cached because `owning_package` has to canonicalize + walk package
    /// roots on every call.
    package_of_file: HashMap<PathBuf, Option<String>>,
    /// Per-file set of resolved imports kept in a normalized form so diffs
    /// against a freshly-parsed set are a plain `HashSet::difference`. The
    /// `ImportTarget` enum collapses "zero or more `ResolvedImport`" into a
    /// hashable form.
    imports_by_file: HashMap<PathBuf, HashSet<ImportTarget>>,
    /// For each external package, how many files currently import it. Lets
    /// `update_file` and `remove_file` drop the synthetic external leaf and
    /// its aggregated edge when the refcount hits zero.
    external_refs: HashMap<String, usize>,
    /// For each (workspace_pkg, external_pkg) pair, how many file imports
    /// currently contribute to the aggregated edge. Used to remove the
    /// aggregated edge when its refcount drops to zero.
    aggregated_refs: HashMap<(String, String), usize>,
    /// Count of file edges per (from_file, to_file) pair. A workspace file
    /// can reference another file through both a static import and a
    /// re-export; we want exactly one graph edge, and the refcount lets us
    /// remove it only when the last contributing import goes away.
    file_edge_refs: HashMap<(NodeId, NodeId), usize>,
    /// Per-file count of unresolvable dynamic imports. Kept so that when a
    /// file is reindexed the running `unresolved_dynamic` total can be
    /// decremented by the file's prior contribution before adding the new.
    file_unresolved: HashMap<PathBuf, usize>,
    /// Per-file parse/read errors currently active in the graph.
    file_errors: HashMap<PathBuf, GruffError>,
}

/// Canonicalized form of what a file imports, used as the key into
/// `imports_by_file`. Collapses every resolver outcome into a hashable shape
/// so diff-by-set works — otherwise we'd be comparing `Vec<ResolvedImport>`
/// pairwise.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ImportTarget {
    /// Resolved to a workspace file — key is the target's canonical path.
    File(PathBuf),
    /// Resolved to an external package (by name only).
    External(String),
}

/// Scan `root` into a `Workspace`, then index every JS/TS file under it.
/// This is the entry point the app uses when the user drops a folder.
pub fn index_folder(root: &Path) -> IndexResult {
    let indexer = Indexer::build(root);
    let errors = indexer.current_errors();
    IndexResult {
        graph: indexer.graph,
        unresolved_dynamic: indexer.unresolved_dynamic,
        errors,
    }
}

/// True if `path` passes the indexer's active file gate: JS/TS source file,
/// not a known build config, and — when `include_tests` is off — not a test
/// file. Surfaced so the watcher can mirror the same filter when deciding
/// whether to forward an event.
pub fn passes_filters(path: &Path, options: IndexerOptions) -> bool {
    if !is_source_file(path) {
        return false;
    }
    if is_config_file(path) {
        return false;
    }
    if !options.include_tests && is_test_file(path) {
        return false;
    }
    true
}

impl Indexer {
    /// Full scan from a repo root. Equivalent to `index_folder` plus keeping
    /// the state alive for subsequent [`Indexer::update_file`] calls.
    pub fn build(root: &Path) -> Self {
        let ws = Workspace::discover(root);
        Self::from_workspace(ws)
    }

    /// Build from an already-discovered workspace. Separated out so tests can
    /// feed a synthetic workspace directly.
    pub fn from_workspace(ws: Workspace) -> Self {
        let ctx = ResolverContext::build(&ws);
        let mut indexer = Indexer {
            ws,
            ctx,
            graph: Graph::new(),
            unresolved_dynamic: 0,
            options: IndexerOptions::default(),
            id_by_path: HashMap::new(),
            package_of_file: HashMap::new(),
            imports_by_file: HashMap::new(),
            external_refs: HashMap::new(),
            aggregated_refs: HashMap::new(),
            file_edge_refs: HashMap::new(),
            file_unresolved: HashMap::new(),
            file_errors: HashMap::new(),
        };
        indexer.full_scan();
        indexer
    }

    /// Full scan: used by `build` at startup and by Cmd+R to recover from
    /// drift. Resets every cached piece of state so we don't carry over
    /// stale refcounts from a previous scan.
    pub fn rescan(&mut self) {
        self.ws = Workspace::discover(&self.ws.root);
        self.ctx = ResolverContext::build(&self.ws);
        self.graph = Graph::new();
        self.unresolved_dynamic = 0;
        self.id_by_path.clear();
        self.package_of_file.clear();
        self.imports_by_file.clear();
        self.external_refs.clear();
        self.aggregated_refs.clear();
        self.file_edge_refs.clear();
        self.file_unresolved.clear();
        self.file_errors.clear();
        self.full_scan();
    }

    pub fn current_errors(&self) -> Vec<GruffError> {
        let mut errors: Vec<_> = self.file_errors.values().cloned().collect();
        errors.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        errors
    }

    /// Flip the "include test files" toggle and rebuild the graph. The test-
    /// set changing is a structural edit — a full rescan is simpler and fast
    /// enough that we don't bother with a surgical "add/remove just the test
    /// files" path.
    pub fn set_include_tests(&mut self, include: bool) {
        if self.options.include_tests == include {
            return;
        }
        self.options.include_tests = include;
        self.rescan();
    }

    fn full_scan(&mut self) {
        let files = collect_source_files(&self.ws.root, self.options);

        for f in &files {
            let canonical = canonicalize_or(f);
            let id = relative_id(&self.ws.root, &canonical);
            let label = canonical
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| id.clone());
            let package = self.ws.owning_package(&canonical).map(|p| p.name.clone());
            self.graph.add_node(Node {
                id: id.clone(),
                path: canonical.clone(),
                label,
                package: package.clone(),
                kind: NodeKind::File,
            });
            self.id_by_path.insert(canonical.clone(), id);
            self.package_of_file.insert(canonical, package);
        }

        for f in &files {
            let canonical = canonicalize_or(f);
            self.reindex_file_imports(&canonical);
        }
    }

    /// Incremental update for a file that was created or modified on disk.
    /// Re-parses the file, resolves its imports, and returns the minimal
    /// diff against the current graph.
    ///
    /// If `path` is outside the workspace or not a source file, the diff is
    /// empty. Applying the diff is the caller's responsibility (normally
    /// [`Graph::apply`] on `indexer.graph` plus the live view).
    pub fn update_file(&mut self, path: &Path) -> GraphDiff {
        if !is_source_file(path) || !path.is_file() {
            return GraphDiff::default();
        }
        // Honor the same gating the initial scan applies — without this, a
        // new test file would sneak in as a live update even though the
        // toggle is off, and a config file dropped in during editing would
        // get parsed for imports.
        if !passes_filters(path, self.options) {
            return GraphDiff::default();
        }
        let canonical = canonicalize_or(path);
        if !canonical.starts_with(&self.ws.root) {
            return GraphDiff::default();
        }

        let mut diff = GraphDiff::default();

        // If this is a new file, materialize the node first so edges we're
        // about to add have a valid source endpoint.
        if !self.id_by_path.contains_key(&canonical) {
            let id = relative_id(&self.ws.root, &canonical);
            let label = canonical
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| id.clone());
            let package = self.ws.owning_package(&canonical).map(|p| p.name.clone());
            let node = Node {
                id: id.clone(),
                path: canonical.clone(),
                label,
                package: package.clone(),
                kind: NodeKind::File,
            };
            self.graph.add_node(node.clone());
            diff.added_nodes.push(node);
            self.id_by_path.insert(canonical.clone(), id);
            self.package_of_file.insert(canonical.clone(), package);
        }

        self.apply_import_diff_for_file(&canonical, &mut diff);
        diff
    }

    /// Incremental update for a file that was deleted from disk. Removes the
    /// file's node, every incoming/outgoing workspace edge, and decrements
    /// refcounts for any external/aggregated edges it contributed to.
    pub fn remove_file(&mut self, path: &Path) -> GraphDiff {
        let canonical = canonicalize_or(path);
        let Some(file_id) = self.id_by_path.remove(&canonical) else {
            return GraphDiff::default();
        };
        self.file_errors.remove(&canonical);

        let mut diff = GraphDiff::default();
        let owning_pkg = self
            .package_of_file
            .remove(&canonical)
            .flatten()
            .unwrap_or_else(|| "(unattributed)".to_string());

        // Roll back every import this file contributed to: external refs,
        // aggregated-edge refs, and workspace file edges.
        if let Some(prev_targets) = self.imports_by_file.remove(&canonical) {
            for target in prev_targets {
                self.release_import(&file_id, &owning_pkg, &target, &mut diff);
            }
        }

        // Incoming edges from other workspace files to this one must also go.
        // They're tracked in `file_edge_refs` — drain entries pointing at the
        // removed id.
        let incoming: Vec<(NodeId, NodeId)> = self
            .file_edge_refs
            .keys()
            .filter(|(_, to)| to == &file_id)
            .cloned()
            .collect();
        for key in incoming {
            if let Some(count) = self.file_edge_refs.remove(&key) {
                // Every contributing import is going away with the file;
                // record one removed_edge regardless of refcount.
                let _ = count;
                diff.removed_edges.push(Edge {
                    from: key.0.clone(),
                    to: key.1.clone(),
                });
            }
        }

        self.graph.remove_node(&file_id);
        diff.removed_nodes.push(file_id);
        diff
    }

    /// Re-parse `canonical` and reconcile `imports_by_file` with the freshly-
    /// parsed set. Shared by the full scan (where the "old" set is empty) and
    /// by `update_file` (where it contains the previous imports).
    fn reindex_file_imports(&mut self, canonical: &Path) {
        // The full scan mutates `graph` directly inside the shared helper;
        // the diff is written to a throwaway buffer because no caller cares.
        let mut scratch = GraphDiff::default();
        self.apply_import_diff_for_file(canonical, &mut scratch);
    }

    /// Core diff loop: parse `file`, resolve every import, then add/remove
    /// refcounted edges so the graph reflects the new import set. Mutates
    /// `self.graph` and appends to `diff` for the caller.
    fn apply_import_diff_for_file(&mut self, file: &Path, diff: &mut GraphDiff) {
        let from_id = match self.id_by_path.get(file) {
            Some(id) => id.clone(),
            None => return,
        };
        let owning_pkg = self
            .package_of_file
            .get(file)
            .cloned()
            .flatten()
            .unwrap_or_else(|| "(unattributed)".to_string());

        let (new_targets, new_unresolved, file_error) = self.resolve_file_imports(file);

        if let Some(error) = file_error {
            self.file_errors.insert(file.to_path_buf(), error);
        } else {
            self.file_errors.remove(file);
        }

        let old_targets = self.imports_by_file.remove(file).unwrap_or_default();
        let old_unresolved = self.unresolved_by_file(file).unwrap_or(0);

        for removed in old_targets.difference(&new_targets) {
            self.release_import(&from_id, &owning_pkg, removed, diff);
        }
        for added in new_targets.difference(&old_targets) {
            self.acquire_import(&from_id, &owning_pkg, added, diff);
        }

        self.imports_by_file.insert(file.to_path_buf(), new_targets);
        self.unresolved_dynamic =
            self.unresolved_dynamic.saturating_sub(old_unresolved) + new_unresolved;
        self.set_unresolved_for_file(file, new_unresolved);
    }

    fn resolve_file_imports(
        &self,
        file: &Path,
    ) -> (HashSet<ImportTarget>, usize, Option<GruffError>) {
        let mut targets: HashSet<ImportTarget> = HashSet::new();
        let mut unresolved: usize = 0;
        let parsed = parse_file_imports(file);

        for imp in parsed.imports {
            let results = resolve_import(&self.ctx, &self.ws, file, &imp);
            if results.is_empty() {
                if imp.kind == ImportKind::Dynamic {
                    unresolved += 1;
                }
                continue;
            }
            let mut produced = false;
            for r in results {
                match r {
                    ResolvedImport::WorkspaceFile(target) => {
                        let canon = canonicalize_or(&target);
                        if self.id_by_path.contains_key(&canon) {
                            targets.insert(ImportTarget::File(canon));
                            produced = true;
                        }
                    }
                    ResolvedImport::External(pkg) => {
                        targets.insert(ImportTarget::External(pkg));
                        produced = true;
                    }
                    ResolvedImport::Unresolved => {}
                }
            }
            if !produced && imp.kind == ImportKind::Dynamic {
                unresolved += 1;
            }
        }

        (targets, unresolved, parsed.error)
    }

    /// Record the newly-added import and mutate `graph` + `diff` to match.
    /// Workspace-file targets contribute to `file_edge_refs`; external
    /// targets contribute to `external_refs` + `aggregated_refs`.
    fn acquire_import(
        &mut self,
        from_id: &NodeId,
        owning_pkg: &str,
        target: &ImportTarget,
        diff: &mut GraphDiff,
    ) {
        match target {
            ImportTarget::File(to_canon) => {
                let Some(to_id) = self.id_by_path.get(to_canon).cloned() else {
                    return;
                };
                let key = (from_id.clone(), to_id.clone());
                let count = self.file_edge_refs.entry(key).or_insert(0);
                *count += 1;
                if *count == 1 {
                    self.graph.add_edge(from_id, &to_id);
                    diff.added_edges.push(Edge {
                        from: from_id.clone(),
                        to: to_id,
                    });
                }
            }
            ImportTarget::External(pkg) => {
                // Ensure the external leaf exists.
                let ext_id = external_node_id(pkg);
                let ext_count = self.external_refs.entry(pkg.clone()).or_insert(0);
                *ext_count += 1;
                if *ext_count == 1 {
                    let node = Node {
                        id: ext_id.clone(),
                        path: PathBuf::from(pkg),
                        label: pkg.clone(),
                        package: None,
                        kind: NodeKind::External,
                    };
                    self.graph.add_node(node.clone());
                    diff.added_nodes.push(node);
                }

                // Ensure the workspace-package aggregator exists.
                let ws_id = workspace_package_node_id(owning_pkg);
                if !self.graph.nodes.contains_key(&ws_id) {
                    let node = Node {
                        id: ws_id.clone(),
                        path: PathBuf::from(owning_pkg),
                        label: owning_pkg.to_string(),
                        package: Some(owning_pkg.to_string()),
                        kind: NodeKind::WorkspacePackage,
                    };
                    self.graph.add_node(node.clone());
                    diff.added_nodes.push(node);
                }

                let agg_key = (owning_pkg.to_string(), pkg.clone());
                let agg_count = self.aggregated_refs.entry(agg_key.clone()).or_insert(0);
                *agg_count += 1;
                if *agg_count == 1 {
                    self.graph.add_edge(&ws_id, &ext_id);
                    diff.added_edges.push(Edge {
                        from: ws_id,
                        to: ext_id,
                    });
                }
            }
        }
    }

    /// Release (decrement-and-maybe-remove) a previously-held import. The
    /// inverse of `acquire_import` — when the last reference to an external
    /// package or aggregated edge goes, its synthetic node/edge is removed
    /// from the graph and emitted in the diff.
    fn release_import(
        &mut self,
        from_id: &NodeId,
        owning_pkg: &str,
        target: &ImportTarget,
        diff: &mut GraphDiff,
    ) {
        match target {
            ImportTarget::File(to_canon) => {
                let Some(to_id) = self.id_by_path.get(to_canon).cloned() else {
                    return;
                };
                let key = (from_id.clone(), to_id.clone());
                let Some(count) = self.file_edge_refs.get_mut(&key) else {
                    return;
                };
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.file_edge_refs.remove(&key);
                    let edge = Edge {
                        from: from_id.clone(),
                        to: to_id,
                    };
                    self.graph.edges.retain(|e| e != &edge);
                    diff.removed_edges.push(edge);
                }
            }
            ImportTarget::External(pkg) => {
                let agg_key = (owning_pkg.to_string(), pkg.clone());
                if let Some(agg) = self.aggregated_refs.get_mut(&agg_key) {
                    *agg = agg.saturating_sub(1);
                    if *agg == 0 {
                        self.aggregated_refs.remove(&agg_key);
                        let ws_id = workspace_package_node_id(owning_pkg);
                        let ext_id = external_node_id(pkg);
                        let edge = Edge {
                            from: ws_id.clone(),
                            to: ext_id.clone(),
                        };
                        self.graph.edges.retain(|e| e != &edge);
                        diff.removed_edges.push(edge);

                        // Drop the aggregator if it no longer has any
                        // outgoing aggregated edges.
                        let still_used = self.aggregated_refs.keys().any(|(p, _)| p == owning_pkg);
                        if !still_used && self.graph.nodes.contains_key(&ws_id) {
                            self.graph.remove_node(&ws_id);
                            diff.removed_nodes.push(ws_id);
                        }
                    }
                }

                if let Some(ext_count) = self.external_refs.get_mut(pkg) {
                    *ext_count = ext_count.saturating_sub(1);
                    if *ext_count == 0 {
                        self.external_refs.remove(pkg);
                        let ext_id = external_node_id(pkg);
                        if self.graph.nodes.contains_key(&ext_id) {
                            self.graph.remove_node(&ext_id);
                            diff.removed_nodes.push(ext_id);
                        }
                    }
                }
            }
        }
    }

    /// Count of unresolvable dynamic imports currently attributed to `file`.
    /// Derived by rescanning the file's imports — we don't persist per-file
    /// counts separately because they're cheap to recompute and it keeps the
    /// state footprint smaller.
    fn unresolved_by_file(&self, _file: &Path) -> Option<usize> {
        // Track per-file unresolved counts implicitly via `file_unresolved`.
        // This helper exists so the control flow in `apply_import_diff_for_file`
        // reads naturally even though the count is computed lazily elsewhere.
        self.file_unresolved.get(_file).copied()
    }

    fn set_unresolved_for_file(&mut self, file: &Path, count: usize) {
        if count == 0 {
            self.file_unresolved.remove(file);
        } else {
            self.file_unresolved.insert(file.to_path_buf(), count);
        }
    }
}

// --- helpers ---------------------------------------------------------------

fn canonicalize_or(p: &Path) -> PathBuf {
    // Direct canonicalize handles the common case (file exists on disk).
    if let Ok(c) = p.canonicalize() {
        return c;
    }
    // File may have just been deleted (relevant to watcher-driven removes).
    // Canonicalize the parent and re-join so callers that store by canonical
    // path can still find the entry by the raw notify event path.
    if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
        if let Ok(cp) = parent.canonicalize() {
            return cp.join(name);
        }
    }
    p.to_path_buf()
}

fn relative_id(root: &Path, canonical: &Path) -> NodeId {
    let rel = canonical.strip_prefix(root).unwrap_or(canonical);
    rel.to_string_lossy().replace('\\', "/")
}

fn collect_source_files(root: &Path, options: IndexerOptions) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let walker = WalkBuilder::new(root)
        .follow_links(false)
        // `WalkBuilder` only honors `.gitignore` inside an actual git repo by
        // default; `require_git(false)` makes the rules apply universally,
        // which matches what a user dropping any folder into Gruff expects.
        .require_git(false)
        // `node_modules` is universal noise and isn't always gitignored in
        // every nested package, so we drop it explicitly.
        .filter_entry(|e| e.file_name() != "node_modules")
        .build();
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() && passes_filters(path, options) {
            files.push(path.to_path_buf());
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn indexes_two_files_with_one_import() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "export const b = 1;").unwrap();

        let g = index_folder(dir.path()).graph;
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn skips_node_modules() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();

        let g = index_folder(dir.path()).graph;
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn skips_paths_ignored_by_root_gitignore() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join(".gitignore"), "dist/\n").unwrap();
        fs::write(dir.path().join("src.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("dist")).unwrap();
        fs::write(dir.path().join("dist/bundle.js"), "").unwrap();

        let g = index_folder(dir.path()).graph;
        assert_eq!(g.nodes.len(), 1);
        assert!(g.nodes.values().any(|n| n.label == "src.ts"));
    }

    #[test]
    fn skips_paths_ignored_by_nested_gitignore() {
        // Regression: the root matcher used to be the only one consulted, so
        // rules living in a nested `.gitignore` (e.g. a Capacitor-style mobile
        // project that ignores its built public/ bundle) were silently lost
        // and the minified chunks showed up as graph nodes.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("root.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("mobile/ios/App/App/public")).unwrap();
        fs::write(dir.path().join("mobile/ios/.gitignore"), "App/App/public\n").unwrap();
        fs::write(
            dir.path().join("mobile/ios/App/App/public/chunk-AAA.js"),
            "",
        )
        .unwrap();
        fs::write(
            dir.path().join("mobile/ios/App/App/public/chunk-BBB.js"),
            "",
        )
        .unwrap();

        let g = index_folder(dir.path()).graph;
        assert!(
            g.nodes.values().all(|n| !n.label.starts_with("chunk-")),
            "nested gitignore should have hidden the chunks, got nodes: {:?}",
            g.nodes.values().map(|n| &n.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn skips_dot_directories() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join(".cache")).unwrap();
        fs::write(dir.path().join(".cache/leaked.ts"), "").unwrap();

        let g = index_folder(dir.path()).graph;
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn external_bare_specifier_adds_single_external_node() {
        // Two files in the same workspace package both import `react`. There
        // should be exactly one `external:react` node regardless of importer
        // count, and exactly one edge from the package aggregator to it.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name":"@org/app"}"#).unwrap();
        fs::write(dir.path().join("a.ts"), r#"import React from "react";"#).unwrap();
        fs::write(dir.path().join("b.ts"), r#"import React from "react";"#).unwrap();

        let g = index_folder(dir.path()).graph;

        let ext_id = external_node_id("react");
        let react_nodes: Vec<_> = g.nodes.values().filter(|n| n.id == ext_id).collect();
        assert_eq!(react_nodes.len(), 1);
        assert_eq!(react_nodes[0].kind, NodeKind::External);

        let edges_to_react: Vec<_> = g.edges.iter().filter(|e| e.to == ext_id).collect();
        // One aggregated edge, even though two files import react.
        assert_eq!(edges_to_react.len(), 1);
        assert_eq!(
            edges_to_react[0].from,
            workspace_package_node_id("@org/app")
        );
    }

    #[test]
    fn scoped_and_subpath_externals_collapse_to_package_name() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name":"@org/app"}"#).unwrap();
        fs::write(
            dir.path().join("a.ts"),
            r#"
                import x from "@scope/pkg/deep";
                import y from "@scope/pkg";
                import z from "lodash/fp";
                import w from "lodash";
            "#,
        )
        .unwrap();

        let g = index_folder(dir.path()).graph;

        let externals: Vec<_> = g
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::External)
            .map(|n| n.label.clone())
            .collect();
        assert!(externals.contains(&"@scope/pkg".to_string()));
        assert!(externals.contains(&"lodash".to_string()));
        assert_eq!(externals.len(), 2);
    }

    #[test]
    fn external_edge_aggregated_per_workspace_package() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("packages/a")).unwrap();
        fs::create_dir_all(dir.path().join("packages/b")).unwrap();
        fs::write(
            dir.path().join("packages/a/package.json"),
            r#"{"name":"@org/a"}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/b/package.json"),
            r#"{"name":"@org/b"}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/a/x.ts"),
            r#"import React from "react";"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/a/y.ts"),
            r#"import React from "react";"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/b/x.ts"),
            r#"import React from "react";"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/b/y.ts"),
            r#"import React from "react";"#,
        )
        .unwrap();

        let g = index_folder(dir.path()).graph;

        let ext_id = external_node_id("react");
        let edges_to_react: Vec<_> = g.edges.iter().filter(|e| e.to == ext_id).collect();
        assert_eq!(edges_to_react.len(), 2);

        let froms: std::collections::HashSet<_> =
            edges_to_react.iter().map(|e| e.from.clone()).collect();
        assert!(froms.contains(&workspace_package_node_id("@org/a")));
        assert!(froms.contains(&workspace_package_node_id("@org/b")));
    }

    #[test]
    fn repo_without_externals_has_no_workspace_package_nodes() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name":"solo"}"#).unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "export const b = 1;").unwrap();

        let g = index_folder(dir.path()).graph;
        let has_pkg_node = g
            .nodes
            .values()
            .any(|n| n.kind == NodeKind::WorkspacePackage);
        let has_ext_node = g.nodes.values().any(|n| n.kind == NodeKind::External);
        assert!(!has_pkg_node);
        assert!(!has_ext_node);
    }

    #[test]
    fn tags_files_with_owning_package() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("packages/a")).unwrap();
        fs::create_dir_all(dir.path().join("packages/b")).unwrap();
        fs::write(
            dir.path().join("packages/a/package.json"),
            r#"{"name":"@org/a"}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/b/package.json"),
            r#"{"name":"@org/b"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("packages/a/index.ts"), "").unwrap();
        fs::write(dir.path().join("packages/b/index.ts"), "").unwrap();

        let g = index_folder(dir.path()).graph;
        let packages: Vec<_> = g.nodes.values().filter_map(|n| n.package.clone()).collect();
        assert!(packages.contains(&"@org/a".to_string()));
        assert!(packages.contains(&"@org/b".to_string()));
    }

    // --- import-mode coverage --------------------------------------------

    #[test]
    fn require_creates_workspace_edge() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.js"), r#"const b = require("./b");"#).unwrap();
        fs::write(dir.path().join("b.js"), "module.exports = 1;").unwrap();

        let g = index_folder(dir.path()).graph;
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn re_export_creates_workspace_edge() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("index.ts"), r#"export * from "./foo";"#).unwrap();
        fs::write(dir.path().join("foo.ts"), "export const x = 1;").unwrap();

        let g = index_folder(dir.path()).graph;
        assert_eq!(g.edges.len(), 1);
        assert!(
            g.edges
                .iter()
                .any(|e| e.from == "index.ts" && e.to == "foo.ts")
        );
    }

    #[test]
    fn dynamic_import_string_creates_workspace_edge() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"const b = import("./b");"#).unwrap();
        fs::write(dir.path().join("b.ts"), "export const x = 1;").unwrap();

        let result = index_folder(dir.path());
        assert_eq!(result.graph.edges.len(), 1);
        assert_eq!(result.unresolved_dynamic, 0);
    }

    #[test]
    fn template_prefix_dynamic_import_creates_one_edge_per_locale() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("loader.ts"),
            r#"const x = import(`./locales/${l}`);"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("locales")).unwrap();
        fs::write(dir.path().join("locales/en.ts"), "").unwrap();
        fs::write(dir.path().join("locales/fr.ts"), "").unwrap();
        fs::write(dir.path().join("locales/de.ts"), "").unwrap();

        let result = index_folder(dir.path());
        let edges_from_loader: Vec<_> = result
            .graph
            .edges
            .iter()
            .filter(|e| e.from == "loader.ts")
            .collect();
        assert_eq!(edges_from_loader.len(), 3);
        assert_eq!(result.unresolved_dynamic, 0);
    }

    #[test]
    fn truly_dynamic_import_counted_in_status() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"const m = import(modName);"#).unwrap();
        let result = index_folder(dir.path());
        assert_eq!(result.unresolved_dynamic, 1);
        assert!(result.graph.edges.is_empty());
    }

    #[test]
    fn malformed_file_is_skipped_but_still_appears_as_target() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("broken.ts"),
            "import { from './still-broken'",
        )
        .unwrap();
        fs::write(
            dir.path().join("consumer.ts"),
            r#"import { x } from "./broken";"#,
        )
        .unwrap();

        let result = index_folder(dir.path());
        assert!(
            result
                .graph
                .nodes
                .values()
                .any(|node| node.label == "broken.ts")
        );
        assert!(
            result
                .graph
                .edges
                .iter()
                .any(|edge| edge.from == "consumer.ts" && edge.to == "broken.ts")
        );
        assert!(result.errors.iter().any(|error| {
            matches!(
                error,
                GruffError::ParseFile { path, .. } if path.ends_with("broken.ts")
            )
        }));
    }

    #[test]
    fn tsconfig_paths_alias_resolves_to_workspace_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name":"root"}"#).unwrap();
        fs::write(
            dir.path().join("tsconfig.json"),
            r#"{
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": { "@myorg/shared/*": ["packages/shared/src/*"] }
                }
            }"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("packages/shared/src")).unwrap();
        fs::write(dir.path().join("packages/shared/src/utils.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("apps/web/src")).unwrap();
        fs::write(
            dir.path().join("apps/web/src/index.ts"),
            r#"import { x } from "@myorg/shared/utils";"#,
        )
        .unwrap();

        let g = index_folder(dir.path()).graph;
        let has_alias_edge = g.edges.iter().any(|e| {
            e.from.ends_with("apps/web/src/index.ts")
                && e.to.ends_with("packages/shared/src/utils.ts")
        });
        assert!(has_alias_edge, "missing alias edge in {:?}", g.edges);
    }

    #[test]
    fn workspace_package_name_resolves_to_entry_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("packages/shared/src")).unwrap();
        fs::write(
            dir.path().join("packages/shared/package.json"),
            r#"{"name":"@org/shared","main":"src/index.ts"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("packages/shared/src/index.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("packages/web/src")).unwrap();
        fs::write(
            dir.path().join("packages/web/package.json"),
            r#"{"name":"@org/web"}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/web/src/main.ts"),
            r#"import { x } from "@org/shared";"#,
        )
        .unwrap();

        let g = index_folder(dir.path()).graph;
        let has_ws_edge = g.edges.iter().any(|e| {
            e.from.ends_with("packages/web/src/main.ts")
                && e.to.ends_with("packages/shared/src/index.ts")
        });
        assert!(has_ws_edge, "missing workspace edge in {:?}", g.edges);
    }

    // --- incremental update --------------------------------------------------

    #[test]
    fn update_file_adds_new_import_edge() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::write(dir.path().join("b.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert!(indexer.graph.edges.is_empty());

        // Add an import from a.ts → b.ts and reindex just a.ts.
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        let diff = indexer.update_file(&dir.path().join("a.ts"));

        assert_eq!(diff.added_edges.len(), 1);
        assert_eq!(diff.removed_edges.len(), 0);
        assert_eq!(indexer.graph.edges.len(), 1);
    }

    #[test]
    fn update_file_removes_deleted_import_edge() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert_eq!(indexer.graph.edges.len(), 1);

        // Remove the import from a.ts and reindex.
        fs::write(dir.path().join("a.ts"), "").unwrap();
        let diff = indexer.update_file(&dir.path().join("a.ts"));

        assert_eq!(diff.removed_edges.len(), 1);
        assert!(indexer.graph.edges.is_empty());
    }

    #[test]
    fn update_file_adds_node_for_new_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert_eq!(indexer.graph.nodes.len(), 1);

        // Creating a new file and notifying the indexer.
        fs::write(dir.path().join("b.ts"), r#"import { a } from "./a";"#).unwrap();
        let diff = indexer.update_file(&dir.path().join("b.ts"));

        assert_eq!(diff.added_nodes.len(), 1);
        assert_eq!(diff.added_edges.len(), 1);
        assert_eq!(indexer.graph.nodes.len(), 2);
        assert_eq!(indexer.graph.edges.len(), 1);
    }

    #[test]
    fn remove_file_drops_node_and_incident_edges() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert_eq!(indexer.graph.nodes.len(), 2);
        assert_eq!(indexer.graph.edges.len(), 1);

        // Simulate b.ts being deleted. The node and incoming edge from a.ts
        // must both go.
        fs::remove_file(dir.path().join("b.ts")).unwrap();
        let diff = indexer.remove_file(&dir.path().join("b.ts"));

        assert_eq!(diff.removed_nodes.len(), 1);
        assert_eq!(diff.removed_edges.len(), 1);
        assert_eq!(indexer.graph.nodes.len(), 1);
        assert!(indexer.graph.edges.is_empty());
    }

    #[test]
    fn rescan_reconciles_drift() {
        // Mirrors the Cmd+R recovery path. Two files exist at scan time; we
        // mutate the filesystem behind the indexer's back (simulating missed
        // events) and then rescan. The final graph must reflect disk exactly.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"import {b} from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert_eq!(indexer.graph.nodes.len(), 2);
        assert_eq!(indexer.graph.edges.len(), 1);

        // Drift: add a new file, remove an existing one, change an import —
        // none of which are notified to the indexer.
        fs::remove_file(dir.path().join("b.ts")).unwrap();
        fs::write(dir.path().join("c.ts"), "").unwrap();
        fs::write(dir.path().join("a.ts"), r#"import {c} from "./c";"#).unwrap();

        indexer.rescan();
        assert_eq!(indexer.graph.nodes.len(), 2);
        assert_eq!(indexer.graph.edges.len(), 1);
        let labels: std::collections::HashSet<_> = indexer
            .graph
            .nodes
            .values()
            .map(|n| n.label.clone())
            .collect();
        assert!(labels.contains("a.ts"));
        assert!(labels.contains("c.ts"));
        assert!(!labels.contains("b.ts"));
    }

    #[test]
    fn update_external_add_and_remove_adjusts_synthetic_nodes() {
        // Adding the first import of `react` should create both the external
        // leaf and the workspace-package aggregator. Removing the last import
        // of `react` should drop both.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name":"@org/app"}"#).unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert_eq!(
            indexer
                .graph
                .nodes
                .values()
                .filter(|n| matches!(n.kind, NodeKind::External))
                .count(),
            0
        );

        // Add the import.
        fs::write(dir.path().join("a.ts"), r#"import React from "react";"#).unwrap();
        let diff = indexer.update_file(&dir.path().join("a.ts"));
        // One external node + one workspace-pkg aggregator + one edge.
        assert_eq!(diff.added_nodes.len(), 2);
        assert_eq!(diff.added_edges.len(), 1);

        // Remove it again.
        fs::write(dir.path().join("a.ts"), "").unwrap();
        let diff = indexer.update_file(&dir.path().join("a.ts"));
        assert!(diff.removed_edges.len() >= 1);
        // After removal the external leaf and aggregator should be gone.
        let has_external = indexer
            .graph
            .nodes
            .values()
            .any(|n| matches!(n.kind, NodeKind::External));
        assert!(!has_external, "external react leaf should be dropped");
    }

    // --- filter gate: tests, configs, declaration files ------------------

    #[test]
    fn test_files_excluded_by_default() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("foo.ts"), "").unwrap();
        fs::write(dir.path().join("foo.test.ts"), "").unwrap();
        fs::write(dir.path().join("bar.spec.tsx"), "").unwrap();

        let g = index_folder(dir.path()).graph;
        let labels: std::collections::HashSet<_> =
            g.nodes.values().map(|n| n.label.clone()).collect();
        assert!(labels.contains("foo.ts"));
        assert!(!labels.contains("foo.test.ts"));
        assert!(!labels.contains("bar.spec.tsx"));
    }

    #[test]
    fn toggle_include_tests_brings_test_files_back() {
        // Flipping the toggle should make test files reappear without the
        // caller needing to rebuild the indexer — a rescan inside the
        // setter does the work.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::write(dir.path().join("a.test.ts"), r#"import { a } from "./a";"#).unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert!(!indexer.graph.nodes.values().any(|n| n.label == "a.test.ts"));

        indexer.set_include_tests(true);
        assert!(indexer.graph.nodes.values().any(|n| n.label == "a.test.ts"));
        // The edge from the test file into the production file should also
        // appear, because the rescan reparsed it.
        assert!(
            indexer
                .graph
                .edges
                .iter()
                .any(|e| e.from.ends_with("a.test.ts") && e.to.ends_with("a.ts"))
        );

        // Toggling off again removes them.
        indexer.set_include_tests(false);
        assert!(!indexer.graph.nodes.values().any(|n| n.label == "a.test.ts"));
    }

    #[test]
    fn config_files_are_skipped() {
        // Common config names should never become nodes — even if they have
        // valid JS/TS extensions and happen to import other workspace files.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("src.ts"), "").unwrap();
        fs::write(
            dir.path().join("vite.config.ts"),
            r#"import { x } from "./src";"#,
        )
        .unwrap();
        fs::write(dir.path().join("jest.config.js"), "module.exports = {};").unwrap();
        fs::write(dir.path().join("babel.config.cjs"), "module.exports = {};").unwrap();
        fs::write(dir.path().join(".eslintrc.js"), "module.exports = {};").unwrap();

        let g = index_folder(dir.path()).graph;
        let labels: std::collections::HashSet<_> =
            g.nodes.values().map(|n| n.label.clone()).collect();
        assert!(labels.contains("src.ts"));
        assert!(!labels.contains("vite.config.ts"));
        assert!(!labels.contains("jest.config.js"));
        assert!(!labels.contains("babel.config.cjs"));
        assert!(!labels.contains(".eslintrc.js"));
        // The config file's import of `./src` must not contribute an edge —
        // the config shouldn't have been parsed at all.
        assert!(
            g.edges.is_empty(),
            "configs must not produce edges: {:?}",
            g.edges
        );
    }

    #[test]
    fn declaration_files_appear_as_nodes() {
        // `.d.ts` files are type-only but still meaningful nodes so users
        // can see what's using them.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("types.d.ts"), "export type X = number;").unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();

        let g = index_folder(dir.path()).graph;
        assert!(
            g.nodes.values().any(|n| n.label == "types.d.ts"),
            "expected types.d.ts node, got {:?}",
            g.nodes.values().map(|n| &n.label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn type_only_import_resolves_to_declaration_file() {
        // `import type { Foo } from './types'` must land on `types.d.ts`
        // when no regular `.ts` exists — TypeScript's resolution order.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("types.d.ts"), "export type Foo = number;").unwrap();
        fs::write(
            dir.path().join("a.ts"),
            r#"import type { Foo } from "./types";"#,
        )
        .unwrap();

        let g = index_folder(dir.path()).graph;
        let has_type_edge = g
            .edges
            .iter()
            .any(|e| e.from.ends_with("a.ts") && e.to.ends_with("types.d.ts"));
        assert!(has_type_edge, "missing type-only edge in {:?}", g.edges);
    }

    #[test]
    fn declaration_file_preferred_only_when_no_ts_alternative() {
        // When both `types.ts` and `types.d.ts` exist, the regular `.ts`
        // wins — mirrors TypeScript's own priority.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("types.ts"), "export const x = 1;").unwrap();
        fs::write(dir.path().join("types.d.ts"), "export const x: number;").unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { x } from "./types";"#).unwrap();

        let g = index_folder(dir.path()).graph;
        let edge_to_ts = g
            .edges
            .iter()
            .any(|e| e.from.ends_with("a.ts") && e.to == "types.ts");
        assert!(edge_to_ts, "expected edge to types.ts, got {:?}", g.edges);
    }
}
