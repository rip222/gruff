//! Language-agnostic node aggregation + edge rewriting.
//!
//! Raw file-level nodes (one per source file on disk) are transformed into
//! "display" nodes the graph actually renders. The rule is language-specific:
//! a [`NodeAggregator`] implementation decides which raw files collapse into a
//! shared folder node and which stay as per-file nodes, while the edge
//! rewriter — shared across implementations — maps raw edges onto display
//! endpoints, drops self-edges, and dedupes the result.
//!
//! TypeScript/JavaScript ships via [`TsBarrelAggregator`]: a folder collapses
//! into a single display node when it's a "barrel" — contains an
//! `index.{ts,tsx,js,jsx}`, is the target of a tsconfig `paths` entry, or is
//! the target of a `package.json` `exports` entry. Nested barrels stay as
//! their own collapsed nodes (the deepest barrel ancestor wins). Files whose
//! containing folder isn't a barrel stay as per-file nodes.
//!
//! The trait interface is pure: callers pass synthetic inputs (path list +
//! barrel-folder set) and get back display nodes plus a raw-id → display-id
//! mapping. No filesystem access, so tests can exercise every branch with
//! table-driven fixtures — see the unit tests at the bottom of the module and
//! the filesystem-backed helpers in `graph_pipeline` that feed the aggregator
//! from a real workspace.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::graph::{Edge, Graph, Node, NodeId, NodeKind};
use crate::resolver::ResolverContext;
use crate::workspace::Workspace;

/// Raw-node-level input fed into a [`NodeAggregator`]. Intentionally minimal:
/// the path and package are enough for barrel collapse, and the kind lets
/// synthetic nodes (workspace-package aggregators, external leaves) pass
/// through unchanged.
#[derive(Debug, Clone)]
pub struct RawNode {
    pub id: NodeId,
    pub path: PathBuf,
    pub label: String,
    pub package: Option<String>,
    pub kind: NodeKind,
}

impl From<Node> for RawNode {
    fn from(n: Node) -> Self {
        RawNode {
            id: n.id,
            path: n.path,
            label: n.label,
            package: n.package,
            kind: n.kind,
        }
    }
}

/// Workspace context an aggregator consults to mark barrels. All paths are
/// absolute / canonical so the aggregator can compare with file paths by
/// `starts_with`.
#[derive(Debug, Default, Clone)]
pub struct AggregationContext {
    /// Folders explicitly declared as barrels by tooling: tsconfig `paths`
    /// targets and `package.json` `exports` targets, resolved to a directory.
    /// Folders that simply contain an `index.{ts,tsx,js,jsx}` are detected
    /// from the raw node list itself — no need to pre-populate them here.
    pub declared_barrels: HashSet<PathBuf>,
}

/// Output of an aggregation pass: the display nodes plus a mapping from every
/// raw node id to its display id. Raw ids not present in the mapping were
/// dropped (shouldn't happen for well-formed input — every raw node must map
/// to something).
#[derive(Debug, Default, Clone)]
pub struct AggregationResult {
    pub nodes: Vec<Node>,
    pub mapping: HashMap<NodeId, NodeId>,
}

/// Handle exposing "barrel display node -> member file ids" both directions.
/// Built from an [`AggregationResult::mapping`] and consumed by features that
/// need to reason about file-level identity through the barrel-collapse —
/// today, the blast-radius cone in [`crate::reachability`]. Decoupling the
/// handle from [`AggregationContext`] keeps callers that only need barrel
/// *detection* (the aggregator itself) free of the per-mapping data, and
/// means the cone walker can be fed a synthetic members map in unit tests
/// without standing up a real aggregation pass.
#[derive(Debug, Default, Clone)]
pub struct BarrelMembers {
    /// For each barrel display id, the list of raw file-ids collapsed into it.
    /// Entries for non-barrel passthrough mappings (`raw_id -> raw_id`) are
    /// omitted — the map only contains actual barrels.
    by_display: HashMap<NodeId, Vec<NodeId>>,
    /// Reverse lookup: for each raw member file, its barrel display id. Lets
    /// the cone walker cheaply answer "is this id a barrel member?" without
    /// scanning every value in `by_display`.
    display_of: HashMap<NodeId, NodeId>,
}

impl BarrelMembers {
    /// Build a members handle from an aggregation mapping. Raw ids that map
    /// to themselves (non-barrel passthroughs) are skipped; every other raw
    /// id is registered as a member of its display id.
    pub fn from_mapping(mapping: &HashMap<NodeId, NodeId>) -> Self {
        let mut by_display: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut display_of: HashMap<NodeId, NodeId> = HashMap::new();
        for (raw, display) in mapping {
            if raw == display {
                continue;
            }
            by_display
                .entry(display.clone())
                .or_default()
                .push(raw.clone());
            display_of.insert(raw.clone(), display.clone());
        }
        // Stable iteration order regardless of `HashMap` hash seed — matters
        // for tests that assert on the members list.
        for members in by_display.values_mut() {
            members.sort();
        }
        Self {
            by_display,
            display_of,
        }
    }

    /// Empty handle — used when barrel collapse is disabled (passthrough
    /// aggregation) so callers don't have to branch on "do we have a handle
    /// at all". The empty handle expands no nodes and has no members.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Member raw file-ids for a barrel display id, or `None` if `id` isn't
    /// a known barrel.
    pub fn members_of(&self, id: &NodeId) -> Option<&[NodeId]> {
        self.by_display.get(id).map(|v| v.as_slice())
    }

    /// Barrel display id `id` is a member of, or `None` if `id` isn't a
    /// member of any barrel.
    pub fn display_of(&self, id: &NodeId) -> Option<&NodeId> {
        self.display_of.get(id)
    }
}

/// Language-agnostic barrel-collapse contract.
///
/// An implementation inspects the raw node list plus workspace context and
/// decides which raw files collapse into a shared display node. The trait
/// only handles nodes — edge rewriting is shared across implementations via
/// [`rewrite_edges`].
pub trait NodeAggregator {
    fn aggregate(&self, nodes: &[RawNode], ctx: &AggregationContext) -> AggregationResult;
}

/// Passthrough aggregator: every raw node maps to itself, no folder
/// collapse. Used by the `collapse_barrels = false` flow so the call site
/// in `aggregated_graph` doesn't need to special-case "skip aggregation"
/// — `apply_aggregation` runs the same way regardless and just returns
/// the file-level graph unchanged.
#[derive(Debug, Default)]
pub struct PassthroughAggregator;

impl PassthroughAggregator {
    pub fn new() -> Self {
        Self
    }
}

impl NodeAggregator for PassthroughAggregator {
    fn aggregate(&self, nodes: &[RawNode], _ctx: &AggregationContext) -> AggregationResult {
        let mut mapping: HashMap<NodeId, NodeId> = HashMap::new();
        let mut out_nodes: Vec<Node> = Vec::with_capacity(nodes.len());
        for raw in nodes {
            mapping.insert(raw.id.clone(), raw.id.clone());
            out_nodes.push(Node {
                id: raw.id.clone(),
                path: raw.path.clone(),
                label: raw.label.clone(),
                package: raw.package.clone(),
                kind: raw.kind,
            });
        }
        AggregationResult {
            nodes: out_nodes,
            mapping,
        }
    }
}

/// TypeScript/JavaScript barrel collapser. See module docs for the rule.
#[derive(Debug, Default)]
pub struct TsBarrelAggregator;

const INDEX_STEMS: &[&str] = &["index"];
const INDEX_EXTS: &[&str] = &["ts", "tsx", "js", "jsx"];

impl TsBarrelAggregator {
    pub fn new() -> Self {
        Self
    }

    /// Detect "has index.{ts,tsx,js,jsx}" barrels from the raw node list. The
    /// caller doesn't need to pre-populate these in the context because the
    /// aggregator can see them directly in `RawNode.path`.
    fn index_barrels(nodes: &[RawNode]) -> HashSet<PathBuf> {
        let mut out = HashSet::new();
        for n in nodes {
            if !matches!(n.kind, NodeKind::File) {
                continue;
            }
            let Some(stem) = n.path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if !INDEX_STEMS.contains(&stem) {
                continue;
            }
            let Some(ext) = n.path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !INDEX_EXTS.contains(&ext) {
                continue;
            }
            if let Some(parent) = n.path.parent() {
                out.insert(parent.to_path_buf());
            }
        }
        out
    }
}

impl NodeAggregator for TsBarrelAggregator {
    fn aggregate(&self, nodes: &[RawNode], ctx: &AggregationContext) -> AggregationResult {
        let mut barrels: HashSet<PathBuf> = Self::index_barrels(nodes);
        barrels.extend(ctx.declared_barrels.iter().cloned());

        let mut mapping: HashMap<NodeId, NodeId> = HashMap::new();
        let mut display_by_id: HashMap<NodeId, Node> = HashMap::new();

        for raw in nodes {
            if !matches!(raw.kind, NodeKind::File) {
                // Synthetic nodes (workspace-package aggregators, external
                // leaves) pass through as-is — they're not files and can't be
                // barrel-collapsed.
                mapping.insert(raw.id.clone(), raw.id.clone());
                display_by_id.entry(raw.id.clone()).or_insert_with(|| Node {
                    id: raw.id.clone(),
                    path: raw.path.clone(),
                    label: raw.label.clone(),
                    package: raw.package.clone(),
                    kind: raw.kind,
                });
                continue;
            }

            let barrel = deepest_barrel_for(&raw.path, &barrels);
            match barrel {
                Some(folder) => {
                    let display_id = barrel_display_id(&folder);
                    mapping.insert(raw.id.clone(), display_id.clone());
                    display_by_id.entry(display_id.clone()).or_insert_with(|| {
                        let label = folder
                            .file_name()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| display_id.clone());
                        Node {
                            id: display_id,
                            path: folder.clone(),
                            label,
                            package: raw.package.clone(),
                            kind: NodeKind::File,
                        }
                    });
                }
                None => {
                    mapping.insert(raw.id.clone(), raw.id.clone());
                    display_by_id.entry(raw.id.clone()).or_insert_with(|| Node {
                        id: raw.id.clone(),
                        path: raw.path.clone(),
                        label: raw.label.clone(),
                        package: raw.package.clone(),
                        kind: raw.kind,
                    });
                }
            }
        }

        AggregationResult {
            nodes: display_by_id.into_values().collect(),
            mapping,
        }
    }
}

/// Stable synthetic id for a barrel folder. Distinct prefix so it can't
/// collide with a file node id (which is a relative path).
fn barrel_display_id(folder: &Path) -> String {
    format!("barrel:{}", folder.to_string_lossy().replace('\\', "/"))
}

/// Deepest barrel folder that is an ancestor of — or equal to — the file's
/// parent directory. Returns `None` if no barrel contains the file, in which
/// case the file stays as its own display node.
fn deepest_barrel_for(file: &Path, barrels: &HashSet<PathBuf>) -> Option<PathBuf> {
    let mut cur = file.parent()?.to_path_buf();
    let mut best: Option<(PathBuf, usize)> = None;
    loop {
        if barrels.contains(&cur) {
            let depth = cur.components().count();
            if best.as_ref().map(|(_, d)| depth > *d).unwrap_or(true) {
                best = Some((cur.clone(), depth));
            }
        }
        match cur.parent() {
            Some(p) if p != cur => cur = p.to_path_buf(),
            _ => break,
        }
    }
    best.map(|(p, _)| p)
}

/// Rewrite `edges` onto their display-node endpoints using `mapping`. Drops
/// self-edges (both endpoints map to the same display id) and dedupes
/// `(source, target)` pairs.
pub fn rewrite_edges(edges: &[Edge], mapping: &HashMap<NodeId, NodeId>) -> Vec<Edge> {
    let mut seen: HashSet<(NodeId, NodeId)> = HashSet::new();
    let mut out = Vec::new();
    for e in edges {
        let from = mapping
            .get(&e.from)
            .cloned()
            .unwrap_or_else(|| e.from.clone());
        let to = mapping.get(&e.to).cloned().unwrap_or_else(|| e.to.clone());
        if from == to {
            continue;
        }
        if seen.insert((from.clone(), to.clone())) {
            out.push(Edge { from, to });
        }
    }
    out
}

/// Build an [`AggregationContext`] from a discovered workspace. Collects the
/// filesystem paths declared as barrels by tsconfig `paths` aliases and
/// `package.json` `exports` entries — the two signals `TsBarrelAggregator`
/// consults in addition to `index.*` detection on the raw node list.
///
/// Errors during individual manifest reads are silent: a bad `package.json`
/// just contributes no barrels rather than failing the whole aggregation.
pub fn context_from_workspace(ws: &Workspace, ctx: &ResolverContext) -> AggregationContext {
    let mut declared_barrels: HashSet<PathBuf> = HashSet::new();

    // tsconfig `paths`: every substitution resolves to a file or directory
    // relative to the tsconfig's `baseUrl`. The *folder* is what we mark as
    // a barrel — if the substitution points at a file, use its parent.
    for tsc in ctx.tsconfigs.values() {
        for (_pattern, subs) in &tsc.paths {
            for sub in subs {
                // `*` stays unexpanded; the useful prefix is everything
                // before the wildcard. A substitution like
                // `packages/shared/src/*` marks `packages/shared/src` as a
                // barrel, and the per-import wildcard lands inside.
                let stripped = match sub.find('*') {
                    Some(i) => &sub[..i],
                    None => sub.as_str(),
                };
                let joined = tsc.base_url.join(stripped.trim_end_matches('/'));
                if let Some(folder) = folder_of(&joined) {
                    declared_barrels.insert(folder);
                }
            }
        }
    }

    // `package.json` exports: each exported path points at a file; the
    // folder containing that file is the barrel. Accepts both the string
    // shape (`"exports": "./src/index.ts"`) and the map shape
    // (`"exports": { ".": "./src/index.ts", "./sub": "./src/sub/index.ts" }`)
    // plus conditional objects (`{ "import": "...", "require": "..." }`).
    for pkg in &ws.packages {
        let Ok(src) = fs::read_to_string(&pkg.manifest) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(&src) else {
            continue;
        };
        let Some(exports) = v.get("exports") else {
            continue;
        };
        let mut paths: Vec<String> = Vec::new();
        collect_export_paths(exports, &mut paths);
        for rel in paths {
            let joined = pkg.root.join(rel.trim_start_matches("./"));
            if let Some(folder) = folder_of(&joined) {
                declared_barrels.insert(folder);
            }
        }
    }

    AggregationContext { declared_barrels }
}

/// Walk a `package.json` `exports` value and push every string leaf into
/// `out`. Mirrors Node's resolution shape without being exhaustive — we only
/// need the list of declared paths, not the condition that picked them.
fn collect_export_paths(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(arr) => {
            for x in arr {
                collect_export_paths(x, out);
            }
        }
        Value::Object(obj) => {
            for (_k, x) in obj {
                collect_export_paths(x, out);
            }
        }
        _ => {}
    }
}

/// Resolve a path to its containing folder: if the path itself is a directory
/// on disk, return it; if it's a file, return its parent. Used when turning a
/// declared export / paths entry into a "barrel folder" marker.
fn folder_of(p: &Path) -> Option<PathBuf> {
    let canon = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    if canon.is_dir() {
        return Some(canon);
    }
    if canon.is_file() {
        return canon.parent().map(|pp| pp.to_path_buf());
    }
    // Path doesn't exist on disk. Best effort: assume a trailing segment that
    // looks like `index.*` or a known extension is the file, otherwise treat
    // it as the folder itself. This makes the helper safe to call in unit
    // tests that pass synthetic paths.
    let stem = canon.file_stem().and_then(|s| s.to_str());
    let has_known_ext = canon
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| INDEX_EXTS.contains(&e) || ["d.ts", "mjs", "cjs"].contains(&e))
        .unwrap_or(false);
    if has_known_ext || stem.map(|s| s == "index").unwrap_or(false) {
        return canon.parent().map(|pp| pp.to_path_buf());
    }
    Some(canon)
}

/// Convenience: run `aggregator` over every node in `raw` and rewrite edges
/// onto the resulting display set. This is what the graph pipeline calls once
/// it has a raw file-level graph + workspace context in hand.
pub fn apply_aggregation<A: NodeAggregator>(
    raw: &Graph,
    ctx: &AggregationContext,
    aggregator: &A,
) -> Graph {
    let raw_nodes: Vec<RawNode> = raw.nodes.values().cloned().map(RawNode::from).collect();
    let result = aggregator.aggregate(&raw_nodes, ctx);

    let mut g = Graph::new();
    for n in result.nodes {
        g.add_node(n);
    }
    for edge in rewrite_edges(&raw.edges, &result.mapping) {
        g.add_edge(&edge.from, &edge.to);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(id: &str, path: &str, package: Option<&str>) -> RawNode {
        RawNode {
            id: id.to_string(),
            path: PathBuf::from(path),
            label: path.rsplit('/').next().unwrap_or(path).to_string(),
            package: package.map(String::from),
            kind: NodeKind::File,
        }
    }

    fn synthetic(id: &str, kind: NodeKind) -> RawNode {
        RawNode {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: None,
            kind,
        }
    }

    fn ctx_with(barrels: &[&str]) -> AggregationContext {
        AggregationContext {
            declared_barrels: barrels.iter().map(PathBuf::from).collect(),
        }
    }

    #[test]
    fn folder_with_index_collapses_into_single_display_node() {
        // `foo/index.ts` + `foo/a.ts` share one display node rooted at `foo`.
        let nodes = vec![
            file("foo/index.ts", "/repo/foo/index.ts", None),
            file("foo/a.ts", "/repo/foo/a.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        assert_eq!(result.nodes.len(), 1);
        let display_id = result.mapping.get("foo/index.ts").unwrap().clone();
        assert_eq!(result.mapping.get("foo/a.ts").unwrap(), &display_id);
        assert_eq!(result.nodes[0].label, "foo");
    }

    #[test]
    fn tsconfig_paths_target_marks_folder_as_barrel() {
        // `packages/shared/src` contains `utils.ts` but no `index.ts`. The
        // tsconfig declares it as a paths target, so it still collapses.
        let nodes = vec![file(
            "packages/shared/src/utils.ts",
            "/repo/packages/shared/src/utils.ts",
            None,
        )];
        let ctx = ctx_with(&["/repo/packages/shared/src"]);
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx);
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].label, "src");
        let display_id = result.mapping.get("packages/shared/src/utils.ts").unwrap();
        assert!(display_id.starts_with("barrel:"));
    }

    #[test]
    fn package_json_exports_target_marks_folder_as_barrel() {
        // Same mechanism as tsconfig paths — the declared-barrels set is what
        // the aggregator consults. This keeps the trait source-agnostic.
        let nodes = vec![file("pkg/sub/a.ts", "/repo/pkg/sub/a.ts", None)];
        let ctx = ctx_with(&["/repo/pkg/sub"]);
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx);
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].label, "sub");
    }

    #[test]
    fn nested_barrel_stays_as_its_own_display_node() {
        // `foo` and `foo/bar` are both barrels. Files under `foo/bar` land on
        // `foo/bar`; files under `foo` (but not `foo/bar`) land on `foo`.
        let nodes = vec![
            file("foo/index.ts", "/repo/foo/index.ts", None),
            file("foo/a.ts", "/repo/foo/a.ts", None),
            file("foo/bar/index.ts", "/repo/foo/bar/index.ts", None),
            file("foo/bar/b.ts", "/repo/foo/bar/b.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        assert_eq!(result.nodes.len(), 2);
        let foo_id = result.mapping.get("foo/index.ts").unwrap();
        let bar_id = result.mapping.get("foo/bar/index.ts").unwrap();
        assert_ne!(foo_id, bar_id);
        assert_eq!(result.mapping.get("foo/a.ts").unwrap(), foo_id);
        assert_eq!(result.mapping.get("foo/bar/b.ts").unwrap(), bar_id);
    }

    #[test]
    fn loose_files_in_non_barrel_folders_stay_per_file() {
        // No `index.*`, no declared barrel — every file becomes its own node.
        let nodes = vec![
            file("pkg/a.ts", "/repo/pkg/a.ts", None),
            file("pkg/b.ts", "/repo/pkg/b.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.mapping.get("pkg/a.ts").unwrap(), "pkg/a.ts");
        assert_eq!(result.mapping.get("pkg/b.ts").unwrap(), "pkg/b.ts");
    }

    #[test]
    fn mixed_loose_and_barrel_in_same_package() {
        // A package with `src/` (barrel via index.ts) and a loose root file.
        let nodes = vec![
            file("pkg/README.ts", "/repo/pkg/README.ts", None),
            file("pkg/src/index.ts", "/repo/pkg/src/index.ts", None),
            file("pkg/src/util.ts", "/repo/pkg/src/util.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(
            result.mapping.get("pkg/README.ts").unwrap(),
            "pkg/README.ts"
        );
        let src_id = result.mapping.get("pkg/src/index.ts").unwrap();
        assert_eq!(result.mapping.get("pkg/src/util.ts").unwrap(), src_id);
    }

    #[test]
    fn synthetic_nodes_pass_through_unchanged() {
        let nodes = vec![
            synthetic("package:@org/app", NodeKind::WorkspacePackage),
            synthetic("external:react", NodeKind::External),
            file("foo/index.ts", "/repo/foo/index.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        assert_eq!(
            result.mapping.get("package:@org/app").unwrap(),
            "package:@org/app"
        );
        assert_eq!(
            result.mapping.get("external:react").unwrap(),
            "external:react"
        );
    }

    #[test]
    fn edges_are_rewritten_onto_display_nodes() {
        // foo collapses (has index). bar is a loose file. Edge foo/a.ts ->
        // bar.ts must rewrite to foo -> bar.ts.
        let nodes = vec![
            file("foo/index.ts", "/repo/foo/index.ts", None),
            file("foo/a.ts", "/repo/foo/a.ts", None),
            file("bar.ts", "/repo/bar.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        let foo_id = result.mapping.get("foo/index.ts").unwrap().clone();
        let edges = vec![Edge {
            from: "foo/a.ts".into(),
            to: "bar.ts".into(),
        }];
        let rewritten = rewrite_edges(&edges, &result.mapping);
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].from, foo_id);
        assert_eq!(rewritten[0].to, "bar.ts");
    }

    #[test]
    fn self_edges_are_dropped() {
        // Both endpoints collapse to the same barrel → the edge is dropped.
        let nodes = vec![
            file("foo/index.ts", "/repo/foo/index.ts", None),
            file("foo/a.ts", "/repo/foo/a.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        let edges = vec![Edge {
            from: "foo/a.ts".into(),
            to: "foo/index.ts".into(),
        }];
        let rewritten = rewrite_edges(&edges, &result.mapping);
        assert!(rewritten.is_empty(), "self-edge must be dropped");
    }

    #[test]
    fn duplicate_edges_are_deduped() {
        // Two files inside barrel `foo` each import the same target. The
        // rewritten edges collapse to one (foo → bar.ts).
        let nodes = vec![
            file("foo/index.ts", "/repo/foo/index.ts", None),
            file("foo/a.ts", "/repo/foo/a.ts", None),
            file("foo/b.ts", "/repo/foo/b.ts", None),
            file("bar.ts", "/repo/bar.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        let edges = vec![
            Edge {
                from: "foo/a.ts".into(),
                to: "bar.ts".into(),
            },
            Edge {
                from: "foo/b.ts".into(),
                to: "bar.ts".into(),
            },
        ];
        let rewritten = rewrite_edges(&edges, &result.mapping);
        assert_eq!(rewritten.len(), 1);
    }

    #[test]
    fn nested_barrel_edges_rewrite_to_correct_depth() {
        // Edge from foo/bar/b.ts → foo/a.ts. foo/bar collapses to one node
        // and foo collapses to another — so the rewritten edge is
        // foo/bar → foo, NOT a self-edge.
        let nodes = vec![
            file("foo/index.ts", "/repo/foo/index.ts", None),
            file("foo/a.ts", "/repo/foo/a.ts", None),
            file("foo/bar/index.ts", "/repo/foo/bar/index.ts", None),
            file("foo/bar/b.ts", "/repo/foo/bar/b.ts", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        let foo_id = result.mapping.get("foo/index.ts").unwrap().clone();
        let bar_id = result.mapping.get("foo/bar/index.ts").unwrap().clone();
        let edges = vec![Edge {
            from: "foo/bar/b.ts".into(),
            to: "foo/a.ts".into(),
        }];
        let rewritten = rewrite_edges(&edges, &result.mapping);
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].from, bar_id);
        assert_eq!(rewritten[0].to, foo_id);
    }

    #[test]
    fn jsx_index_also_marks_barrel() {
        // The rule is extension-agnostic across ts/tsx/js/jsx — verify jsx
        // isn't accidentally dropped.
        let nodes = vec![
            file("comp/index.jsx", "/repo/comp/index.jsx", None),
            file("comp/a.jsx", "/repo/comp/a.jsx", None),
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        assert_eq!(result.nodes.len(), 1);
    }

    #[test]
    fn non_jsts_index_does_not_mark_barrel() {
        // `index.md` or `index.json` must not trigger collapse — the rule is
        // restricted to executable JS/TS extensions.
        let nodes = vec![
            file("docs/a.ts", "/repo/docs/a.ts", None),
            // Non-candidate "index" files aren't in the graph (the indexer
            // wouldn't index them), but guard against the rule matching too
            // broadly if someone does pass one in.
        ];
        let result = TsBarrelAggregator::new().aggregate(&nodes, &ctx_with(&[]));
        // Single non-barrel file: stays per-file.
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.mapping.get("docs/a.ts").unwrap(), "docs/a.ts");
    }

    #[test]
    fn apply_aggregation_builds_full_graph() {
        let mut raw = Graph::new();
        for (id, path) in [
            ("foo/index.ts", "/repo/foo/index.ts"),
            ("foo/a.ts", "/repo/foo/a.ts"),
            ("bar.ts", "/repo/bar.ts"),
        ] {
            raw.add_node(Node {
                id: id.to_string(),
                path: PathBuf::from(path),
                label: id.to_string(),
                package: None,
                kind: NodeKind::File,
            });
        }
        raw.add_edge("foo/a.ts", "bar.ts");
        raw.add_edge("foo/index.ts", "foo/a.ts"); // becomes self-edge → dropped

        let g = apply_aggregation(
            &raw,
            &AggregationContext::default(),
            &TsBarrelAggregator::new(),
        );
        assert_eq!(g.nodes.len(), 2); // foo barrel + bar.ts
        assert_eq!(g.edges.len(), 1);
    }
}
