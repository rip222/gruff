//! Pure hierarchical view of the graph for the sidebar's `Packages` pane.
//!
//! Builds a tree from the current [`Graph`](crate::graph::Graph):
//!
//! - Workspace **packages** each root a subtree. Under a package, files are
//!   nested by their directory hierarchy — a file at `packages/foo/src/a.ts`
//!   belonging to package `foo` is reached through folder nodes
//!   `src/` (the deepest directory prefix common to every file in `foo` is
//!   stripped so a user expanding `foo` sees its subtrees, not the repo path
//!   to the package root).
//! - A flat **`(unpackaged)`** bucket holds every file node that has no
//!   owning workspace package (stray files outside every declared package).
//! - A separate flat **`externals`** bucket holds every external dependency
//!   leaf (one entry per `node_modules` package the workspace imports).
//!
//! Ordering is alphabetical at every level: packages, externals, unpackaged
//! files, and folder/file siblings under each package.
//!
//! Checkbox state is tracked on every tree node so the sidebar can render a
//! checkbox at every level. File-level `CheckState` reflects the caller's
//! [`FilterState`] hide set — hidden files render as `Unchecked`, visible
//! files render as `Checked`. Package- and folder-level tristate cascade
//! lands in #23; until then parent rows always render `Checked` regardless
//! of their descendants.
//!
//! Kept egui-free and purely data-driven. See `tests` for structural
//! assertions that don't require a renderer.
//!
//! [`FilterState`]: crate::filter_state::FilterState

use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path, PathBuf};

use crate::graph::{Graph, NodeId, NodeKind};

/// Display label for the unpackaged bucket. Surfaced as a constant so the
/// renderer and tests agree on one spelling.
pub const UNPACKAGED_LABEL: &str = "(unpackaged)";

/// Display label for the externals bucket.
pub const EXTERNALS_LABEL: &str = "externals";

/// Three-state checkbox. File-level leaves use `Checked`/`Unchecked` to
/// reflect the caller's [`FilterState`] hide set. `Mixed` is reserved for
/// parent rows once tristate/cascade lands in #23; until then parents
/// always build as `Checked`.
///
/// [`FilterState`]: crate::filter_state::FilterState
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckState {
    Checked,
    Unchecked,
    /// Mixed: some descendants checked, others unchecked. Produced by a
    /// future filter layer; not yet built by [`PackageTree::build`].
    Mixed,
}

/// Single-level node under a package subtree. Folders nest folders and files;
/// files are leaves.
#[derive(Debug, Clone)]
pub enum FolderChild {
    Folder(FolderNode),
    File(FileLeaf),
}

impl FolderChild {
    /// Sort key for alphabetical ordering under a folder. Folders and files
    /// both sort by their display name so a user reading the list sees one
    /// combined alphabetical pass rather than "folders first, files second".
    fn sort_key(&self) -> &str {
        match self {
            FolderChild::Folder(f) => f.name.as_str(),
            FolderChild::File(f) => f.label.as_str(),
        }
    }

    pub fn check_state(&self) -> CheckState {
        match self {
            FolderChild::Folder(f) => f.check,
            FolderChild::File(f) => f.check,
        }
    }
}

/// Intermediate folder node. Owns its children; the children list is kept
/// alphabetical by [`PackageTree::build`].
#[derive(Debug, Clone)]
pub struct FolderNode {
    pub name: String,
    pub children: Vec<FolderChild>,
    pub check: CheckState,
}

/// Leaf file reference. Carries the node id so the sidebar can look up node
/// state (color, selection) in the live app without re-deriving anything.
#[derive(Debug, Clone)]
pub struct FileLeaf {
    pub id: NodeId,
    pub label: String,
    pub check: CheckState,
}

/// A workspace-package subtree — the first level under the `Packages` root.
#[derive(Debug, Clone)]
pub struct PackageNode {
    pub name: String,
    /// Stable color-palette index for this package, matching
    /// `GruffApp::package_indices`. Drives the sidebar's color swatch.
    pub color_index: Option<usize>,
    /// Directory prefix stripped from every file path before building the
    /// folder hierarchy. Kept around so callers can show it next to the
    /// package name if they want to; the sidebar in this slice doesn't.
    pub common_prefix: PathBuf,
    pub children: Vec<FolderChild>,
    pub check: CheckState,
}

/// A single external dependency leaf (e.g. `lodash`, `@org/utils`).
#[derive(Debug, Clone)]
pub struct ExternalLeaf {
    pub id: NodeId,
    pub label: String,
    pub check: CheckState,
}

/// Flat bucket of unpackaged files. Rendered as a single collapsible section
/// in the sidebar; items are alphabetical by label.
#[derive(Debug, Clone)]
pub struct UnpackagedBucket {
    pub files: Vec<FileLeaf>,
    pub check: CheckState,
}

/// Flat bucket of `node_modules` externals. Rendered as a single collapsible
/// section; items are alphabetical by name.
#[derive(Debug, Clone)]
pub struct ExternalsBucket {
    pub externals: Vec<ExternalLeaf>,
    pub check: CheckState,
}

/// Root of the sidebar tree. Owns the per-package subtrees and the two flat
/// buckets. Empty buckets are still present so the renderer can decide on a
/// consistent "hide when empty" policy in one place.
#[derive(Debug, Clone, Default)]
pub struct PackageTree {
    pub packages: Vec<PackageNode>,
    pub unpackaged: UnpackagedBucket,
    pub externals: ExternalsBucket,
}

impl Default for UnpackagedBucket {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            check: CheckState::Checked,
        }
    }
}

impl Default for ExternalsBucket {
    fn default() -> Self {
        Self {
            externals: Vec::new(),
            check: CheckState::Checked,
        }
    }
}

impl PackageTree {
    /// Build the tree from the current graph.
    ///
    /// `package_indices` is the app's `package_name → color_palette_index`
    /// map (see `GruffApp::package_indices`). A missing package name yields
    /// `None` for `color_index` so tests and downstream code that don't care
    /// about colors can pass an empty map.
    ///
    /// `hidden` is the current filter's hide set. File leaves whose id is in
    /// the set render as [`CheckState::Unchecked`]; all other leaves (and
    /// every parent-level `CheckState`) stay `Checked` in this slice —
    /// tristate/cascade for package and folder rows lands in #23.
    ///
    /// Alphabetical ordering is enforced at every level: packages, external
    /// leaves, unpackaged files, and folder/file siblings under each
    /// package.
    pub fn build(
        graph: &Graph,
        package_indices: &std::collections::HashMap<String, usize>,
        hidden: &HashSet<NodeId>,
    ) -> Self {
        // Partition nodes into three disjoint sets: workspace files grouped
        // by package name, externals, and unpackaged files. Synthetic
        // `WorkspacePackage` aggregator nodes don't appear in the tree —
        // packages are represented by their own subtree, not by the
        // aggregator leaf the indexer uses to bundle external edges.
        let mut by_package: BTreeMap<String, Vec<&crate::graph::Node>> = BTreeMap::new();
        let mut externals: Vec<&crate::graph::Node> = Vec::new();
        let mut unpackaged: Vec<&crate::graph::Node> = Vec::new();

        for node in graph.nodes.values() {
            match node.kind {
                NodeKind::External => externals.push(node),
                NodeKind::WorkspacePackage => {
                    // Skip synthetic aggregators — they aren't real files
                    // the user can toggle on or off. The containing package
                    // subtree already represents the package.
                }
                NodeKind::File => match node.package.as_deref() {
                    Some(name) => by_package
                        .entry(name.to_string())
                        .or_default()
                        .push(node),
                    None => unpackaged.push(node),
                },
            }
        }

        // BTreeMap gives us alphabetical package order for free; just walk
        // in iteration order.
        let packages: Vec<PackageNode> = by_package
            .into_iter()
            .map(|(name, nodes)| {
                let common_prefix = common_dir_prefix(nodes.iter().map(|n| n.path.as_path()));
                let children = build_children(&nodes, &common_prefix, hidden);
                PackageNode {
                    color_index: package_indices.get(&name).copied(),
                    name,
                    common_prefix,
                    children,
                    check: CheckState::Checked,
                }
            })
            .collect();

        // Alphabetical externals by label. Externals typically have the
        // package name as their label (see `node_label::display_label`).
        externals.sort_by(|a, b| a.label.cmp(&b.label));
        let externals = ExternalsBucket {
            externals: externals
                .into_iter()
                .map(|n| ExternalLeaf {
                    id: n.id.clone(),
                    label: n.label.clone(),
                    check: check_from_hidden(&n.id, hidden),
                })
                .collect(),
            check: CheckState::Checked,
        };

        // Flat list; keep alphabetical by display label. `display_label`
        // lives in `node_label` but we stay pure here and sort by the raw
        // `label` field — it's always the filename for `File` nodes, which
        // is stable enough for a deterministic order.
        unpackaged.sort_by(|a, b| a.label.cmp(&b.label));
        let unpackaged = UnpackagedBucket {
            files: unpackaged
                .into_iter()
                .map(|n| FileLeaf {
                    id: n.id.clone(),
                    label: n.label.clone(),
                    check: check_from_hidden(&n.id, hidden),
                })
                .collect(),
            check: CheckState::Checked,
        };

        PackageTree {
            packages,
            unpackaged,
            externals,
        }
    }

    /// True when the tree has no packages, no externals, and no unpackaged
    /// files. Saves the renderer one `.is_empty()` test per section.
    pub fn is_empty(&self) -> bool {
        self.packages.is_empty()
            && self.externals.externals.is_empty()
            && self.unpackaged.files.is_empty()
    }
}

/// Build the ordered alphabetical child list for a package subtree. Each
/// file's path is made relative to `common_prefix`, then walked component by
/// component to build folders. `hidden` threads through so leaf file
/// [`CheckState`]s reflect the current filter.
fn build_children(
    nodes: &[&crate::graph::Node],
    common_prefix: &Path,
    hidden: &HashSet<NodeId>,
) -> Vec<FolderChild> {
    // Insert into a temporary intermediate tree keyed by component name.
    // `BTreeMap` gives us alphabetical folder-vs-folder / file-vs-file
    // ordering for free; we only have to interleave the two at flatten time.
    #[derive(Default)]
    struct Inter {
        folders: BTreeMap<String, Inter>,
        files: BTreeMap<String, FileLeaf>,
    }

    let mut root = Inter::default();
    for node in nodes {
        let rel = node
            .path
            .strip_prefix(common_prefix)
            .unwrap_or(&node.path)
            .to_path_buf();
        insert_file(&mut root, &rel, node, hidden);
    }

    fn insert_file(
        inter: &mut Inter,
        rel: &Path,
        node: &crate::graph::Node,
        hidden: &HashSet<NodeId>,
    ) {
        let mut comps: Vec<String> = rel
            .components()
            .filter_map(|c| match c {
                Component::Normal(os) => Some(os.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect();
        if comps.is_empty() {
            // Degenerate: file path equals the prefix. Index it by label.
            inter.files.insert(
                node.label.clone(),
                FileLeaf {
                    id: node.id.clone(),
                    label: node.label.clone(),
                    check: check_from_hidden(&node.id, hidden),
                },
            );
            return;
        }
        let filename = comps.pop().expect("non-empty after pop guard");
        let mut cursor: &mut Inter = inter;
        for dir in comps {
            cursor = cursor.folders.entry(dir).or_default();
        }
        cursor.files.insert(
            filename.clone(),
            FileLeaf {
                id: node.id.clone(),
                label: node.label.clone(),
                check: check_from_hidden(&node.id, hidden),
            },
        );
    }

    fn flatten(inter: Inter) -> Vec<FolderChild> {
        // Merge folders and files into one alphabetical sibling list. We sort
        // after concatenating so a folder named `a` sits above a file named
        // `b.ts`, matching how a user reads a directory listing.
        let mut out: Vec<FolderChild> = Vec::with_capacity(inter.folders.len() + inter.files.len());
        for (name, child) in inter.folders {
            let children = flatten(child);
            out.push(FolderChild::Folder(FolderNode {
                name,
                children,
                check: CheckState::Checked,
            }));
        }
        for (_, f) in inter.files {
            out.push(FolderChild::File(f));
        }
        out.sort_by(|a, b| a.sort_key().cmp(b.sort_key()));
        out
    }

    flatten(root)
}

/// Map a leaf's id to a [`CheckState`] given the caller's hide set:
/// `Unchecked` when the id is hidden, `Checked` otherwise. Used uniformly
/// for every file leaf and every external leaf so the tree mirrors the
/// filter without pulling `FilterState` into this module.
fn check_from_hidden(id: &NodeId, hidden: &HashSet<NodeId>) -> CheckState {
    if hidden.contains(id) {
        CheckState::Unchecked
    } else {
        CheckState::Checked
    }
}

/// Longest directory prefix shared by every path in `paths`. Returns an
/// empty `PathBuf` when the input is empty or the paths share no root.
///
/// "Directory prefix" means whole components — `/a/b/c1.ts` and `/a/b/c2.ts`
/// share `/a/b`, not `/a/b/c`. We're prefixing folders, not filenames.
fn common_dir_prefix<'a, I>(paths: I) -> PathBuf
where
    I: IntoIterator<Item = &'a Path>,
{
    let mut iter = paths.into_iter();
    let Some(first) = iter.next() else {
        return PathBuf::new();
    };
    // Start from the first file's parent directory — we never want the
    // filename itself in the prefix.
    let mut prefix: Vec<Component<'_>> = first
        .parent()
        .map(|p| p.components().collect())
        .unwrap_or_default();

    for path in iter {
        let parent_components: Vec<Component<'_>> = path
            .parent()
            .map(|p| p.components().collect())
            .unwrap_or_default();
        let shared = prefix
            .iter()
            .zip(parent_components.iter())
            .take_while(|(a, b)| a == b)
            .count();
        prefix.truncate(shared);
        if prefix.is_empty() {
            break;
        }
    }

    let mut out = PathBuf::new();
    for c in prefix {
        out.push(c.as_os_str());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, Node, NodeKind};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn file(id: &str, path: &str, package: Option<&str>) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(path),
            label: PathBuf::from(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| id.to_string()),
            package: package.map(str::to_string),
            kind: NodeKind::File,
        }
    }

    fn external(name: &str) -> Node {
        Node {
            id: format!("external:{name}"),
            path: PathBuf::from(name),
            label: name.to_string(),
            package: None,
            kind: NodeKind::External,
        }
    }

    fn workspace_agg(name: &str) -> Node {
        Node {
            id: format!("package:{name}"),
            path: PathBuf::from(name),
            label: name.to_string(),
            package: Some(name.to_string()),
            kind: NodeKind::WorkspacePackage,
        }
    }

    /// Fixture matching the issue's acceptance criterion:
    /// "3 packages, nested folders, externals, unpackaged files → expected
    /// shape". Reused by multiple shape assertions below.
    fn build_fixture() -> (Graph, HashMap<String, usize>) {
        let mut g = Graph::new();
        // Package alpha: two files under src/, one nested under src/deep/.
        g.add_node(file(
            "packages/alpha/src/a.ts",
            "/repo/packages/alpha/src/a.ts",
            Some("alpha"),
        ));
        g.add_node(file(
            "packages/alpha/src/b.ts",
            "/repo/packages/alpha/src/b.ts",
            Some("alpha"),
        ));
        g.add_node(file(
            "packages/alpha/src/deep/c.ts",
            "/repo/packages/alpha/src/deep/c.ts",
            Some("alpha"),
        ));
        // Package beta: single file, no nesting.
        g.add_node(file(
            "packages/beta/index.ts",
            "/repo/packages/beta/index.ts",
            Some("beta"),
        ));
        // Package gamma: two files, one with nested folder.
        g.add_node(file(
            "packages/gamma/entry.ts",
            "/repo/packages/gamma/entry.ts",
            Some("gamma"),
        ));
        g.add_node(file(
            "packages/gamma/utils/helper.ts",
            "/repo/packages/gamma/utils/helper.ts",
            Some("gamma"),
        ));

        // Unpackaged files — should land in the flat unpackaged bucket.
        g.add_node(file("tools/scratch.ts", "/repo/tools/scratch.ts", None));
        g.add_node(file("README.ts", "/repo/README.ts", None));

        // Externals + one synthetic workspace aggregator that must be
        // ignored in the tree (package subtree already covers it).
        g.add_node(external("react"));
        g.add_node(external("@org/utils"));
        g.add_node(external("lodash"));
        g.add_node(workspace_agg("alpha"));

        let mut idx = HashMap::new();
        idx.insert("alpha".to_string(), 0);
        idx.insert("beta".to_string(), 1);
        idx.insert("gamma".to_string(), 2);

        (g, idx)
    }

    #[test]
    fn build_produces_expected_shape_from_mixed_graph() {
        let (g, idx) = build_fixture();
        let tree = PackageTree::build(&g, &idx, &HashSet::new());

        // Exactly three packages, alphabetical: alpha, beta, gamma.
        let pkg_names: Vec<&str> = tree.packages.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(pkg_names, vec!["alpha", "beta", "gamma"]);

        // Color indices propagate from the supplied map.
        assert_eq!(tree.packages[0].color_index, Some(0));
        assert_eq!(tree.packages[1].color_index, Some(1));
        assert_eq!(tree.packages[2].color_index, Some(2));

        // Externals rendered flat, alphabetical by name.
        let ext_names: Vec<&str> = tree
            .externals
            .externals
            .iter()
            .map(|e| e.label.as_str())
            .collect();
        assert_eq!(ext_names, vec!["@org/utils", "lodash", "react"]);

        // Unpackaged rendered flat, alphabetical by label. Two files exactly.
        let unpk: Vec<&str> = tree
            .unpackaged
            .files
            .iter()
            .map(|f| f.label.as_str())
            .collect();
        assert_eq!(unpk, vec!["README.ts", "scratch.ts"]);

        // Workspace-package aggregators must NOT appear as their own nodes.
        // Only the three real packages show up.
        assert_eq!(tree.packages.len(), 3);
    }

    #[test]
    fn alphabetical_ordering_at_every_level() {
        let (g, idx) = build_fixture();
        let tree = PackageTree::build(&g, &idx, &HashSet::new());

        // Packages alphabetical.
        let names: Vec<&str> = tree.packages.iter().map(|p| p.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "package order must be alphabetical");

        // Within each package, recursively: every sibling list alphabetical.
        fn check(children: &[FolderChild]) {
            let keys: Vec<&str> = children.iter().map(FolderChild::sort_key).collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(
                keys, sorted,
                "folder/file siblings must be alphabetical: {keys:?}"
            );
            for c in children {
                if let FolderChild::Folder(f) = c {
                    check(&f.children);
                }
            }
        }
        for pkg in &tree.packages {
            check(&pkg.children);
        }

        // Externals alphabetical.
        let ext: Vec<&str> = tree
            .externals
            .externals
            .iter()
            .map(|e| e.label.as_str())
            .collect();
        let mut sorted = ext.clone();
        sorted.sort();
        assert_eq!(ext, sorted);

        // Unpackaged alphabetical.
        let unpk: Vec<&str> = tree
            .unpackaged
            .files
            .iter()
            .map(|f| f.label.as_str())
            .collect();
        let mut sorted = unpk.clone();
        sorted.sort();
        assert_eq!(unpk, sorted);
    }

    #[test]
    fn package_subtree_nests_folders_and_files() {
        let (g, idx) = build_fixture();
        let tree = PackageTree::build(&g, &idx, &HashSet::new());

        // alpha has files all under src/, with one under src/deep/.
        // Common directory prefix is the package's src/, so alpha's
        // children are: `a.ts`, `b.ts`, and a folder `deep/` containing `c.ts`.
        let alpha = &tree.packages[0];
        assert_eq!(alpha.name, "alpha");

        let labels: Vec<&str> = alpha.children.iter().map(FolderChild::sort_key).collect();
        assert_eq!(labels, vec!["a.ts", "b.ts", "deep"]);

        // The `deep` folder contains exactly `c.ts`.
        let deep_children = match &alpha.children[2] {
            FolderChild::Folder(f) => &f.children,
            _ => panic!("expected `deep` to be a folder"),
        };
        let deep_labels: Vec<&str> =
            deep_children.iter().map(FolderChild::sort_key).collect();
        assert_eq!(deep_labels, vec!["c.ts"]);
    }

    #[test]
    fn gamma_nesting_preserves_single_folder_under_package_root() {
        // gamma has `entry.ts` at the package root and `utils/helper.ts`.
        // Common prefix is `packages/gamma/`, so gamma's children are
        // `entry.ts` and a folder `utils/` containing `helper.ts`.
        let (g, idx) = build_fixture();
        let tree = PackageTree::build(&g, &idx, &HashSet::new());

        let gamma = tree
            .packages
            .iter()
            .find(|p| p.name == "gamma")
            .expect("gamma present");
        let labels: Vec<&str> = gamma.children.iter().map(FolderChild::sort_key).collect();
        assert_eq!(labels, vec!["entry.ts", "utils"]);

        let utils_children = match &gamma.children[1] {
            FolderChild::Folder(f) => &f.children,
            _ => panic!("expected `utils` to be a folder"),
        };
        let utils_labels: Vec<&str> = utils_children
            .iter()
            .map(FolderChild::sort_key)
            .collect();
        assert_eq!(utils_labels, vec!["helper.ts"]);
    }

    #[test]
    fn single_file_package_has_no_folder_nesting() {
        // beta has one file; common prefix equals its parent dir so the
        // single child is the file itself, not a redundant folder chain.
        let (g, idx) = build_fixture();
        let tree = PackageTree::build(&g, &idx, &HashSet::new());

        let beta = tree
            .packages
            .iter()
            .find(|p| p.name == "beta")
            .expect("beta present");
        assert_eq!(beta.children.len(), 1);
        match &beta.children[0] {
            FolderChild::File(f) => assert_eq!(f.label, "index.ts"),
            _ => panic!("expected single file child, got a folder"),
        }
    }

    #[test]
    fn empty_graph_produces_empty_tree() {
        let g = Graph::new();
        let tree = PackageTree::build(&g, &HashMap::new(), &HashSet::new());
        assert!(tree.is_empty());
        assert!(tree.packages.is_empty());
        assert!(tree.externals.externals.is_empty());
        assert!(tree.unpackaged.files.is_empty());
    }

    #[test]
    fn workspace_aggregator_nodes_are_ignored() {
        // A graph with only synthetic aggregator nodes + externals must
        // produce zero packages in the tree. Packages are represented by
        // their real file subtrees, not by the aggregator stand-in.
        let mut g = Graph::new();
        g.add_node(workspace_agg("alpha"));
        g.add_node(workspace_agg("beta"));
        g.add_node(external("lodash"));
        let tree = PackageTree::build(&g, &HashMap::new(), &HashSet::new());
        assert!(tree.packages.is_empty());
        assert_eq!(tree.externals.externals.len(), 1);
    }

    #[test]
    fn missing_color_index_is_none_not_panic() {
        // A package without an entry in the color map yields `None` so
        // the sidebar can fall back to a neutral swatch.
        let mut g = Graph::new();
        g.add_node(file(
            "packages/solo/a.ts",
            "/repo/packages/solo/a.ts",
            Some("solo"),
        ));
        let tree = PackageTree::build(&g, &HashMap::new(), &HashSet::new());
        assert_eq!(tree.packages.len(), 1);
        assert_eq!(tree.packages[0].color_index, None);
    }

    #[test]
    fn default_check_state_is_checked_everywhere() {
        // With an empty hide set, every leaf (and every parent) renders as
        // Checked — the "nothing hidden" baseline callers see on a fresh
        // folder load before any user toggle.
        let (g, idx) = build_fixture();
        let tree = PackageTree::build(&g, &idx, &HashSet::new());

        assert_eq!(tree.unpackaged.check, CheckState::Checked);
        assert_eq!(tree.externals.check, CheckState::Checked);
        for ext in &tree.externals.externals {
            assert_eq!(ext.check, CheckState::Checked);
        }
        for f in &tree.unpackaged.files {
            assert_eq!(f.check, CheckState::Checked);
        }

        fn walk(children: &[FolderChild]) {
            for c in children {
                assert_eq!(c.check_state(), CheckState::Checked);
                if let FolderChild::Folder(f) = c {
                    walk(&f.children);
                }
            }
        }
        for pkg in &tree.packages {
            assert_eq!(pkg.check, CheckState::Checked);
            walk(&pkg.children);
        }
    }

    #[test]
    fn hidden_file_leaves_render_unchecked() {
        // A file id that's in the hide set must surface as Unchecked at the
        // leaf level; siblings outside the hide set keep their Checked
        // state. Parent rows stay Checked in this slice — tristate/cascade
        // for packages and folders lands in #23.
        let (g, idx) = build_fixture();
        let mut hidden: HashSet<NodeId> = HashSet::new();
        hidden.insert("packages/alpha/src/a.ts".to_string());
        hidden.insert("packages/gamma/utils/helper.ts".to_string());
        let tree = PackageTree::build(&g, &idx, &hidden);

        // Walk to alpha's a.ts leaf and assert it's Unchecked; b.ts (not
        // hidden) stays Checked.
        let alpha = &tree.packages[0];
        let a = match &alpha.children[0] {
            FolderChild::File(f) => f,
            _ => panic!("expected a.ts at alpha.children[0]"),
        };
        let b = match &alpha.children[1] {
            FolderChild::File(f) => f,
            _ => panic!("expected b.ts at alpha.children[1]"),
        };
        assert_eq!(a.check, CheckState::Unchecked);
        assert_eq!(b.check, CheckState::Checked);

        // Deep: gamma/utils/helper.ts is Unchecked inside its folder.
        let gamma = tree
            .packages
            .iter()
            .find(|p| p.name == "gamma")
            .expect("gamma present");
        let utils = match &gamma.children[1] {
            FolderChild::Folder(f) => f,
            _ => panic!("expected utils folder"),
        };
        let helper = match &utils.children[0] {
            FolderChild::File(f) => f,
            _ => panic!("expected helper.ts leaf"),
        };
        assert_eq!(helper.check, CheckState::Unchecked);

        // Parents stay Checked pending #23.
        assert_eq!(alpha.check, CheckState::Checked);
        assert_eq!(utils.check, CheckState::Checked);
    }

    #[test]
    fn common_dir_prefix_handles_single_path_and_empty() {
        // A single path's "prefix" is its parent directory — no other paths
        // to conflict with.
        let only = PathBuf::from("/a/b/c.ts");
        let p = common_dir_prefix([only.as_path()]);
        assert_eq!(p, PathBuf::from("/a/b"));

        // Empty input yields an empty prefix rather than panicking.
        let empty: Vec<&Path> = Vec::new();
        assert_eq!(common_dir_prefix(empty), PathBuf::new());
    }
}
