use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::graph::{Graph, Node, NodeKind};
use crate::parser::parse_file_imports;
use crate::resolver::{resolve, ResolvedImport};
use crate::workspace::Workspace;

const SOURCE_EXTS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];

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

/// Scan `root` into a `Workspace`, then index every JS/TS file under it.
/// This is the entry point the app uses when the user drops a folder.
pub fn index_folder(root: &Path) -> Graph {
    let ws = Workspace::discover(root);
    index_workspace(&ws)
}

/// Walk the workspace, parse every JS/TS file, resolve relative imports, and
/// build a file-level dependency graph with each node tagged by its owning
/// package. Respects the workspace's `.gitignore`-aware matcher.
pub fn index_workspace(ws: &Workspace) -> Graph {
    let mut graph = Graph::new();

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(&ws.root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| e.depth() == 0 || !ws.is_ignored(e.path()))
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !SOURCE_EXTS.contains(&ext) {
            continue;
        }
        files.push(path.to_path_buf());
    }

    let mut id_by_path: HashMap<PathBuf, String> = HashMap::new();
    let mut package_of_file: HashMap<PathBuf, Option<String>> = HashMap::new();
    for f in &files {
        let canonical = f.canonicalize().unwrap_or_else(|_| f.clone());
        let rel = canonical
            .strip_prefix(&ws.root)
            .unwrap_or(&canonical)
            .to_path_buf();
        let id = rel.to_string_lossy().replace('\\', "/");
        let label = canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        let package = ws.owning_package(&canonical).map(|p| p.name.clone());
        graph.add_node(Node {
            id: id.clone(),
            path: canonical.clone(),
            label,
            package: package.clone(),
            kind: NodeKind::File,
        });
        id_by_path.insert(canonical.clone(), id);
        package_of_file.insert(canonical, package);
    }

    // Track which external packages we've already added as nodes so we create
    // exactly one `External` node per distinct package name across the whole
    // graph, no matter how many workspace files import it.
    let mut external_added: HashSet<String> = HashSet::new();
    // Workspace-package aggregator nodes are created lazily — only for
    // packages that actually import externals, so repos with no external
    // dependencies don't get extra clutter in the graph.
    let mut workspace_pkg_added: HashSet<String> = HashSet::new();
    // `add_edge` is already idempotent, but we still track (pkg, external)
    // pairs explicitly so aggregation is obvious from reading the code.
    let mut aggregated_edges: HashSet<(String, String)> = HashSet::new();

    for f in &files {
        let from_canonical = f.canonicalize().unwrap_or_else(|_| f.clone());
        let Some(from_id) = id_by_path.get(&from_canonical).cloned() else {
            continue;
        };
        let owning_pkg = package_of_file
            .get(&from_canonical)
            .cloned()
            .flatten();

        for imp in parse_file_imports(&from_canonical) {
            match resolve(&from_canonical, &imp.source) {
                ResolvedImport::WorkspaceFile(target) => {
                    let resolved_canonical = target.canonicalize().unwrap_or(target);
                    if let Some(to_id) = id_by_path.get(&resolved_canonical) {
                        graph.add_edge(&from_id, to_id);
                    }
                }
                ResolvedImport::External(pkg_name) => {
                    // Source endpoint: the workspace-package aggregator. Fall
                    // back to a stable synthetic name when the file isn't
                    // attributed to any package so orphaned imports still get
                    // aggregated (and don't fan out to the external node).
                    let source_pkg = owning_pkg
                        .clone()
                        .unwrap_or_else(|| "(unattributed)".to_string());

                    let ext_id = external_node_id(&pkg_name);
                    if external_added.insert(pkg_name.clone()) {
                        graph.add_node(Node {
                            id: ext_id.clone(),
                            // Externals don't have an on-disk path we care
                            // about — use the bare package name as a marker so
                            // the editor-open affordance doesn't try to cd
                            // into node_modules.
                            path: PathBuf::from(&pkg_name),
                            label: pkg_name.clone(),
                            // Externals don't belong to a workspace package;
                            // leaving this `None` also keeps them out of the
                            // layout's clustering force.
                            package: None,
                            kind: NodeKind::External,
                        });
                    }

                    let ws_pkg_id = workspace_package_node_id(&source_pkg);
                    if workspace_pkg_added.insert(source_pkg.clone()) {
                        graph.add_node(Node {
                            id: ws_pkg_id.clone(),
                            path: PathBuf::from(&source_pkg),
                            label: source_pkg.clone(),
                            // Tag the aggregator with its own package name so
                            // the layout's clustering force pulls it into the
                            // same cluster as its files.
                            package: Some(source_pkg.clone()),
                            kind: NodeKind::WorkspacePackage,
                        });
                    }

                    if aggregated_edges.insert((source_pkg.clone(), pkg_name.clone())) {
                        graph.add_edge(&ws_pkg_id, &ext_id);
                    }
                }
                ResolvedImport::Unresolved => {}
            }
        }
    }

    graph
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

        let g = index_folder(dir.path());
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn skips_node_modules() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();

        let g = index_folder(dir.path());
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn external_bare_specifier_adds_single_external_node() {
        // Two files in the same workspace package both import `react`. There
        // should be exactly one `external:react` node regardless of importer
        // count, and exactly one edge from the package aggregator to it.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"@org/app"}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("a.ts"),
            r#"import React from "react";"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("b.ts"),
            r#"import React from "react";"#,
        )
        .unwrap();

        let g = index_folder(dir.path());

        let ext_id = external_node_id("react");
        let react_nodes: Vec<_> = g
            .nodes
            .values()
            .filter(|n| n.id == ext_id)
            .collect();
        assert_eq!(react_nodes.len(), 1);
        assert_eq!(react_nodes[0].kind, NodeKind::External);

        let edges_to_react: Vec<_> = g
            .edges
            .iter()
            .filter(|e| e.to == ext_id)
            .collect();
        // One aggregated edge, even though two files import react.
        assert_eq!(edges_to_react.len(), 1);
        assert_eq!(edges_to_react[0].from, workspace_package_node_id("@org/app"));
    }

    #[test]
    fn scoped_and_subpath_externals_collapse_to_package_name() {
        // `@scope/pkg/deep` and `lodash/fp` must both resolve to the package
        // portion only, producing a single leaf per package regardless of the
        // subpath used to import it.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"@org/app"}"#,
        )
        .unwrap();
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

        let g = index_folder(dir.path());

        let externals: Vec<_> = g
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::External)
            .map(|n| n.label.clone())
            .collect();
        assert!(externals.contains(&"@scope/pkg".to_string()));
        assert!(externals.contains(&"lodash".to_string()));
        // Exactly two distinct external packages — subpath variants must not
        // produce separate nodes.
        assert_eq!(externals.len(), 2);
    }

    #[test]
    fn external_edge_aggregated_per_workspace_package() {
        // Two workspace packages, each with two files importing `react`.
        // Expectation: exactly two aggregated edges to the `react` node —
        // one per importing workspace package.
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

        let g = index_folder(dir.path());

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
        // Aggregator nodes are created lazily. A repo that only uses relative
        // imports must not sprout synthetic package nodes.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("package.json"), r#"{"name":"solo"}"#).unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "export const b = 1;").unwrap();

        let g = index_folder(dir.path());
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
        // Two workspace packages — each file must carry its owning package name.
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

        let g = index_folder(dir.path());
        let packages: Vec<_> = g
            .nodes
            .values()
            .filter_map(|n| n.package.clone())
            .collect();
        assert!(packages.contains(&"@org/a".to_string()));
        assert!(packages.contains(&"@org/b".to_string()));
    }
}
