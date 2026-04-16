//! JSON export of the current dependency graph.
//!
//! Schema (stable, matches the PRD user story 43 contract):
//!
//! ```json
//! {
//!   "nodes": [{ "id": "...", "path": "...", "package": "...", "type": "workspace|external|declaration" }],
//!   "edges": [{ "from": "...", "to": "..." }],
//!   "cycles": [["node-id", "node-id"]]
//! }
//! ```
//!
//! Consumers (graph diffing, downstream analysis) read id/edges/cycles; the
//! `path` field is a convenience for grep/open workflows and is workspace-
//! relative when a root is known. `package` is omitted when the node has no
//! owning workspace package so parsers that expect a string don't choke on
//! `null`.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::graph::{Graph, NodeId, NodeKind};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExportedNodeType {
    Workspace,
    External,
    Declaration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedNode {
    pub id: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    #[serde(rename = "type")]
    pub node_type: ExportedNodeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedEdge {
    pub from: NodeId,
    pub to: NodeId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedGraph {
    pub nodes: Vec<ExportedNode>,
    pub edges: Vec<ExportedEdge>,
    pub cycles: Vec<Vec<NodeId>>,
}

/// Build the exportable view of the graph. Nodes and edges are emitted in a
/// deterministic (id-sorted / from-then-to) order so diffing two exports with
/// a plain line-based tool works. `cycles` is trusted to already be in the
/// order the sidebar shows — the caller owns presentation order.
pub fn build_export(graph: &Graph, cycles: &[Vec<NodeId>], root: Option<&Path>) -> ExportedGraph {
    let mut nodes: Vec<ExportedNode> = graph
        .nodes
        .values()
        .map(|n| ExportedNode {
            id: n.id.clone(),
            path: display_path(&n.path, &n.kind, root),
            package: n.package.clone(),
            node_type: classify(n.kind, &n.path),
        })
        .collect();
    nodes.sort_by(|a, b| a.id.cmp(&b.id));

    let mut edges: Vec<ExportedEdge> = graph
        .edges
        .iter()
        .map(|e| ExportedEdge {
            from: e.from.clone(),
            to: e.to.clone(),
        })
        .collect();
    edges.sort_by(|a, b| a.from.cmp(&b.from).then(a.to.cmp(&b.to)));

    ExportedGraph {
        nodes,
        edges,
        cycles: cycles.to_vec(),
    }
}

/// Write `graph` as pretty-printed JSON to `path`. Pretty-printed so humans
/// can read the file in a text editor — the size cost is negligible compared
/// to gzipped storage and graphs of practical size.
pub fn write_json(path: &Path, exported: &ExportedGraph) -> io::Result<()> {
    let json = serde_json::to_string_pretty(exported).map_err(io::Error::other)?;
    fs::write(path, json)
}

fn classify(kind: NodeKind, path: &Path) -> ExportedNodeType {
    match kind {
        NodeKind::External => ExportedNodeType::External,
        NodeKind::WorkspacePackage => ExportedNodeType::Workspace,
        NodeKind::File => {
            if is_declaration_path(path) {
                ExportedNodeType::Declaration
            } else {
                ExportedNodeType::Workspace
            }
        }
    }
}

/// Synthetic nodes (external / workspace-pkg) store their package name in
/// `path`; emit that verbatim. File nodes store a canonical absolute path;
/// strip the workspace root so the export stays portable across machines.
fn display_path(path: &Path, kind: &NodeKind, root: Option<&Path>) -> String {
    if matches!(kind, NodeKind::External | NodeKind::WorkspacePackage) {
        return path.to_string_lossy().into_owned();
    }
    if let Some(root) = root
        && let Ok(rel) = path.strip_prefix(root)
    {
        return rel.to_string_lossy().replace('\\', "/");
    }
    path.to_string_lossy().into_owned()
}

fn is_declaration_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|name| name.ends_with(".d.ts") || name.ends_with(".d.cts") || name.ends_with(".d.mts"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn file_node(id: &str, rel: &str, package: Option<&str>, root: &Path) -> Node {
        Node {
            id: id.to_string(),
            path: root.join(rel),
            label: id.to_string(),
            package: package.map(|s| s.to_string()),
            kind: NodeKind::File,
        }
    }

    #[test]
    fn classifies_workspace_external_and_declaration() {
        let root = PathBuf::from("/repo");
        let mut g = Graph::new();
        g.add_node(file_node("src/a.ts", "src/a.ts", Some("@org/app"), &root));
        g.add_node(file_node(
            "src/types.d.ts",
            "src/types.d.ts",
            Some("@org/app"),
            &root,
        ));
        g.add_node(Node {
            id: "external:react".into(),
            path: PathBuf::from("react"),
            label: "react".into(),
            package: None,
            kind: NodeKind::External,
        });
        g.add_node(Node {
            id: "package:@org/app".into(),
            path: PathBuf::from("@org/app"),
            label: "@org/app".into(),
            package: Some("@org/app".into()),
            kind: NodeKind::WorkspacePackage,
        });

        let exported = build_export(&g, &[], Some(&root));
        let type_by_id: std::collections::HashMap<_, _> = exported
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), n.node_type.clone()))
            .collect();
        assert_eq!(type_by_id["src/a.ts"], ExportedNodeType::Workspace);
        assert_eq!(
            type_by_id["src/types.d.ts"],
            ExportedNodeType::Declaration
        );
        assert_eq!(type_by_id["external:react"], ExportedNodeType::External);
        assert_eq!(
            type_by_id["package:@org/app"],
            ExportedNodeType::Workspace
        );
    }

    #[test]
    fn emits_workspace_relative_paths_for_files() {
        let root = PathBuf::from("/repo");
        let mut g = Graph::new();
        g.add_node(file_node(
            "src/a.ts",
            "src/a.ts",
            Some("@org/app"),
            &root,
        ));

        let exported = build_export(&g, &[], Some(&root));
        assert_eq!(exported.nodes[0].path, "src/a.ts");
    }

    #[test]
    fn omits_package_field_when_none() {
        let root = PathBuf::from("/repo");
        let mut g = Graph::new();
        g.add_node(Node {
            id: "external:react".into(),
            path: PathBuf::from("react"),
            label: "react".into(),
            package: None,
            kind: NodeKind::External,
        });

        let exported = build_export(&g, &[], Some(&root));
        let json = serde_json::to_string(&exported).unwrap();
        assert!(
            !json.contains("\"package\""),
            "package field should be omitted when None, got: {json}"
        );
    }

    #[test]
    fn round_trip_preserves_node_edge_cycle_counts() {
        // Acceptance criterion: parse the exported JSON and assert that node /
        // edge / cycle counts match the live graph.
        let root = PathBuf::from("/repo");
        let mut g = Graph::new();
        g.add_node(file_node("a.ts", "a.ts", Some("@org/app"), &root));
        g.add_node(file_node("b.ts", "b.ts", Some("@org/app"), &root));
        g.add_node(file_node("c.ts", "c.ts", Some("@org/app"), &root));
        g.add_edge("a.ts", "b.ts");
        g.add_edge("b.ts", "c.ts");
        g.add_edge("c.ts", "a.ts");

        let cycles = g.cycles();
        let exported = build_export(&g, &cycles, Some(&root));
        let json = serde_json::to_string(&exported).unwrap();
        let round: ExportedGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(round.nodes.len(), g.nodes.len());
        assert_eq!(round.edges.len(), g.edges.len());
        assert_eq!(round.cycles.len(), cycles.len());
        assert_eq!(round.cycles[0].len(), cycles[0].len());
    }

    #[test]
    fn write_json_creates_readable_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("graph.json");

        let mut g = Graph::new();
        g.add_node(file_node("a.ts", "a.ts", Some("@org/app"), dir.path()));

        let exported = build_export(&g, &[], Some(dir.path()));
        write_json(&path, &exported).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let parsed: ExportedGraph = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].id, "a.ts");
    }

    #[test]
    fn nodes_and_edges_are_sorted_for_stable_diffs() {
        let root = PathBuf::from("/repo");
        let mut g = Graph::new();
        g.add_node(file_node("b.ts", "b.ts", None, &root));
        g.add_node(file_node("a.ts", "a.ts", None, &root));
        g.add_node(file_node("c.ts", "c.ts", None, &root));
        g.edges.push(Edge {
            from: "c.ts".into(),
            to: "a.ts".into(),
        });
        g.edges.push(Edge {
            from: "a.ts".into(),
            to: "b.ts".into(),
        });

        let exported = build_export(&g, &[], Some(&root));
        let ids: Vec<_> = exported.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, vec!["a.ts", "b.ts", "c.ts"]);
        assert_eq!(exported.edges[0].from, "a.ts");
        assert_eq!(exported.edges[1].from, "c.ts");
    }
}
