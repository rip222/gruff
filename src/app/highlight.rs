use std::collections::{HashMap, HashSet};

use crate::graph::NodeId;

use super::GruffApp;

/// Snapshot of the highlight a user clicked into — the clicked node plus
/// its direct predecessors and successors and the edges that connect them.
/// Lets the renderer highlight/dim in O(1) per element instead of
/// recomputing the set every frame.
#[derive(Debug, Clone, Default)]
pub(super) struct PathHighlight {
    /// `(from, to)` pairs for every direct-neighbour edge. Used by the
    /// edge render loop to decide highlight vs dim.
    pub(super) edges: HashSet<(NodeId, NodeId)>,
    /// The clicked node plus every direct neighbour. Used to decide which
    /// nodes stay full-opacity and which fade out.
    pub(super) nodes: HashSet<NodeId>,
}

impl GruffApp {
    /// Build a [`PathHighlight`] for a clicked node, one hop in both
    /// directions: the clicked node + every direct predecessor + every
    /// direct successor + the edges connecting the clicked node to those
    /// neighbours. No transitive walk — grandparents and grandchildren stay
    /// dim. Cycles keep their red tint from the separate cycle-tinting
    /// system; a 2-cycle with the clicked node naturally highlights both
    /// edges because each is itself a direct-neighbour edge.
    pub(super) fn build_node_highlight(&self, node: &NodeId) -> PathHighlight {
        compute_node_highlight(node, &self.imports, &self.imported_by)
    }
}

/// Pure one-hop highlight builder: the clicked node + every direct
/// predecessor (`to == node`) + every direct successor (`from == node`),
/// plus the edges that connect the clicked node to those neighbours.
/// Factored out of [`GruffApp`] so it can be unit-tested without standing
/// up an egui context.
fn compute_node_highlight(
    node: &NodeId,
    imports: &HashMap<NodeId, Vec<NodeId>>,
    imported_by: &HashMap<NodeId, Vec<NodeId>>,
) -> PathHighlight {
    let mut nodes: HashSet<NodeId> = HashSet::new();
    nodes.insert(node.clone());
    let mut hl_edges: HashSet<(NodeId, NodeId)> = HashSet::new();

    if let Some(succs) = imports.get(node) {
        for s in succs {
            nodes.insert(s.clone());
            hl_edges.insert((node.clone(), s.clone()));
        }
    }
    if let Some(preds) = imported_by.get(node) {
        for p in preds {
            nodes.insert(p.clone());
            hl_edges.insert((p.clone(), node.clone()));
        }
    }

    PathHighlight {
        edges: hl_edges,
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graph::Edge;

    fn id(s: &str) -> NodeId {
        s.to_string()
    }

    /// Build forward/reverse adjacency lists from a flat edge list, matching
    /// the shape `GruffApp` caches at load time.
    fn adj(edges: &[Edge]) -> (HashMap<NodeId, Vec<NodeId>>, HashMap<NodeId, Vec<NodeId>>) {
        let mut fwd: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut rev: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for e in edges {
            fwd.entry(e.from.clone()).or_default().push(e.to.clone());
            rev.entry(e.to.clone()).or_default().push(e.from.clone());
        }
        (fwd, rev)
    }

    fn edge(from: &str, to: &str) -> Edge {
        Edge {
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    #[test]
    fn one_hop_multi_parent_multi_child_only_direct_neighbours() {
        // gp -> p1 -> n -> c1 -> gc
        //       p2 ----^   c2 ----^
        // Clicking n must include p1, p2, c1, c2 + n itself, but exclude
        // the grandparent gp and grandchild gc — they're two hops away.
        let edges = vec![
            edge("gp", "p1"),
            edge("p1", "n"),
            edge("p2", "n"),
            edge("n", "c1"),
            edge("n", "c2"),
            edge("c1", "gc"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("n"), &fwd, &rev);
        for expected in ["n", "p1", "p2", "c1", "c2"] {
            assert!(
                hl.nodes.contains(&id(expected)),
                "expected node {expected} in one-hop highlight"
            );
        }
        assert!(
            !hl.nodes.contains(&id("gp")),
            "grandparent must stay dim — it's two hops up"
        );
        assert!(
            !hl.nodes.contains(&id("gc")),
            "grandchild must stay dim — it's two hops down"
        );
        // Exactly the four direct edges: p1->n, p2->n, n->c1, n->c2.
        assert_eq!(hl.edges.len(), 4);
        for (f, t) in [("p1", "n"), ("p2", "n"), ("n", "c1"), ("n", "c2")] {
            assert!(
                hl.edges.contains(&(id(f), id(t))),
                "expected edge {f}->{t} in one-hop highlight"
            );
        }
        // Two-hop edges intentionally excluded.
        assert!(!hl.edges.contains(&(id("gp"), id("p1"))));
        assert!(!hl.edges.contains(&(id("c1"), id("gc"))));
    }

    #[test]
    fn one_hop_only_outgoing_edges() {
        // A leaf-of-the-graph by upstream: nothing imports `root`. Clicking
        // it must include only `root` and its direct successors, no
        // "upstream" half — that half is empty.
        let edges = vec![edge("root", "a"), edge("root", "b"), edge("a", "c")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("root"), &fwd, &rev);
        for expected in ["root", "a", "b"] {
            assert!(hl.nodes.contains(&id(expected)));
        }
        assert!(
            !hl.nodes.contains(&id("c")),
            "two-hop descendant must stay dim"
        );
        assert_eq!(hl.edges.len(), 2);
        assert!(hl.edges.contains(&(id("root"), id("a"))));
        assert!(hl.edges.contains(&(id("root"), id("b"))));
    }

    #[test]
    fn one_hop_only_incoming_edges() {
        // A pure leaf: nothing imports anything from `leaf`. Clicking it
        // must include `leaf` and its direct predecessors, no downstream
        // half.
        let edges = vec![edge("a", "leaf"), edge("b", "leaf"), edge("c", "a")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("leaf"), &fwd, &rev);
        for expected in ["leaf", "a", "b"] {
            assert!(hl.nodes.contains(&id(expected)));
        }
        assert!(
            !hl.nodes.contains(&id("c")),
            "two-hop ancestor must stay dim"
        );
        assert_eq!(hl.edges.len(), 2);
        assert!(hl.edges.contains(&(id("a"), id("leaf"))));
        assert!(hl.edges.contains(&(id("b"), id("leaf"))));
    }

    #[test]
    fn one_hop_isolated_node_has_no_edges() {
        // Isolated node sits in the graph alone. Clicking it must include
        // just the node itself with no edges.
        let edges: Vec<Edge> = vec![edge("a", "b")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("orphan"), &fwd, &rev);
        assert_eq!(hl.nodes.len(), 1);
        assert!(hl.nodes.contains(&id("orphan")));
        assert!(hl.edges.is_empty());
    }

    #[test]
    fn one_hop_two_cycle_highlights_both_directions() {
        // 2-cycle A <-> B: clicking A must light up both A->B and B->A
        // because each is a direct-neighbour edge of A. Verifies the PRD's
        // "cycles fall out naturally" claim for the simplest cycle case.
        let edges = vec![edge("a", "b"), edge("b", "a")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("a"), &fwd, &rev);
        assert_eq!(hl.nodes.len(), 2);
        assert!(hl.nodes.contains(&id("a")));
        assert!(hl.nodes.contains(&id("b")));
        assert_eq!(hl.edges.len(), 2);
        assert!(hl.edges.contains(&(id("a"), id("b"))));
        assert!(hl.edges.contains(&(id("b"), id("a"))));
    }

    #[test]
    fn one_hop_disjoint_component_stays_dim() {
        // Two unconnected components. Clicking a node in one must leave
        // every node and edge in the other untouched.
        let edges = vec![
            edge("a", "b"),
            edge("b", "c"),
            edge("x", "y"),
            edge("y", "z"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("b"), &fwd, &rev);
        for expected in ["a", "b", "c"] {
            assert!(hl.nodes.contains(&id(expected)));
        }
        for forbidden in ["x", "y", "z"] {
            assert!(
                !hl.nodes.contains(&id(forbidden)),
                "disjoint-component node {forbidden} must stay dim"
            );
        }
        for forbidden in [("x", "y"), ("y", "z")] {
            assert!(
                !hl.edges.contains(&(id(forbidden.0), id(forbidden.1))),
                "disjoint-component edge {forbidden:?} must stay dim"
            );
        }
    }
}
