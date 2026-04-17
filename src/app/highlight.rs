use std::collections::{HashMap, HashSet};

use crate::graph::{Edge, NodeId};

use super::GruffApp;

/// Snapshot of the dependency chain a user clicked into — the clicked edge
/// plus every edge that lies on some path passing through it. Lets the
/// renderer highlight/dim in O(1) per edge instead of recomputing chains
/// every frame.
#[derive(Debug, Clone, Default)]
pub(super) struct PathHighlight {
    /// `(from, to)` pairs for every edge on the chain, including the clicked
    /// edge itself. Used by the edge render loop to decide highlight vs dim.
    pub(super) edges: HashSet<(NodeId, NodeId)>,
    /// Every node touched by an edge in `edges`. Used to decide which nodes
    /// stay full-opacity and which fade out.
    pub(super) nodes: HashSet<NodeId>,
}

impl GruffApp {
    /// Build a [`PathHighlight`] for a clicked node: the union of the node's
    /// recursive upstream closure (every ancestor that can reach it) and its
    /// recursive downstream closure (every descendant it can reach), plus
    /// every edge whose endpoints both lie in that union. Sibling branches
    /// — nodes that share a parent with the clicked node but aren't on its
    /// chain — are excluded.
    #[allow(dead_code)]
    pub(super) fn build_node_highlight(&self, node: &NodeId) -> PathHighlight {
        compute_node_highlight(node, &self.graph.edges, &self.imports, &self.imported_by)
    }

    /// Build a [`PathHighlight`] for a clicked node, one hop in both
    /// directions: the clicked node + every direct predecessor + every
    /// direct successor + the edges connecting the clicked node to those
    /// neighbours. No transitive walk — grandparents and grandchildren stay
    /// dim. The cycle-tinting system handles any cycle visualisation
    /// independently.
    pub(super) fn build_one_hop_highlight(&self, node: &NodeId) -> PathHighlight {
        compute_one_hop_highlight(node, &self.imports, &self.imported_by)
    }
}

/// Pure node-highlight builder: collect the clicked node's recursive
/// upstream closure (every ancestor that can reach it via reverse edges)
/// unioned with its recursive downstream closure (every descendant it can
/// reach via forward edges), plus every edge whose endpoints both live in
/// that union. Siblings of the clicked node that share a parent but aren't
/// on its chain are intentionally excluded — an edge from a shared parent
/// out to a sibling has one endpoint (the sibling) outside the union, so
/// it isn't picked up.
fn compute_node_highlight(
    node: &NodeId,
    edges: &[Edge],
    imports: &HashMap<NodeId, Vec<NodeId>>,
    imported_by: &HashMap<NodeId, Vec<NodeId>>,
) -> PathHighlight {
    // Walk reverse edges for ancestors, forward edges for descendants. The
    // clicked node appears in both sets (bfs_visit includes `start`), which
    // is exactly what we want — it's the hub of the highlight.
    let upstream = bfs_visit(node, imported_by);
    let downstream = bfs_visit(node, imports);

    let mut nodes: HashSet<NodeId> = HashSet::new();
    nodes.extend(upstream);
    nodes.extend(downstream);

    // An edge belongs to the highlight iff both endpoints live in the
    // union. A shared parent `p` points at both the clicked node and a
    // sibling `s`; `p` is in the union (ancestor) but `s` isn't, so the
    // `p -> s` edge is correctly excluded — the "sibling branches stay
    // dim" guarantee from PRD #16.
    let mut hl_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
    for e in edges {
        if nodes.contains(&e.from) && nodes.contains(&e.to) {
            hl_edges.insert((e.from.clone(), e.to.clone()));
        }
    }

    PathHighlight {
        edges: hl_edges,
        nodes,
    }
}

/// Pure one-hop highlight builder: the clicked node + every direct
/// predecessor (`to == node`) + every direct successor (`from == node`),
/// plus the edges that connect the clicked node to those neighbours. No
/// transitive traversal — grandparents and grandchildren are intentionally
/// excluded. A 2-cycle with a direct neighbour highlights both edges
/// naturally because each is itself a direct neighbour edge of the clicked
/// node. Larger cycles keep their red tint from the separate cycle system.
fn compute_one_hop_highlight(
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

/// Iterative BFS over `adj` starting from `start`, returning every visited
/// node (including `start`). The visited set is what makes this safe on
/// cyclic graphs: a node only ever gets pushed once.
fn bfs_visit(start: &NodeId, adj: &HashMap<NodeId, Vec<NodeId>>) -> HashSet<NodeId> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = Vec::new();
    seen.insert(start.clone());
    stack.push(start.clone());
    while let Some(n) = stack.pop() {
        if let Some(neighbors) = adj.get(&n) {
            for nb in neighbors {
                if seen.insert(nb.clone()) {
                    stack.push(nb.clone());
                }
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn node_highlight_multi_parent_multi_child_full_closure() {
        // Two parents p1, p2 both point at n. n points at two children c1, c2.
        // Grandparent gp sits above p1; grandchild gc sits below c1. Clicking
        // n must pull in the full upstream and downstream closure — gp, p1,
        // p2 on the ancestor side; c1, c2, gc on the descendant side — plus
        // every edge whose endpoints both live in that set.
        let edges = vec![
            edge("gp", "p1"),
            edge("p1", "n"),
            edge("p2", "n"),
            edge("n", "c1"),
            edge("n", "c2"),
            edge("c1", "gc"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("n"), &edges, &fwd, &rev);
        for expected in ["gp", "p1", "p2", "n", "c1", "c2", "gc"] {
            assert!(
                hl.nodes.contains(&id(expected)),
                "expected node {expected} in node-click highlight"
            );
        }
        // Every edge in the graph has both endpoints inside the closure
        // here — no sibling branches exist — so all six edges highlight.
        assert_eq!(hl.edges.len(), 6);
        for (f, t) in [
            ("gp", "p1"),
            ("p1", "n"),
            ("p2", "n"),
            ("n", "c1"),
            ("n", "c2"),
            ("c1", "gc"),
        ] {
            assert!(
                hl.edges.contains(&(id(f), id(t))),
                "expected edge {f}->{t} in node-click highlight"
            );
        }
    }

    #[test]
    fn node_highlight_excludes_sibling_branches() {
        // Shared parent `p` points at both `n` (clicked) and `sibling`. `n`
        // has its own child `c` and `sibling` has its own child `sc`.
        // Clicking `n` must include p, n, c — but NOT sibling or sc. The
        // `p -> sibling` edge must also be dim because `sibling` isn't in
        // the closure.
        let edges = vec![
            edge("p", "n"),
            edge("p", "sibling"),
            edge("n", "c"),
            edge("sibling", "sc"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("n"), &edges, &fwd, &rev);
        assert!(hl.nodes.contains(&id("p")));
        assert!(hl.nodes.contains(&id("n")));
        assert!(hl.nodes.contains(&id("c")));
        assert!(
            !hl.nodes.contains(&id("sibling")),
            "sibling must stay dim on node-click highlight"
        );
        assert!(
            !hl.nodes.contains(&id("sc")),
            "sibling's descendant must stay dim on node-click highlight"
        );
        // The shared-parent → sibling edge must be excluded even though one
        // endpoint (the parent) is in the closure — the PRD's "siblings
        // stay dim" guarantee only holds if the connecting edge also dims.
        assert!(
            !hl.edges.contains(&(id("p"), id("sibling"))),
            "sibling-branch edge must be excluded"
        );
        assert!(!hl.edges.contains(&(id("sibling"), id("sc"))));
        // And the on-chain edges are all present.
        for (f, t) in [("p", "n"), ("n", "c")] {
            assert!(hl.edges.contains(&(id(f), id(t))));
        }
    }

    #[test]
    fn node_highlight_on_leaf_collapses_to_ancestors() {
        // Leaf node with no outgoing edges: the downstream closure is just
        // the node itself, so the highlight is purely the upstream chain
        // plus the node.
        let edges = vec![edge("a", "b"), edge("b", "leaf")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("leaf"), &edges, &fwd, &rev);
        assert!(hl.nodes.contains(&id("a")));
        assert!(hl.nodes.contains(&id("b")));
        assert!(hl.nodes.contains(&id("leaf")));
        assert_eq!(hl.nodes.len(), 3);
        assert_eq!(hl.edges.len(), 2);
    }

    #[test]
    fn node_highlight_cycle_terminates() {
        // Clicking a node inside a cycle must terminate and include every
        // cycle member. The `start` is in both bfs visits, so the whole
        // SCC lands in the closure.
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("c", "a")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_node_highlight(&id("a"), &edges, &fwd, &rev);
        assert_eq!(hl.nodes.len(), 3);
        assert_eq!(hl.edges.len(), 3);
    }

    // --- One-hop node highlight (#29) -------------------------------------

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
        let hl = compute_one_hop_highlight(&id("n"), &fwd, &rev);
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
        let hl = compute_one_hop_highlight(&id("root"), &fwd, &rev);
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
        let hl = compute_one_hop_highlight(&id("leaf"), &fwd, &rev);
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
        // just the node itself with no edges — same as today's behaviour.
        let edges: Vec<Edge> = vec![edge("a", "b")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_one_hop_highlight(&id("orphan"), &fwd, &rev);
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
        let hl = compute_one_hop_highlight(&id("a"), &fwd, &rev);
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
        let hl = compute_one_hop_highlight(&id("b"), &fwd, &rev);
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
