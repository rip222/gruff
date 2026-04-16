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
    /// Build a [`PathHighlight`] for the clicked edge `u -> v`: walk backward
    /// from `u` collecting ancestors and forward from `v` collecting
    /// descendants, then gather every edge whose endpoints both lie in one
    /// of those reachable sets. The result represents every simple path
    /// through the graph that passes through the clicked edge. BFS uses a
    /// visited set, so cyclic regions don't cause infinite loops.
    pub(super) fn build_path_highlight(&self, from: &NodeId, to: &NodeId) -> PathHighlight {
        compute_path_highlight(
            from,
            to,
            &self.graph.edges,
            &self.imports,
            &self.imported_by,
        )
    }
}

/// Pure path-highlight builder: collect every edge that lies on a path
/// passing through the clicked edge `(from -> to)`. Factored out of
/// [`GruffApp`] so it can be unit-tested without standing up an egui
/// context.
fn compute_path_highlight(
    from: &NodeId,
    to: &NodeId,
    edges: &[Edge],
    imports: &HashMap<NodeId, Vec<NodeId>>,
    imported_by: &HashMap<NodeId, Vec<NodeId>>,
) -> PathHighlight {
    // Backward reach from `from` along reverse edges — every ancestor whose
    // imports can eventually land on `from`. Forward reach from `to` along
    // forward edges — every descendant `to` can reach. Together these bound
    // the set of nodes that lie on any path through the clicked edge.
    let upstream = bfs_visit(from, imported_by);
    let downstream = bfs_visit(to, imports);

    let mut hl_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
    hl_edges.insert((from.clone(), to.clone()));
    for e in edges {
        // An edge is on some path through (from -> to) iff both endpoints
        // live in the same reach set. Crossing sets (e.g. an ancestor
        // pointing into a descendant without going through the clicked
        // edge) are intentionally excluded — they aren't a chain through
        // the clicked edge.
        let in_up = upstream.contains(&e.from) && upstream.contains(&e.to);
        let in_down = downstream.contains(&e.from) && downstream.contains(&e.to);
        if in_up || in_down {
            hl_edges.insert((e.from.clone(), e.to.clone()));
        }
    }

    let mut nodes: HashSet<NodeId> = HashSet::new();
    nodes.extend(upstream);
    nodes.extend(downstream);
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
    fn linear_chain_click_middle_highlights_everything() {
        // a -> b -> c -> d : clicking (b -> c) should highlight every edge
        // and every node — the chain fills the graph.
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("c", "d")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("b"), &id("c"), &edges, &fwd, &rev);
        assert!(hl.edges.contains(&(id("a"), id("b"))));
        assert!(hl.edges.contains(&(id("b"), id("c"))));
        assert!(hl.edges.contains(&(id("c"), id("d"))));
        assert_eq!(hl.edges.len(), 3);
        for n in ["a", "b", "c", "d"] {
            assert!(hl.nodes.contains(&id(n)), "expected node {n} in highlight");
        }
    }

    #[test]
    fn branching_graph_excludes_sibling_branches() {
        //        /-> x
        //   a -> b -> c -> d
        //        \-> y
        // Clicking (b -> c) should include a and d (on the chain) but NOT
        // x or y — they branch away from the clicked edge.
        let edges = vec![
            edge("a", "b"),
            edge("b", "c"),
            edge("c", "d"),
            edge("b", "x"),
            edge("b", "y"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("b"), &id("c"), &edges, &fwd, &rev);
        assert!(hl.nodes.contains(&id("a")));
        assert!(hl.nodes.contains(&id("d")));
        assert!(!hl.nodes.contains(&id("x")));
        assert!(!hl.nodes.contains(&id("y")));
        // And the sibling-branch edges are excluded from the highlight.
        assert!(!hl.edges.contains(&(id("b"), id("x"))));
        assert!(!hl.edges.contains(&(id("b"), id("y"))));
    }

    #[test]
    fn cycle_does_not_cause_infinite_loop() {
        // a -> b -> c -> a forms a cycle. Clicking any edge inside the cycle
        // should terminate and highlight every member edge.
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("c", "a")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("a"), &id("b"), &edges, &fwd, &rev);
        assert_eq!(hl.edges.len(), 3);
        assert_eq!(hl.nodes.len(), 3);
        for from_to in [("a", "b"), ("b", "c"), ("c", "a")] {
            assert!(
                hl.edges.contains(&(id(from_to.0), id(from_to.1))),
                "missing cycle edge {from_to:?}"
            );
        }
    }

    #[test]
    fn cycle_with_external_tail_and_head() {
        // root -> a -> b -> a (cycle a<->b) -> c -> leaf
        // Clicking (a -> b) inside the cycle should reach root (upstream of
        // the cycle) and leaf (downstream of the cycle) — the full chain —
        // without looping.
        let edges = vec![
            edge("root", "a"),
            edge("a", "b"),
            edge("b", "a"),
            edge("b", "c"),
            edge("c", "leaf"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("a"), &id("b"), &edges, &fwd, &rev);
        for n in ["root", "a", "b", "c", "leaf"] {
            assert!(hl.nodes.contains(&id(n)), "expected {n} in path highlight");
        }
    }

    #[test]
    fn unrelated_component_is_not_highlighted() {
        // Two disjoint chains: clicking one must leave the other dim.
        let edges = vec![
            edge("a", "b"),
            edge("b", "c"),
            edge("x", "y"),
            edge("y", "z"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("a"), &id("b"), &edges, &fwd, &rev);
        assert!(hl.nodes.contains(&id("a")));
        assert!(hl.nodes.contains(&id("c")));
        assert!(!hl.nodes.contains(&id("x")));
        assert!(!hl.nodes.contains(&id("y")));
        assert!(!hl.nodes.contains(&id("z")));
    }
}
