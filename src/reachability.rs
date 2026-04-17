//! Transitive-dependent reachability over an already-built [`Graph`].
//!
//! The blast-radius UI (issue #35) needs to answer "if I change this file,
//! what downstream set would be affected?" — i.e. every node that reaches the
//! selected node through any chain of import edges. That's a reverse BFS over
//! the existing directed graph: start from the selected node's predecessors,
//! walk predecessor edges, collect everything touched.
//!
//! This module keeps the walk pure: it takes a [`Graph`] plus a start node
//! and returns a [`HashSet<NodeId>`]. No egui, no app state, no dedicated
//! reverse adjacency cache — the function scans `graph.edges` directly, which
//! is more than fast enough for the graph sizes we care about (a hub with
//! hundreds of dependents still scans O(edges) once per selection, not per
//! frame). Barrel aggregation is explicitly out of scope for this first cut
//! — commit 2 adds a second entry point that expands barrel display nodes
//! into their members.

use std::collections::{HashSet, VecDeque};

use crate::graph::{Graph, NodeId};

/// Every node that transitively reaches `start` via the directed edge set.
/// A node `n` is in the returned set iff there is a non-empty chain of edges
/// `n -> ... -> start`. `start` itself is included only when it lies on such
/// a chain (i.e. it's inside a cycle that reaches back to itself) — isolated
/// nodes therefore produce an empty cone, which is what the blast-radius UI
/// wants to surface as "0 files depend on this".
///
/// Implemented as an iterative reverse BFS over `graph.edges` to avoid the
/// recursion-depth hazard a DFS would present on deep dependency chains. The
/// result set is unordered — callers that care about ordering should impose
/// their own.
pub fn transitive_dependents(graph: &Graph, start: &NodeId) -> HashSet<NodeId> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    // Unknown start nodes yield an empty cone rather than one containing
    // only themselves — the status bar reports "0 files" for a selection
    // that's been removed underneath us, which is what the caller wants in
    // that edge case.
    if !graph.nodes.contains_key(start) {
        return seen;
    }
    // Reverse BFS seeded from `start`'s direct predecessors rather than
    // `start` itself. That way self enters the cone only if the walk comes
    // back around — exactly the "full SCC for a cycle member, empty for an
    // isolated node" shape the blast-radius UI needs.
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for edge in &graph.edges {
        if edge.to == *start && seen.insert(edge.from.clone()) {
            queue.push_back(edge.from.clone());
        }
    }
    while let Some(cur) = queue.pop_front() {
        for edge in &graph.edges {
            if edge.to == cur && seen.insert(edge.from.clone()) {
                queue.push_back(edge.from.clone());
            }
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeKind};
    use std::path::PathBuf;

    fn file_node(id: &str) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: None,
            kind: NodeKind::File,
        }
    }

    /// Linear chain `a -> b -> c -> d`. The cone from `d` covers every
    /// upstream importer (`a`, `b`, `c`) but excludes `d` itself — the PRD's
    /// "isolated node → empty cone" rule generalises: `start` is only in
    /// the cone when a chain loops back through it.
    #[test]
    fn linear_chain_cone_walks_all_the_way_up() {
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "d");

        let cone = transitive_dependents(&g, &"d".to_string());
        let expected: HashSet<NodeId> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cone, expected);
    }

    #[test]
    fn fan_out_tree_cone_includes_every_ancestor() {
        // Fan-in toward `root`: root's cone is every ancestor (but not
        // root itself); a leaf's cone is empty.
        //
        //   a -> b -+
        //           |--> root
        //   c -> d -+
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d", "root"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "root");
        g.add_edge("c", "d");
        g.add_edge("d", "root");

        let cone = transitive_dependents(&g, &"root".to_string());
        let expected: HashSet<NodeId> =
            ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cone, expected);

        let leaf = transitive_dependents(&g, &"a".to_string());
        assert!(leaf.is_empty(), "a is a source leaf — no one imports it");
    }

    #[test]
    fn node_inside_cycle_cone_contains_full_scc() {
        // `a -> b -> c -> a` is a 3-cycle. Selecting any member yields the
        // full SCC — every member reaches every other member transitively,
        // and the reverse walk from `start` comes back around to include
        // `start` itself.
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "a");

        for start in ["a", "b", "c"] {
            let cone = transitive_dependents(&g, &start.to_string());
            let expected: HashSet<NodeId> =
                ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
            assert_eq!(cone, expected, "cone from {start} must be the full SCC");
        }
    }

    #[test]
    fn isolated_node_cone_is_empty() {
        // `orphan` has no edges in or out — nothing imports it, so its cone
        // is empty per the PRD: "0 files depend on this" on selection.
        let mut g = Graph::new();
        g.add_node(file_node("orphan"));
        g.add_node(file_node("a"));
        g.add_node(file_node("b"));
        g.add_edge("a", "b");

        let cone = transitive_dependents(&g, &"orphan".to_string());
        assert!(cone.is_empty(), "isolated node has no dependents");
    }

    #[test]
    fn disjoint_component_stays_out_of_cone() {
        // Two unconnected components. The cone from a node in one must not
        // include any node from the other, and excludes `start` itself in
        // the non-cyclic case.
        let mut g = Graph::new();
        for id in ["a", "b", "c", "x", "y", "z"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("x", "y");
        g.add_edge("y", "z");

        let cone = transitive_dependents(&g, &"c".to_string());
        let expected: HashSet<NodeId> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cone, expected);
        for forbidden in ["x", "y", "z"] {
            assert!(
                !cone.contains(forbidden),
                "disjoint-component node {forbidden} must stay out of the cone"
            );
        }
    }

    #[test]
    fn missing_start_node_returns_empty_cone() {
        // A selection that no longer exists in the graph (e.g. removed by a
        // watcher diff before the blast-radius UI cleared itself) should fall
        // back to an empty cone rather than reporting a phantom single-node
        // reach.
        let mut g = Graph::new();
        g.add_node(file_node("a"));

        let cone = transitive_dependents(&g, &"missing".to_string());
        assert!(cone.is_empty());
    }
}
