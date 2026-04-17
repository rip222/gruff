//! Dead-code orphan detection over a built graph.
//!
//! Composition layer on top of [`reachability::unreachable_from`] + a simple
//! in-degree scan. The PRD insists on surfacing the two distinct notions of
//! "orphan":
//!
//! * **Unreachable from entry** — no entry-point-rooted walk touches the
//!   file. Catches mutually-importing islands that an in-degree check would
//!   miss.
//! * **No incoming imports** — the file has zero in-degree. Catches files
//!   that aren't declared as entries but have nothing importing them; may
//!   still be reachable from an entry (e.g. a test fixture that imports a
//!   helper which is itself the test target).
//!
//! Keeping them separate follows the decision doc verbatim: "Two distinct
//! sets, not one." Callers render them as two sub-lists in the sidebar and
//! flag a node as orphan-for-the-canvas when it appears in either.

use std::collections::HashSet;

use crate::graph::{Graph, NodeId};
use crate::reachability;

/// Two-part orphan report. Both sets are populated in a single `detect` call
/// and stored verbatim on [`GruffApp`] so subsequent frames can render
/// without re-walking the graph.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OrphanSets {
    /// Files no entry-point-rooted forward walk reaches. Superset of islands,
    /// two-file mutual-imports, etc. that an in-degree check would miss.
    pub unreachable_from_entries: HashSet<NodeId>,
    /// Files with zero incoming edges. May overlap with
    /// `unreachable_from_entries` (a file with no imports and no importers
    /// appears in both); may also lie entirely outside it (a file reachable
    /// from an entry that happens to have no importers, e.g. the entry
    /// itself).
    pub no_incoming_imports: HashSet<NodeId>,
}

/// Run both orphan passes against `graph` using `entries` as the reachability
/// roots. The pass is cheap enough (one `unreachable_from` call + one linear
/// edge scan) that callers rebuild the whole [`OrphanSets`] on every full
/// graph build or watcher diff rather than trying to patch it incrementally.
pub fn detect(graph: &Graph, entries: &HashSet<NodeId>) -> OrphanSets {
    let unreachable_from_entries = reachability::unreachable_from(graph, entries);

    // One linear sweep over edges builds the "has at least one incoming" set;
    // the complement (restricted to the graph's nodes so synthetic ids don't
    // leak in) is the zero-in-degree set we want.
    let mut has_incoming: HashSet<&NodeId> = HashSet::new();
    for edge in &graph.edges {
        has_incoming.insert(&edge.to);
    }
    let mut no_incoming_imports: HashSet<NodeId> = HashSet::new();
    for id in graph.nodes.keys() {
        if !has_incoming.contains(id) {
            no_incoming_imports.insert(id.clone());
        }
    }

    OrphanSets {
        unreachable_from_entries,
        no_incoming_imports,
    }
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

    /// Sanity: the two sub-sets are conceptually distinct — a simple graph
    /// where one file is in one set and a different file is in the other
    /// should keep them apart. `entry` is reachable (it's the seed) but has
    /// no importers. `island` is unreachable but imports something else, so
    /// it has no incoming imports either — but the point here is that the
    /// sets don't collapse to one under the detector.
    #[test]
    fn two_sets_are_not_forced_to_be_identical() {
        let mut g = Graph::new();
        for id in ["entry", "a", "b"] {
            g.add_node(file_node(id));
        }
        g.add_edge("entry", "a");
        g.add_edge("a", "b");
        let mut entries = HashSet::new();
        entries.insert("entry".to_string());
        let sets = detect(&g, &entries);
        // Nothing is unreachable in the linear entry→a→b chain, so the
        // unreachable set is empty. `entry` has no importers so it's in
        // `no_incoming_imports`. The sets are therefore different — one
        // empty, one with a single member — which is enough to prove they
        // aren't the same pass under a different name.
        assert!(sets.unreachable_from_entries.is_empty());
        assert!(sets.no_incoming_imports.contains("entry"));
    }

    /// A lone file with no edges at all appears in both sets: it has no
    /// importers (zero in-degree) AND it isn't reachable from any entry
    /// that's wired to anything else. Matches the PRD's explicit matrix
    /// row.
    #[test]
    fn file_with_no_edges_appears_in_both_sets() {
        let mut g = Graph::new();
        g.add_node(file_node("entry"));
        g.add_node(file_node("lonely"));
        g.add_edge("entry", "entry"); // self-loop on entry only

        let mut entries = HashSet::new();
        entries.insert("entry".to_string());

        let sets = detect(&g, &entries);
        assert!(
            sets.unreachable_from_entries.contains("lonely"),
            "isolated file must be reported unreachable from entry",
        );
        assert!(
            sets.no_incoming_imports.contains("lonely"),
            "isolated file has zero in-degree and must show up in the no-incoming set",
        );
    }

    /// The PRD's canonical "island" case: `a <-> b` forms a two-node cycle
    /// disconnected from any entry. Every member has at least one incoming
    /// edge (from the other island member), so an in-degree check alone
    /// would miss them — they only surface through the reachability pass.
    #[test]
    fn mutual_import_island_only_surfaces_in_unreachable_set() {
        let mut g = Graph::new();
        for id in ["entry", "a", "b"] {
            g.add_node(file_node(id));
        }
        // Entry loops to itself so it has an incoming edge; keeps the
        // no-incoming assertion tight.
        g.add_edge("entry", "entry");
        // Island:
        g.add_edge("a", "b");
        g.add_edge("b", "a");

        let mut entries = HashSet::new();
        entries.insert("entry".to_string());

        let sets = detect(&g, &entries);
        assert!(
            sets.unreachable_from_entries.contains("a"),
            "island node must be unreachable from entry",
        );
        assert!(
            sets.unreachable_from_entries.contains("b"),
            "island node must be unreachable from entry",
        );
        // But both island members have incoming edges from each other, so
        // the in-degree check is blind to them.
        assert!(
            !sets.no_incoming_imports.contains("a"),
            "island member 'a' has an incoming edge from 'b' — must not appear in zero-in-degree set",
        );
        assert!(
            !sets.no_incoming_imports.contains("b"),
            "island member 'b' has an incoming edge from 'a' — must not appear in zero-in-degree set",
        );
    }

    /// A file reachable from an entry with zero incoming edges appears ONLY
    /// in `no_incoming_imports`. The classic example: the entry point
    /// itself, which is reachable from the entry set trivially but has no
    /// importers.
    #[test]
    fn entry_itself_is_only_in_no_incoming_set() {
        // entry -> a -> b. `entry` has no incoming edge and is reachable
        // from itself, so it surfaces only in `no_incoming_imports`.
        let mut g = Graph::new();
        for id in ["entry", "a", "b"] {
            g.add_node(file_node(id));
        }
        g.add_edge("entry", "a");
        g.add_edge("a", "b");

        let mut entries = HashSet::new();
        entries.insert("entry".to_string());

        let sets = detect(&g, &entries);
        assert!(
            !sets.unreachable_from_entries.contains("entry"),
            "entry is reachable from itself — must not be reported unreachable",
        );
        assert!(
            sets.no_incoming_imports.contains("entry"),
            "entry has no importers — must be in the zero-in-degree set",
        );
    }
}
