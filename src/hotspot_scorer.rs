//! Pure hotspot scoring on top of the graph + a churn map.
//!
//! The hotspot heatmap (PRD #37) ranks nodes by
//! `score = fan_in * (1 + ln(1 + churn_30d))`. This module owns the formula
//! and returns the sorted-descending list — nothing in here touches IO, the
//! renderer, or the app. Callers supply a graph and a churn map produced by
//! [`crate::churn_provider`]; the scorer walks every node once, so it's
//! cheap enough to rerun on demand when the overlay toggle flips.
//!
//! Formula rationale: multiplying by `1 + ln(1 + churn)` keeps zero-churn
//! files scored at their pure `fan_in` — the `+1` inside the logarithm
//! ensures `ln(1 + 0) = 0`, and the outer `+1` keeps the multiplier at 1.
//! This matches the PRD's "the `1 +` term ensures zero-churn files still
//! rank by fan-in" decision. Churn is log-compressed so a single thrashing
//! file doesn't drown out the high-fan-in signal.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::graph::{Graph, NodeId};

/// Compute the full sorted-descending hotspot ranking for every node in
/// `graph`. The returned `Vec` pairs each node id with its score; the
/// caller is responsible for deciding how many entries to render.
///
/// `churn` is keyed by the same canonical path the indexer stores in
/// `Node.path` (see `GitChurn` — it joins workspace-relative git output
/// against the root). A node whose path is absent from the map is treated
/// as zero-churn rather than skipped, matching the PRD's "missing path is
/// scored as zero-churn (not skipped)" testing decision — this keeps
/// untracked / newly-added files visible in the ranking even before git
/// sees them.
pub fn score_all(graph: &Graph, churn: &HashMap<PathBuf, u32>) -> Vec<(NodeId, f32)> {
    let mut out: Vec<(NodeId, f32)> = graph
        .nodes
        .values()
        .map(|node| {
            let fan_in = graph.dependents_count(&node.id) as f32;
            let churn_30d = churn.get(&node.path).copied().unwrap_or(0);
            let score = fan_in * (1.0 + (1.0 + churn_30d as f32).ln());
            (node.id.clone(), score)
        })
        .collect();
    // Sort descending by score. `f32` doesn't implement `Ord`, so use
    // `partial_cmp` with a reverse; NaN shouldn't appear here — fan-in is
    // a non-negative integer cast and `ln(1+x)` is defined for `x >= 0` —
    // but if it ever did, the fallback drops it to the end.
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeKind};

    fn node(id: &str) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: None,
            kind: NodeKind::File,
        }
    }

    /// Find a node's score in the returned ranking. Panics on miss so
    /// tests fail with a clear message rather than a `None` unwrap at an
    /// unrelated assertion downstream.
    fn score_of(ranked: &[(NodeId, f32)], id: &str) -> f32 {
        ranked
            .iter()
            .find(|(n, _)| n == id)
            .map(|(_, s)| *s)
            .unwrap_or_else(|| panic!("id {id:?} missing from ranking {ranked:?}"))
    }

    #[test]
    fn empty_graph_scores_empty() {
        // A fresh graph with no nodes → empty ranking. No panic, no
        // spurious entries.
        let g = Graph::new();
        let churn: HashMap<PathBuf, u32> = HashMap::new();
        assert!(score_all(&g, &churn).is_empty());
    }

    #[test]
    fn monotone_in_fan_in_at_fixed_churn() {
        // Two target nodes with identical churn but different fan-in:
        // the one pointed at by more edges must rank higher.
        //
        // Graph: a, b, x, y. a -> x, b -> x, b -> y  (x has fan-in 2, y has 1)
        let mut g = Graph::new();
        for id in ["a", "b", "x", "y"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "x");
        g.add_edge("b", "x");
        g.add_edge("b", "y");

        let mut churn = HashMap::new();
        churn.insert(PathBuf::from("x"), 5);
        churn.insert(PathBuf::from("y"), 5);

        let ranked = score_all(&g, &churn);
        assert!(
            score_of(&ranked, "x") > score_of(&ranked, "y"),
            "fan-in 2 must outrank fan-in 1 at equal churn: {ranked:?}",
        );
    }

    #[test]
    fn monotone_in_churn_at_fixed_fan_in() {
        // Two target nodes with identical fan-in but different churn:
        // the one touched more often must rank higher.
        //
        // Graph: a, x, y. a -> x, a -> y  (both fan-in 1)
        let mut g = Graph::new();
        for id in ["a", "x", "y"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "x");
        g.add_edge("a", "y");

        let mut churn = HashMap::new();
        churn.insert(PathBuf::from("x"), 10);
        churn.insert(PathBuf::from("y"), 1);

        let ranked = score_all(&g, &churn);
        assert!(
            score_of(&ranked, "x") > score_of(&ranked, "y"),
            "churn 10 must outrank churn 1 at equal fan-in: {ranked:?}",
        );
    }

    #[test]
    fn zero_churn_degrades_to_fan_in_ranking() {
        // With an empty churn map, the ranking must still separate nodes
        // by fan-in — the `1 +` term inside the `(1 + ln(1+churn))` factor
        // is what makes this work. Without it, zero-churn files would all
        // score zero and collapse into a tie.
        //
        // Graph: three targets with fan-in 3, 2, 1.
        let mut g = Graph::new();
        for id in ["s1", "s2", "s3", "high", "mid", "low"] {
            g.add_node(node(id));
        }
        g.add_edge("s1", "high");
        g.add_edge("s2", "high");
        g.add_edge("s3", "high");
        g.add_edge("s1", "mid");
        g.add_edge("s2", "mid");
        g.add_edge("s1", "low");

        let churn: HashMap<PathBuf, u32> = HashMap::new();
        let ranked = score_all(&g, &churn);

        assert!(score_of(&ranked, "high") > score_of(&ranked, "mid"));
        assert!(score_of(&ranked, "mid") > score_of(&ranked, "low"));
        // And the scores must equal pure fan-in (1 + ln(1)) = 1 multiplier.
        assert_eq!(score_of(&ranked, "high"), 3.0);
        assert_eq!(score_of(&ranked, "mid"), 2.0);
        assert_eq!(score_of(&ranked, "low"), 1.0);
    }

    #[test]
    fn missing_path_scored_as_zero_churn() {
        // A node whose `path` isn't in the churn map is treated as
        // zero-churn, not dropped. The resulting score must equal pure
        // fan-in so the node still appears in the ranking next to its
        // churn-mapped peers.
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("untracked"));
        g.add_edge("a", "untracked");

        // Deliberately empty churn map — "untracked" is absent.
        let churn: HashMap<PathBuf, u32> = HashMap::new();
        let ranked = score_all(&g, &churn);

        assert_eq!(
            score_of(&ranked, "untracked"),
            1.0,
            "missing path must be zero-churn (score = fan-in): {ranked:?}",
        );
        // And the node is present, not filtered out.
        assert!(ranked.iter().any(|(id, _)| id == "untracked"));
    }

    #[test]
    fn ranking_is_sorted_descending() {
        // Overall invariant: the returned vector is sorted descending by
        // score. Downstream renderers rely on this — they take the prefix
        // for "top N" displays without re-sorting.
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("a", "c");
        g.add_edge("d", "c");
        g.add_edge("a", "d");
        g.add_edge("b", "d");
        g.add_edge("c", "d");

        let mut churn = HashMap::new();
        churn.insert(PathBuf::from("b"), 7);
        churn.insert(PathBuf::from("c"), 2);

        let ranked = score_all(&g, &churn);
        for pair in ranked.windows(2) {
            assert!(
                pair[0].1 >= pair[1].1,
                "ranking must be sorted descending: {ranked:?}",
            );
        }
    }
}
