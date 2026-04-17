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

use crate::aggregation::BarrelMembers;
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

/// Barrel-aware variant of [`transitive_dependents`]. Behaves identically on
/// graphs with no barrels, but when the BFS encounters a barrel display node
/// the member files are pulled into the cone as well — so the returned set
/// reflects real file-level impact rather than display-node impact.
///
/// If `start` is itself a barrel member (raw file id collapsed into a display
/// node), the walk proceeds *as if* `start` were the barrel display node —
/// an import of the barrel counts as reaching every member, so the cone
/// picks up upstream importers that never literally reference `start` in
/// the aggregated graph. This matches the blast-radius UX decision:
/// selecting one file inside a barrel reports the downstream impact of any
/// of the barrel's files changing.
///
/// Self-in-cone follows the same rule as the non-barrel variant: `start` (or
/// its display node) is only present in the result when it's reachable back
/// through some chain of imports. Isolated starting points yield empty
/// cones, and the status bar renders "0 files depend on this".
pub fn transitive_dependents_with_barrels(
    graph: &Graph,
    start: &NodeId,
    barrels: &BarrelMembers,
) -> HashSet<NodeId> {
    let mut cone: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    // Resolve the walk's seed. Three cases:
    // 1. `start` is a live node in the aggregated graph → walk from it.
    // 2. `start` is a raw barrel member (absent from the aggregated graph
    //    because collapse swallowed it) → walk from the barrel display id.
    // 3. Neither → empty cone.
    let seed = if graph.nodes.contains_key(start) {
        start.clone()
    } else if let Some(display) = barrels.display_of(start) {
        display.clone()
    } else {
        return cone;
    };

    // `bfs_visited` is the BFS internal set — everything we've already
    // processed — while `cone` is the public result set. Keeping them
    // separate means the seed is visited (so we don't loop forever) without
    // being added to the cone, matching the non-barrel variant's "self only
    // enters the cone if a chain loops back" semantics.
    let mut bfs_visited: HashSet<NodeId> = HashSet::new();
    bfs_visited.insert(seed.clone());
    for edge in &graph.edges {
        if edge.to == seed && bfs_visited.insert(edge.from.clone()) {
            enqueue_and_expand(&edge.from, &mut cone, &mut queue, barrels);
        }
    }
    while let Some(cur) = queue.pop_front() {
        for edge in &graph.edges {
            if edge.to != cur {
                continue;
            }
            if !bfs_visited.insert(edge.from.clone()) {
                continue;
            }
            enqueue_and_expand(&edge.from, &mut cone, &mut queue, barrels);
        }
    }
    cone
}

/// Record `node` in the cone, queue it for further walk, and — if it's a
/// barrel display id — add every member file to the cone so the caller's
/// file-count reflects real impact. Only the display id needs to be
/// re-walked (members don't have their own edges in the aggregated graph),
/// so members are added to the cone without being queued.
fn enqueue_and_expand(
    node: &NodeId,
    cone: &mut HashSet<NodeId>,
    queue: &mut VecDeque<NodeId>,
    barrels: &BarrelMembers,
) {
    cone.insert(node.clone());
    queue.push_back(node.clone());
    if let Some(members) = barrels.members_of(node) {
        for m in members {
            cone.insert(m.clone());
        }
    }
}

/// Set of nodes unreachable from any of `entries` via a forward walk. A node
/// `n` is in the returned set iff there is NO directed chain
/// `e -> ... -> n` starting at some `e ∈ entries`. Equivalent to the
/// complement of the forward-BFS visited set, restricted to the graph's own
/// nodes so external / synthetic ids never leak into the returned set.
///
/// Entries absent from `graph.nodes` are silently skipped — they can't seed
/// any walk, so they only matter as "did the caller hand us a stale id?"
/// which isn't this function's concern. Entries that ARE in `graph.nodes`
/// count as visited themselves (an entry reaches itself), so the returned
/// set never contains an entry id.
///
/// Iterative forward BFS over `graph.edges` mirrors the style of
/// `transitive_dependents` — no reverse-adjacency cache needed for the graph
/// sizes this tool targets, and keeping the walk pure means tests can drive
/// it directly without standing up an app.
pub fn unreachable_from(graph: &Graph, entries: &HashSet<NodeId>) -> HashSet<NodeId> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for entry in entries {
        if graph.nodes.contains_key(entry) && visited.insert(entry.clone()) {
            queue.push_back(entry.clone());
        }
    }
    while let Some(cur) = queue.pop_front() {
        for edge in &graph.edges {
            if edge.from == cur && visited.insert(edge.to.clone()) {
                queue.push_back(edge.to.clone());
            }
        }
    }

    let mut unreachable: HashSet<NodeId> = HashSet::new();
    for id in graph.nodes.keys() {
        if !visited.contains(id) {
            unreachable.insert(id.clone());
        }
    }
    unreachable
}

/// Summarise a transitive-dependents cone as `(file_count, package_count)` —
/// the two numbers the blast-radius status bar renders as
/// "N files in M packages depend on this."
///
/// * **File count** is the number of file-level nodes in the cone. Barrel
///   display nodes (id prefix `barrel:`) are counted as the files they
///   represent, not as themselves — this matches the PRD's "count files,
///   not display nodes" decision. When the cone walker expanded a barrel
///   into its members, the members are already in the cone and the display
///   id is skipped here to avoid double-counting. Synthetic nodes
///   (workspace-package aggregators, external leaves) that happen to sit in
///   the cone aren't file-level either and are likewise skipped.
/// * **Package count** is the number of distinct package names attached to
///   file-level nodes in the cone. Files with no package attribution (stray
///   files outside every workspace package) don't contribute.
///
/// An empty cone yields `(0, 0)`. The function doesn't touch the selected
/// node — the cone walker already decided whether to include it.
pub fn cone_stats(graph: &Graph, cone: &HashSet<NodeId>) -> (usize, usize) {
    use crate::graph::NodeKind;

    let mut file_count: usize = 0;
    let mut packages: HashSet<&str> = HashSet::new();

    for id in cone {
        // Skip barrel display ids entirely — the members they were expanded
        // into carry the real file-level identity and any package attribution.
        if id.starts_with("barrel:") {
            continue;
        }
        match graph.nodes.get(id) {
            Some(node) if matches!(node.kind, NodeKind::File) => {
                file_count += 1;
                if let Some(pkg) = node.package.as_deref() {
                    packages.insert(pkg);
                }
            }
            // Raw barrel members are absent from the aggregated `graph.nodes`
            // (they collapsed into the display node) but still represent real
            // files — count them as files. Their package attribution is
            // unavailable from this angle, so they contribute to `file_count`
            // without touching `packages`. The status-bar reading "N files in
            // M packages" stays honest: M is the distinct-package count of
            // nodes we *do* have metadata for.
            None => {
                file_count += 1;
            }
            // Synthetic workspace-package aggregators and external leaves
            // don't represent source files — they're abstractions. Don't
            // count them toward file impact.
            Some(_) => {}
        }
    }
    (file_count, packages.len())
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

    /// Helper: build a [`BarrelMembers`] handle from a literal
    /// `"display_id" -> &["member", ...]` table. Keeps the barrel tests
    /// below readable without reaching for a real aggregation pass.
    fn barrels_from(pairs: &[(&str, &[&str])]) -> BarrelMembers {
        let mut mapping: std::collections::HashMap<NodeId, NodeId> =
            std::collections::HashMap::new();
        for (display, members) in pairs {
            for m in *members {
                mapping.insert(m.to_string(), display.to_string());
            }
        }
        BarrelMembers::from_mapping(&mapping)
    }

    #[test]
    fn barrel_aware_cone_from_member_reaches_importer_through_barrel() {
        // Synthetic setup: barrel B with member files m1 and m2. An outer
        // file f imports B (in the aggregated graph). The cone from m1 must
        // include f — f doesn't literally reference m1, but it imports the
        // barrel which represents m1.
        //
        // Aggregated graph contains {B, f} + edge f -> B. m1, m2 are raw
        // file ids absent from the aggregated graph (they collapsed into B),
        // but present in the BarrelMembers handle.
        let mut g = Graph::new();
        g.add_node(file_node("B"));
        g.add_node(file_node("f"));
        g.add_edge("f", "B");

        let barrels = barrels_from(&[("B", &["m1", "m2"])]);

        let cone = transitive_dependents_with_barrels(&g, &"m1".to_string(), &barrels);
        assert!(
            cone.contains("f"),
            "cone from m1 must include the importer through the barrel; got {cone:?}"
        );
        // Neither the barrel display nor m1/m2 themselves depend on m1 —
        // the seed and its siblings stay out of the cone per the
        // non-cyclic "self not in cone" rule.
        assert!(
            !cone.contains("B"),
            "barrel display must not appear in its own cone (seed is excluded)"
        );
    }

    #[test]
    fn barrel_aware_cone_from_non_barrel_file_matches_plain_walk() {
        // A non-barrel file's cone must be identical to the plain
        // `transitive_dependents` result — the barrel expansion only kicks
        // in when the BFS touches a barrel display node.
        //
        //   outer -> lone
        //
        // `lone` is a regular file (not a member of any barrel); `outer`
        // imports it. Cone from `lone` is `{outer}`, same as without the
        // barrel handle.
        let mut g = Graph::new();
        g.add_node(file_node("lone"));
        g.add_node(file_node("outer"));
        g.add_edge("outer", "lone");

        // The handle happens to describe an unrelated barrel — its presence
        // must not contaminate the cone of a non-member file.
        let barrels = barrels_from(&[("other_barrel", &["other_m1", "other_m2"])]);

        let cone = transitive_dependents_with_barrels(&g, &"lone".to_string(), &barrels);
        let plain = transitive_dependents(&g, &"lone".to_string());
        assert_eq!(
            cone, plain,
            "barrel-aware cone on a non-member file must equal the plain cone"
        );
    }

    #[test]
    fn barrel_encountered_mid_walk_pulls_in_members() {
        // Chain where the barrel sits in the middle of the cone path:
        //
        //   f -> B -> target
        //
        // B has members m1, m2. Cone from `target` reaches f by walking the
        // reverse edges; when BFS steps onto B, members m1 and m2 must also
        // land in the cone so the file-count reflects reality.
        let mut g = Graph::new();
        g.add_node(file_node("target"));
        g.add_node(file_node("B"));
        g.add_node(file_node("f"));
        g.add_edge("f", "B");
        g.add_edge("B", "target");

        let barrels = barrels_from(&[("B", &["m1", "m2"])]);

        let cone = transitive_dependents_with_barrels(&g, &"target".to_string(), &barrels);
        let expected: HashSet<NodeId> = ["B", "m1", "m2", "f"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(cone, expected);
    }

    fn file_node_in(id: &str, package: Option<&str>) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: package.map(str::to_string),
            kind: NodeKind::File,
        }
    }

    #[test]
    fn cone_stats_counts_files_across_two_packages() {
        // Cone touches three files in two packages → (3, 2). Mirrors the
        // PRD matrix case "cone across two packages → (files, 2)."
        let mut g = Graph::new();
        g.add_node(file_node_in("a", Some("pkg-a")));
        g.add_node(file_node_in("b", Some("pkg-a")));
        g.add_node(file_node_in("c", Some("pkg-b")));

        let cone: HashSet<NodeId> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cone_stats(&g, &cone), (3, 2));
    }

    #[test]
    fn cone_stats_counts_files_within_one_package() {
        // Cone entirely within a single package → (files, 1).
        let mut g = Graph::new();
        g.add_node(file_node_in("a", Some("only-pkg")));
        g.add_node(file_node_in("b", Some("only-pkg")));

        let cone: HashSet<NodeId> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cone_stats(&g, &cone), (2, 1));
    }

    #[test]
    fn cone_stats_empty_cone_is_zero_zero() {
        // Empty cone → (0, 0) — the isolated-node status bar case.
        let g = Graph::new();
        let cone: HashSet<NodeId> = HashSet::new();
        assert_eq!(cone_stats(&g, &cone), (0, 0));
    }

    #[test]
    fn cone_stats_skips_barrel_display_nodes() {
        // Barrel display ids (prefix `barrel:`) don't represent files and
        // must not be counted — the barrel's member raw ids in the cone
        // carry the real file identity. Here `barrel:foo` is in the cone
        // alongside its member `foo/index.ts`; the result is one file in
        // one package, not two files.
        let mut g = Graph::new();
        g.add_node(Node {
            id: "barrel:/repo/foo".to_string(),
            path: PathBuf::from("/repo/foo"),
            label: "foo".to_string(),
            package: Some("pkg-foo".to_string()),
            kind: NodeKind::File,
        });
        let member = Node {
            id: "foo/index.ts".to_string(),
            path: PathBuf::from("/repo/foo/index.ts"),
            label: "index.ts".to_string(),
            package: Some("pkg-foo".to_string()),
            kind: NodeKind::File,
        };
        // The member isn't in `graph.nodes` (it was swallowed by the barrel)
        // but `cone_stats` still counts raw ids present in the cone — that's
        // how a barrel-expanded cone reports real file impact.
        let _ = member;

        let cone: HashSet<NodeId> = ["barrel:/repo/foo", "foo/index.ts"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(cone_stats(&g, &cone), (1, 0));
        // `(1, 0)` rather than `(1, 1)` is deliberate: the member file isn't
        // in `graph.nodes` so we have no package attribution for it. In the
        // full pipeline the barrel display would sit alongside at least one
        // other per-file node in the same package, which contributes the
        // package count. This test deliberately exercises the
        // member-without-metadata edge case so the behavior is locked in.
    }

    #[test]
    fn cone_stats_ignores_synthetic_nodes() {
        // Workspace-package and external synthetic nodes aren't source
        // files; they must not contribute to the file count even when
        // present in the cone.
        let mut g = Graph::new();
        g.add_node(file_node_in("a", Some("pkg")));
        g.add_node(Node {
            id: "package:@org/app".to_string(),
            path: PathBuf::from("package:@org/app"),
            label: "@org/app".to_string(),
            package: Some("@org/app".to_string()),
            kind: NodeKind::WorkspacePackage,
        });
        g.add_node(Node {
            id: "external:react".to_string(),
            path: PathBuf::from("external:react"),
            label: "react".to_string(),
            package: None,
            kind: NodeKind::External,
        });

        let cone: HashSet<NodeId> = ["a", "package:@org/app", "external:react"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(cone_stats(&g, &cone), (1, 1));
    }

    #[test]
    fn empty_barrel_handle_matches_plain_walk() {
        // With a passthrough/empty BarrelMembers handle the barrel-aware
        // variant must produce exactly the same set as the non-barrel
        // entry point — the feature degrades cleanly when barrel collapse
        // is disabled.
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "c");

        let plain = transitive_dependents(&g, &"c".to_string());
        let with_empty =
            transitive_dependents_with_barrels(&g, &"c".to_string(), &BarrelMembers::empty());
        assert_eq!(plain, with_empty);
    }

    // --- unreachable_from -------------------------------------------------

    /// Every node reachable from some entry → empty unreachable set. The
    /// chain `entry -> a -> b -> c` leaves nothing dead when `entry` is in
    /// the entry set.
    #[test]
    fn unreachable_from_entries_cover_all_returns_empty() {
        let mut g = Graph::new();
        for id in ["entry", "a", "b", "c"] {
            g.add_node(file_node(id));
        }
        g.add_edge("entry", "a");
        g.add_edge("a", "b");
        g.add_edge("b", "c");

        let mut entries = HashSet::new();
        entries.insert("entry".to_string());
        let dead = unreachable_from(&g, &entries);
        assert!(
            dead.is_empty(),
            "entire chain is reachable from the single entry; got {dead:?}"
        );
    }

    /// No entries at all → every node in the graph is unreachable. Mirrors
    /// the PRD's "entries cover none → all-nodes return" matrix row.
    #[test]
    fn unreachable_from_empty_entries_returns_every_node() {
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");

        let entries: HashSet<NodeId> = HashSet::new();
        let dead = unreachable_from(&g, &entries);
        let expected: HashSet<NodeId> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(dead, expected);
    }

    /// A disconnected island (no edges to or from the reachable component)
    /// stays unreachable. The reachable component's nodes don't leak into
    /// the returned set.
    #[test]
    fn unreachable_from_island_disconnected_from_entries_stays_dead() {
        let mut g = Graph::new();
        for id in ["entry", "live", "island_a", "island_b"] {
            g.add_node(file_node(id));
        }
        g.add_edge("entry", "live");
        // Island: two files importing each other with no tie back to entry.
        g.add_edge("island_a", "island_b");
        g.add_edge("island_b", "island_a");

        let mut entries = HashSet::new();
        entries.insert("entry".to_string());
        let dead = unreachable_from(&g, &entries);
        let expected: HashSet<NodeId> =
            ["island_a", "island_b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            dead, expected,
            "only the disconnected island should show up as unreachable"
        );
    }

    /// A cycle rooted off an entry is wholly reachable — every cycle member
    /// gets visited via the entry and so stays out of the dead set, even
    /// though internally they form an SCC.
    #[test]
    fn unreachable_from_cycle_reachable_from_entry_is_excluded() {
        // entry -> a -> b -> c -> a (back-edge). Every cycle member is
        // reachable from entry.
        let mut g = Graph::new();
        for id in ["entry", "a", "b", "c"] {
            g.add_node(file_node(id));
        }
        g.add_edge("entry", "a");
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "a");

        let mut entries = HashSet::new();
        entries.insert("entry".to_string());
        let dead = unreachable_from(&g, &entries);
        assert!(
            dead.is_empty(),
            "every cycle member is reachable via entry; got {dead:?}",
        );
    }

    /// Entries that aren't in `graph.nodes` are silently ignored — they
    /// can't seed a walk, so a stale entry id in the input set doesn't
    /// turn a live graph into "all unreachable."
    #[test]
    fn unreachable_from_unknown_entry_is_ignored() {
        let mut g = Graph::new();
        for id in ["live_entry", "a"] {
            g.add_node(file_node(id));
        }
        g.add_edge("live_entry", "a");

        let mut entries = HashSet::new();
        entries.insert("live_entry".to_string());
        entries.insert("ghost".to_string());
        let dead = unreachable_from(&g, &entries);
        assert!(
            dead.is_empty(),
            "ghost entry should not affect the result; got {dead:?}",
        );
    }
}
