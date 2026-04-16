use std::collections::HashMap;
use std::path::PathBuf;

pub type NodeId = String;

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub path: PathBuf,
    pub label: String,
    /// Name of the workspace package that owns this file, if any. `None` means
    /// the node isn't attributed to a package (e.g. stray file outside every
    /// workspace package). Set by the indexer from [`crate::workspace::Workspace`].
    pub package: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub nodes: HashMap<NodeId, Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn remove_node(&mut self, id: &str) {
        self.nodes.remove(id);
        self.edges.retain(|e| e.from != id && e.to != id);
    }

    pub fn add_edge(&mut self, from: &str, to: &str) {
        let edge = Edge {
            from: from.to_string(),
            to: to.to_string(),
        };
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    pub fn update_node_label(&mut self, id: &str, label: String) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.label = label;
        }
    }

    pub fn neighbors<'a>(&'a self, id: &str) -> Vec<&'a NodeId> {
        self.edges
            .iter()
            .filter(|e| e.from == id)
            .map(|e| &e.to)
            .collect()
    }

    pub fn dependents_count(&self, id: &str) -> usize {
        self.edges.iter().filter(|e| e.to == id).count()
    }

    /// Detect cycles via Tarjan's strongly-connected-components algorithm.
    ///
    /// Returns one entry per SCC that contains at least one cycle: either
    /// an SCC with more than one node, or a single node with a self-loop.
    /// Acyclic (trivial) SCCs are omitted, so the result is exactly the
    /// set of circular-dependency groups in the graph.
    ///
    /// Node order within each returned cycle reflects Tarjan's stack-pop
    /// order; callers that care about presentation should impose their own
    /// order.
    ///
    /// Implementation is iterative to stay safe on large graphs — a
    /// recursive DFS would risk blowing Rust's thread stack on repos with
    /// thousands of transitively-connected files.
    pub fn cycles(&self) -> Vec<Vec<NodeId>> {
        let ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        let n = ids.len();
        if n == 0 {
            return Vec::new();
        }
        let index_of: HashMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut has_self_loop: Vec<bool> = vec![false; n];
        for e in &self.edges {
            let (Some(&u), Some(&v)) = (
                index_of.get(e.from.as_str()),
                index_of.get(e.to.as_str()),
            ) else {
                continue;
            };
            adj[u].push(v);
            if u == v {
                has_self_loop[u] = true;
            }
        }

        let mut next_index: usize = 0;
        let mut indices: Vec<Option<usize>> = vec![None; n];
        let mut lowlinks: Vec<usize> = vec![0; n];
        let mut on_stack: Vec<bool> = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut sccs: Vec<Vec<NodeId>> = Vec::new();

        for start in 0..n {
            if indices[start].is_some() {
                continue;
            }

            // (node, next child cursor) — Tarjan's recursive DFS flattened
            // into an explicit frame stack.
            let mut frames: Vec<(usize, usize)> = Vec::new();

            indices[start] = Some(next_index);
            lowlinks[start] = next_index;
            next_index += 1;
            stack.push(start);
            on_stack[start] = true;
            frames.push((start, 0));

            while let Some(&(v, i)) = frames.last() {
                if i < adj[v].len() {
                    let w = adj[v][i];
                    frames.last_mut().unwrap().1 = i + 1;

                    if indices[w].is_none() {
                        indices[w] = Some(next_index);
                        lowlinks[w] = next_index;
                        next_index += 1;
                        stack.push(w);
                        on_stack[w] = true;
                        frames.push((w, 0));
                    } else if on_stack[w] {
                        // Tree/back edge into active SCC — tighten lowlink.
                        lowlinks[v] = lowlinks[v].min(indices[w].unwrap());
                    }
                    continue;
                }

                // Exhausted v's successors — decide if v roots an SCC.
                if lowlinks[v] == indices[v].unwrap() {
                    let mut component: Vec<NodeId> = Vec::new();
                    loop {
                        let w = stack.pop().expect("tarjan: stack cannot be empty at scc close");
                        on_stack[w] = false;
                        component.push(ids[w].clone());
                        if w == v {
                            break;
                        }
                    }
                    // Skip acyclic singletons — only keep real cycles.
                    if component.len() > 1 || has_self_loop[v] {
                        sccs.push(component);
                    }
                }

                frames.pop();
                // Propagate v's lowlink up to its parent frame, matching the
                // "lowlink[parent] = min(lowlink[parent], lowlink[child])"
                // step that a recursive implementation would do on return.
                let v_low = lowlinks[v];
                if let Some(&(parent, _)) = frames.last() {
                    lowlinks[parent] = lowlinks[parent].min(v_low);
                }
            }
        }

        sccs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn node(id: &str) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: None,
        }
    }

    #[test]
    fn add_and_remove_node() {
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_edge("a", "b");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);

        g.remove_node("a");
        assert_eq!(g.nodes.len(), 1);
        assert!(g.edges.is_empty());
    }

    #[test]
    fn neighbors_and_dependents() {
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_node(node("c"));
        g.add_edge("a", "b");
        g.add_edge("c", "b");

        assert_eq!(g.neighbors("a"), vec![&"b".to_string()]);
        assert_eq!(g.dependents_count("b"), 2);
        assert_eq!(g.dependents_count("a"), 0);
    }

    #[test]
    fn add_edge_is_idempotent() {
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_edge("a", "b");
        g.add_edge("a", "b");
        assert_eq!(g.edges.len(), 1);
    }

    fn cycle_sets(cycles: &[Vec<NodeId>]) -> Vec<std::collections::BTreeSet<String>> {
        let mut out: Vec<_> = cycles
            .iter()
            .map(|c| c.iter().cloned().collect::<std::collections::BTreeSet<_>>())
            .collect();
        out.sort_by_key(|s| s.iter().next().cloned().unwrap_or_default());
        out
    }

    #[test]
    fn cycles_empty_for_acyclic_graph() {
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_node(node("c"));
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        assert!(g.cycles().is_empty());
    }

    #[test]
    fn cycles_detects_single_cycle() {
        // a -> b -> c -> a
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_node(node("c"));
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "a");

        let cycles = g.cycles();
        assert_eq!(cycles.len(), 1);
        let expected: std::collections::BTreeSet<_> =
            ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cycle_sets(&cycles), vec![expected]);
    }

    #[test]
    fn cycles_detects_self_loop_as_cycle() {
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_edge("a", "a");
        let cycles = g.cycles();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0], vec!["a".to_string()]);
    }

    #[test]
    fn cycles_detects_multiple_disjoint_cycles() {
        // Two independent cycles: {a,b} and {c,d,e}, with a bridge edge
        // b -> c that does NOT merge the SCCs (it's one-way).
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d", "e"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "a");
        g.add_edge("b", "c"); // one-way bridge — not part of any cycle
        g.add_edge("c", "d");
        g.add_edge("d", "e");
        g.add_edge("e", "c");

        let sets = cycle_sets(&g.cycles());
        assert_eq!(sets.len(), 2);
        let ab: std::collections::BTreeSet<_> =
            ["a", "b"].iter().map(|s| s.to_string()).collect();
        let cde: std::collections::BTreeSet<_> =
            ["c", "d", "e"].iter().map(|s| s.to_string()).collect();
        assert!(sets.contains(&ab));
        assert!(sets.contains(&cde));
    }

    #[test]
    fn cycles_collapses_nested_cycles_into_one_scc() {
        // Two simple cycles that share node `b`:
        //   a -> b -> c -> a   and   b -> d -> b
        // Tarjan reports them as a single strongly-connected component
        // containing all four nodes — which matches the user's intent of
        // "the cyclic region," not "every simple loop."
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "a");
        g.add_edge("b", "d");
        g.add_edge("d", "b");

        let cycles = g.cycles();
        assert_eq!(cycles.len(), 1);
        let got: std::collections::BTreeSet<_> = cycles[0].iter().cloned().collect();
        let expected: std::collections::BTreeSet<_> =
            ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn cycles_ignores_acyclic_tail_off_a_cycle() {
        // a <-> b is a cycle; c and d hang off but aren't part of any cycle.
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d"] {
            g.add_node(node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "a");
        g.add_edge("b", "c");
        g.add_edge("c", "d");

        let cycles = g.cycles();
        assert_eq!(cycles.len(), 1);
        let got: std::collections::BTreeSet<_> = cycles[0].iter().cloned().collect();
        let expected: std::collections::BTreeSet<_> =
            ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(got, expected);
    }
}
