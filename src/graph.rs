use std::collections::HashMap;
use std::path::PathBuf;

pub type NodeId = String;

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub path: PathBuf,
    pub label: String,
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
}
