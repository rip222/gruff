//! Session-scoped visibility filter for the dependency graph.
//!
//! Thin adapter over a [`HashSet<NodeId>`] of hidden ids. The set is the
//! canonical "what is off" signal that the canvas, layout simulation, cycle
//! detection, and sidebar all read from. A node is *hidden* when its id is in
//! the set; *visible* otherwise. Edges with any hidden endpoint are treated
//! as hidden too — that derivation happens at the caller, since this module
//! deliberately knows nothing about graph structure.
//!
//! Per PRD #16 the filter is in-memory only and must reset on every folder
//! open. See [`GruffApp::load_folder`](crate::app) which calls [`clear`] to
//! enforce that.
//!
//! Kept as a deep, pure module with no egui, no graph, and no layout
//! dependencies so the invariants can be unit-tested cheaply.
//!
//! [`clear`]: FilterState::clear

use std::collections::HashSet;

use crate::graph::NodeId;

/// Mutable set of hidden node ids.
///
/// Invariant: a node is considered hidden iff its id is in [`Self::hidden`].
/// All public mutators route through this struct so call sites don't
/// construct transient "is this node hidden?" state out of band.
#[derive(Debug, Default, Clone)]
pub struct FilterState {
    hidden: HashSet<NodeId>,
}

impl FilterState {
    /// Fresh filter with nothing hidden.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when `id` is currently in the hide set.
    pub fn is_hidden(&self, id: &str) -> bool {
        self.hidden.contains(id)
    }

    /// True when nothing is hidden — lets callers skip the filtered-graph
    /// build when the user hasn't toggled anything.
    pub fn is_empty(&self) -> bool {
        self.hidden.is_empty()
    }

    /// Borrow the underlying hide set. Canvas rendering and layout-sync
    /// paths both only need read access, so exposing the set avoids a
    /// per-call `clone`.
    pub fn hidden(&self) -> &HashSet<NodeId> {
        &self.hidden
    }

    /// Flip `id`'s membership: hidden → visible, visible → hidden. Returns
    /// the new hidden state so callers can decide whether a repaint /
    /// layout-sync is worth doing (always true for now, but keeping the
    /// return value lets future callers short-circuit).
    pub fn toggle(&mut self, id: &NodeId) -> bool {
        if !self.hidden.insert(id.clone()) {
            self.hidden.remove(id);
            false
        } else {
            true
        }
    }

    /// Mark `id` as hidden. No-op if it's already in the set.
    pub fn hide(&mut self, id: &NodeId) {
        self.hidden.insert(id.clone());
    }

    /// Mark `id` as visible. No-op if it wasn't hidden.
    pub fn show(&mut self, id: &str) {
        self.hidden.remove(id);
    }

    /// Drop every hidden id. Called on folder open so filter state stays
    /// session-scoped per PRD #16.
    pub fn clear(&mut self) {
        self.hidden.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> NodeId {
        s.to_string()
    }

    #[test]
    fn fresh_filter_hides_nothing() {
        let f = FilterState::new();
        assert!(f.is_empty());
        assert!(!f.is_hidden("anything"));
        assert!(f.hidden().is_empty());
    }

    #[test]
    fn toggle_flips_membership_and_reports_new_state() {
        let mut f = FilterState::new();
        // First toggle hides.
        assert!(f.toggle(&id("a")));
        assert!(f.is_hidden("a"));
        // Second toggle unhides.
        assert!(!f.toggle(&id("a")));
        assert!(!f.is_hidden("a"));
    }

    #[test]
    fn hide_and_show_are_idempotent() {
        let mut f = FilterState::new();
        f.hide(&id("a"));
        f.hide(&id("a"));
        assert!(f.is_hidden("a"));
        assert_eq!(f.hidden().len(), 1);

        f.show("a");
        f.show("a");
        assert!(!f.is_hidden("a"));
        assert!(f.is_empty());
    }

    #[test]
    fn clear_drops_every_hidden_id() {
        let mut f = FilterState::new();
        f.hide(&id("a"));
        f.hide(&id("b"));
        f.hide(&id("c"));
        assert_eq!(f.hidden().len(), 3);
        f.clear();
        assert!(f.is_empty());
    }

    #[test]
    fn independent_ids_dont_collide() {
        let mut f = FilterState::new();
        f.hide(&id("a"));
        assert!(f.is_hidden("a"));
        assert!(!f.is_hidden("b"));
        f.hide(&id("b"));
        assert!(f.is_hidden("b"));
        f.show("a");
        assert!(!f.is_hidden("a"));
        assert!(f.is_hidden("b"));
    }
}
