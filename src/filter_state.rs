//! Visibility filter for the dependency graph.
//!
//! Thin adapter over a [`HashSet<NodeId>`] of hidden ids. The set is the
//! canonical "what is off" signal that the canvas, layout simulation, cycle
//! detection, and sidebar all read from. A node is *hidden* when its id is in
//! the set; *visible* otherwise. Edges with any hidden endpoint are treated
//! as hidden too — that derivation happens at the caller, since this module
//! deliberately knows nothing about graph structure.
//!
//! Per PRD #34 the filter is persisted per-workspace under
//! `~/.gruff/workspaces/<hash>.toml`. This module stays graph-agnostic:
//! every mutation fires a [`FilterState::set_persist_callback`] hook so the
//! caller (the app) can pipe the current hide-set into
//! [`crate::workspace_state`] without this module having to know anything
//! about TOML, `$HOME`, or workspace hashing. The default callback is a
//! no-op so constructing a `FilterState` in a unit test never touches disk.
//!
//! Kept as a deep, pure module with no egui, no graph, and no layout
//! dependencies so the invariants can be unit-tested cheaply.

use std::collections::HashSet;

use crate::graph::NodeId;

/// Type alias matching the canvas / layout callers that borrow the hide set
/// read-only. Keeps the persist-callback signature short and makes it
/// obvious the callback shouldn't mutate.
pub type HideSet = HashSet<NodeId>;

/// Mutable set of hidden node ids.
///
/// Invariant: a node is considered hidden iff its id is in [`Self::hidden`].
/// All public mutators route through this struct so call sites don't
/// construct transient "is this node hidden?" state out of band, and so the
/// persist callback fires exactly once per mutation.
pub struct FilterState {
    hidden: HideSet,
    /// Fired after every mutation with a borrow of the post-mutation set.
    /// Default is a no-op closure so `FilterState::new` / `::default` stay
    /// side-effect-free. See [`Self::set_persist_callback`].
    persist: Box<dyn Fn(&HideSet)>,
}

impl std::fmt::Debug for FilterState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `persist` is a trait object and has no useful `Debug` impl; we
        // elide it and just surface the hide-set so `dbg!` output stays
        // readable and tests that snapshot the debug shape keep working.
        f.debug_struct("FilterState")
            .field("hidden", &self.hidden)
            .finish_non_exhaustive()
    }
}

impl Clone for FilterState {
    fn clone(&self) -> Self {
        // Cloning a `FilterState` drops the persist callback on the
        // clone — callers rewiring a clone are expected to re-install
        // whatever side-effect they want. Keeping the callback would
        // require `Fn + Clone`, which `Box<dyn Fn>` doesn't offer and
        // which no caller actually needs today.
        Self {
            hidden: self.hidden.clone(),
            persist: Box::new(noop_persist),
        }
    }
}

impl Default for FilterState {
    fn default() -> Self {
        Self {
            hidden: HashSet::new(),
            persist: Box::new(noop_persist),
        }
    }
}

/// The default callback: do nothing. Named so the `Default` and `Clone` impls
/// can share the same zero-arg constructor without allocating two separate
/// trivial closures in different places.
fn noop_persist(_: &HideSet) {}

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
    pub fn hidden(&self) -> &HideSet {
        &self.hidden
    }

    /// Install a persistence callback fired after every mutation with a
    /// borrow of the post-mutation hide-set. The default callback is a
    /// no-op so unit tests and pre-#34 call sites keep working unchanged.
    /// The app's `load_folder` installs one that pipes through
    /// [`crate::workspace_state::save`].
    ///
    /// Read-only accessors (`is_hidden`, `is_empty`, `hidden`) deliberately
    /// don't fire the callback — disk churn should track actual user intent,
    /// not sidebar re-renders.
    pub fn set_persist_callback<F: Fn(&HideSet) + 'static>(&mut self, callback: F) {
        self.persist = Box::new(callback);
    }

    /// Replace the hide-set in bulk without firing the persist callback.
    /// Used at folder-load to install the previously-persisted set: firing
    /// `persist` there would trigger a disk write with the exact contents
    /// we just read off disk, which is pointless churn.
    ///
    /// Every other mutation path goes through a method that *does* fire the
    /// callback, so this is the only quiet entry and its caller (the app's
    /// `load_folder`) is responsible for not abusing it.
    pub fn install_loaded(&mut self, hidden: HideSet) {
        self.hidden = hidden;
    }

    /// Flip `id`'s membership: hidden → visible, visible → hidden. Returns
    /// the new hidden state so callers can decide whether a repaint /
    /// layout-sync is worth doing (always true for now, but keeping the
    /// return value lets future callers short-circuit).
    pub fn toggle(&mut self, id: &NodeId) -> bool {
        let new_state = if !self.hidden.insert(id.clone()) {
            self.hidden.remove(id);
            false
        } else {
            true
        };
        (self.persist)(&self.hidden);
        new_state
    }

    /// Mark `id` as hidden. No-op if it's already in the set.
    pub fn hide(&mut self, id: &NodeId) {
        self.hidden.insert(id.clone());
        (self.persist)(&self.hidden);
    }

    /// Mark `id` as visible. No-op if it wasn't hidden.
    pub fn show(&mut self, id: &str) {
        self.hidden.remove(id);
        (self.persist)(&self.hidden);
    }

    /// Drop every hidden id. Fires the persist callback so a cleared
    /// filter overwrites any stale on-disk state.
    pub fn clear(&mut self) {
        self.hidden.clear();
        (self.persist)(&self.hidden);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn id(s: &str) -> NodeId {
        s.to_string()
    }

    /// Test helper: build a `FilterState` whose persist callback bumps a
    /// shared counter and records the post-mutation hide-set. The counter
    /// is the primary assertion surface; the recorded set lets tests that
    /// care about the exact payload at callback time spot-check it.
    fn state_with_counter() -> (FilterState, Rc<RefCell<u32>>, Rc<RefCell<Vec<HideSet>>>) {
        let counter = Rc::new(RefCell::new(0u32));
        let snapshots = Rc::new(RefCell::new(Vec::<HideSet>::new()));
        let mut f = FilterState::new();
        let c = counter.clone();
        let s = snapshots.clone();
        f.set_persist_callback(move |hidden| {
            *c.borrow_mut() += 1;
            s.borrow_mut().push(hidden.clone());
        });
        (f, counter, snapshots)
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

    // --- Persist callback (#34) -------------------------------------------

    #[test]
    fn default_persist_callback_is_noop() {
        // Constructing a `FilterState` via `new` / `Default` must not
        // touch disk or produce observable side effects. The counter
        // here is only wired once the test chooses to — until then
        // mutations go through the no-op default.
        let mut f = FilterState::new();
        f.toggle(&id("a"));
        f.hide(&id("b"));
        f.show("b");
        f.clear();
        // Reaching this line without panicking is the guarantee; the
        // assertion is the absence of any failure path.
        assert!(f.is_empty());
    }

    #[test]
    fn toggle_fires_callback_exactly_once() {
        // Toggling is one mutation — exactly one callback fire, no
        // matter whether the toggle added or removed an id. Acceptance
        // criterion from #34's commit 2 spec.
        let (mut f, counter, snapshots) = state_with_counter();
        f.toggle(&id("a"));
        assert_eq!(*counter.borrow(), 1);
        assert!(snapshots.borrow()[0].contains("a"));

        f.toggle(&id("a"));
        assert_eq!(*counter.borrow(), 2);
        assert!(snapshots.borrow()[1].is_empty());
    }

    #[test]
    fn hide_and_show_each_fire_callback_once() {
        // Two mutations → two fires. Idempotent hide (already in set)
        // still fires because the caller can't tell from outside, and
        // the cost of one redundant TOML write after a no-op click is
        // negligible compared to introducing a "did this change?" gate
        // that the sidebar would have to replicate.
        let (mut f, counter, _snapshots) = state_with_counter();
        f.hide(&id("a"));
        f.show("a");
        assert_eq!(*counter.borrow(), 2);
    }

    #[test]
    fn clear_fires_callback_once() {
        // A bulk clear is one observable mutation, not one per id — the
        // callback fires once so the persisted file gets a single empty
        // write rather than N successive shrinking writes.
        let (mut f, counter, snapshots) = state_with_counter();
        f.hide(&id("a"));
        f.hide(&id("b"));
        f.hide(&id("c"));
        assert_eq!(*counter.borrow(), 3, "precondition: three hides = three fires");
        f.clear();
        assert_eq!(*counter.borrow(), 4, "clear must add exactly one more fire");
        let last = snapshots.borrow().last().cloned().unwrap();
        assert!(last.is_empty(), "callback must see the post-clear (empty) set");
    }

    #[test]
    fn read_only_ops_never_fire_callback() {
        // `is_hidden` / `is_empty` / `hidden()` on a populated filter
        // must not trigger persistence — sidebar rendering polls these
        // per frame and can't afford to churn disk.
        let (mut f, counter, _snapshots) = state_with_counter();
        f.hide(&id("a"));
        let before = *counter.borrow();
        let _ = f.is_hidden("a");
        let _ = f.is_empty();
        let _ = f.hidden();
        assert_eq!(
            *counter.borrow(),
            before,
            "read-only accessors must not fire the persist callback",
        );
    }

    #[test]
    fn install_loaded_does_not_fire_callback() {
        // `install_loaded` is the load-from-disk path; firing the
        // callback there would write back the exact contents we just
        // read. Assert the quiet install so a future refactor doesn't
        // silently add a write-amplification loop.
        let (mut f, counter, _snapshots) = state_with_counter();
        let mut loaded = HashSet::new();
        loaded.insert(id("x"));
        loaded.insert(id("y"));
        f.install_loaded(loaded);
        assert_eq!(*counter.borrow(), 0, "install_loaded must not fire persist");
        assert!(f.is_hidden("x") && f.is_hidden("y"));
    }

    #[test]
    fn callback_sees_post_mutation_state() {
        // The callback's borrow is the set *after* the mutation, not
        // before. This matters because the app persists whatever the
        // callback sees, and persisting pre-mutation state would be
        // stale by one click.
        let (mut f, _counter, snapshots) = state_with_counter();
        f.hide(&id("a"));
        let seen = snapshots.borrow().last().cloned().unwrap();
        assert!(seen.contains("a"), "callback must see `a` in the post-hide set");
    }
}
