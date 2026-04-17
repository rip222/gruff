//! Most-recently-used list of opened folders.
//!
//! Pure data structure: zero egui, zero filesystem. Callers decide when to
//! push, when to prune (with an injected liveness predicate), and when to
//! persist. The only policy baked in here is the cap and the move-to-top
//! dedupe on push — the two invariants an MRU list has to enforce itself.
//!
//! Path canonicalization is deliberately *not* done here; the CLI and picker
//! flows normalize paths upstream so this list stores exactly what it's given.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Cap on the number of entries retained. Matches the decision in issue #32
/// — five is enough to cover a developer juggling a handful of active repos
/// without turning the submenu into a scrolling list.
pub const MAX_RECENTS: usize = 5;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct RecentList {
    /// Most-recent entry first. Length bounded by [`MAX_RECENTS`].
    #[serde(default)]
    paths: Vec<PathBuf>,
}

impl RecentList {
    /// Empty list — matches `Default`, provided explicitly for call sites
    /// that want to be obvious about constructing an empty MRU.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push `path` to the top of the list. If the path is already present
    /// it's moved to the front (dedupe by path equality) rather than
    /// duplicated. The list is truncated to [`MAX_RECENTS`] so the oldest
    /// entry is evicted when the cap is exceeded.
    pub fn push(&mut self, path: PathBuf) {
        // Move-to-top: if we've seen this path before, yank it out so the
        // insert below puts it at position 0. `PathBuf` equality is byte-
        // wise, which matches what the callers feed us (canonicalized paths
        // upstream, so we don't have to worry about `./foo` vs `foo`).
        self.paths.retain(|p| p != &path);
        self.paths.insert(0, path);
        if self.paths.len() > MAX_RECENTS {
            self.paths.truncate(MAX_RECENTS);
        }
    }

    /// Drop entries for which `is_live` returns false, preserving the order
    /// of survivors. Bounded by the cap (≤ 5 calls to the predicate), so
    /// calling at render time is cheap. The predicate is injected rather
    /// than hardcoded to `fs::metadata` so unit tests stay hermetic.
    pub fn prune_stale<F: Fn(&Path) -> bool>(&mut self, is_live: F) {
        self.paths.retain(|p| is_live(p));
    }

    /// Wipe the list. Backs the "Clear Recent" menu action — immediate with
    /// no confirm dialog per the decision doc.
    pub fn clear(&mut self) {
        self.paths.clear();
    }

    /// True if the list has no entries. Used by the menu renderer to grey
    /// out the submenu when there's nothing to show.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Number of entries currently stored. Exposed for tests and assertions;
    /// the renderer walks [`Self::iter_excluding`] directly.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Iterate over entries in MRU order, skipping `current` when it matches.
    /// The currently-loaded folder is filtered (not removed) so the stored
    /// list keeps its MRU ordering across folder switches — reopening a
    /// previous repo still finds it at the top of the list.
    pub fn iter_excluding<'a>(
        &'a self,
        current: Option<&'a Path>,
    ) -> impl Iterator<Item = &'a Path> + 'a {
        self.paths
            .iter()
            .map(PathBuf::as_path)
            .filter(move |p| current.is_none_or(|c| *p != c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn push_new_prepends() {
        // A never-seen path lands at the top and the previous top slides
        // down one slot. Nothing else shifts — MRU ordering in miniature.
        let mut list = RecentList::new();
        list.push(p("/a"));
        list.push(p("/b"));
        list.push(p("/c"));
        let paths: Vec<&Path> = list.iter_excluding(None).collect();
        assert_eq!(paths, vec![p("/c").as_path(), p("/b").as_path(), p("/a").as_path()]);
    }

    #[test]
    fn push_existing_moves_to_top() {
        // Reopening an existing repo yanks it from wherever it was and
        // slots it at position 0 — no duplicates, no gaps.
        let mut list = RecentList::new();
        list.push(p("/a"));
        list.push(p("/b"));
        list.push(p("/c"));
        list.push(p("/a")); // bump
        let paths: Vec<&Path> = list.iter_excluding(None).collect();
        assert_eq!(paths, vec![p("/a").as_path(), p("/c").as_path(), p("/b").as_path()]);
        assert_eq!(list.len(), 3, "no duplicate entry for /a");
    }

    #[test]
    fn push_exceeding_cap_evicts_oldest() {
        // The cap is a hard ceiling: the 6th distinct push drops the
        // oldest entry rather than extending the list.
        let mut list = RecentList::new();
        for i in 0..7 {
            list.push(p(&format!("/r{i}")));
        }
        assert_eq!(list.len(), MAX_RECENTS);
        let paths: Vec<&Path> = list.iter_excluding(None).collect();
        // Newest first — /r6 at the top, /r2 at the bottom; /r0 and /r1
        // have been evicted.
        assert_eq!(
            paths,
            vec![
                p("/r6").as_path(),
                p("/r5").as_path(),
                p("/r4").as_path(),
                p("/r3").as_path(),
                p("/r2").as_path(),
            ]
        );
    }

    #[test]
    fn prune_removes_failing_entries_preserving_order() {
        // The predicate is injected so the test doesn't touch the
        // filesystem. Survivors stay in their original MRU order.
        let mut list = RecentList::new();
        list.push(p("/alive-1"));
        list.push(p("/dead-1"));
        list.push(p("/alive-2"));
        list.push(p("/dead-2"));
        list.prune_stale(|p| !p.to_string_lossy().contains("dead"));
        let paths: Vec<&Path> = list.iter_excluding(None).collect();
        // Order reflects push order (newest first): /alive-2 then /alive-1.
        assert_eq!(paths, vec![p("/alive-2").as_path(), p("/alive-1").as_path()]);
    }

    #[test]
    fn clear_empties() {
        let mut list = RecentList::new();
        list.push(p("/a"));
        list.push(p("/b"));
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn iter_excluding_filters_current() {
        // The submenu hides the currently-loaded folder so it isn't a
        // pointless "open this again" entry. Filtering (not removal) keeps
        // the underlying list intact for the next folder switch.
        let mut list = RecentList::new();
        list.push(p("/a"));
        list.push(p("/b"));
        list.push(p("/c"));
        let current = p("/b");
        let paths: Vec<&Path> = list.iter_excluding(Some(&current)).collect();
        assert_eq!(paths, vec![p("/c").as_path(), p("/a").as_path()]);
        // None means "no current folder" — everything shows.
        let all: Vec<&Path> = list.iter_excluding(None).collect();
        assert_eq!(all.len(), 3);
    }
}
