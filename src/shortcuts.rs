//! Single source of truth for every keyboard shortcut the app handles.
//!
//! The registry is a flat `&'static [Shortcut]` — one table, two consumers:
//! the keybinding handlers in `app.rs` iterate it to dispatch, and the help
//! modal in `app/help_modal.rs` renders it grouped for the user. New
//! shortcuts get added here and picked up by both paths automatically, which
//! kills the old drift between code and `SPEC.md`.
//!
//! Shape is deliberately flat: linear scan over a handful of entries is free
//! next to frame work, and a `HashMap` would make the modal's grouped render
//! awkward. The `action` field is an enum so dispatch stays typed — see
//! `ShortcutAction` for the full set.

/// Display categories for the help modal. Kept closed rather than string-
/// based so a stray new group is caught at compile time instead of silently
/// rendering under a typoed header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortcutGroup {
    Navigation,
    View,
    Filters,
    Help,
}

impl ShortcutGroup {
    /// Order groups appear in the help modal. Navigation first (the most
    /// frequently-used category), Help last (self-referential).
    pub const ORDER: &'static [ShortcutGroup] = &[
        ShortcutGroup::Navigation,
        ShortcutGroup::View,
        ShortcutGroup::Filters,
        ShortcutGroup::Help,
    ];

    /// Human-readable header used in the help modal.
    pub fn header(self) -> &'static str {
        match self {
            ShortcutGroup::Navigation => "Navigation",
            ShortcutGroup::View => "View",
            ShortcutGroup::Filters => "Filters",
            ShortcutGroup::Help => "Help",
        }
    }
}

/// Dispatch target for a shortcut. Commit 2 wires the registry-driven
/// dispatcher to match on these; `Noop` is a placeholder for entries that
/// are registered for documentation but not yet wired (the `B` toggle comes
/// in issue #35; the `?` toggle gets its real variant in commit 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortcutAction {
    /// Open a folder via picker (Cmd+O).
    OpenFolder,
    /// Toggle the fuzzy-search overlay (Cmd+F).
    ToggleSearch,
    /// Re-scan the currently-loaded folder (Cmd+R).
    Rescan,
    /// Pause / resume the force-layout physics simulation (Space).
    TogglePhysics,
    /// Refit the viewport to the visible graph (F).
    FitView,
    /// Dismiss whichever overlay is open, else clear selection (Escape).
    Dismiss,
    /// Reserved entry — dispatch is handled elsewhere or not yet wired.
    Noop,
}

/// One row in the registry. Fields are `&'static str` so the whole table
/// lives in static memory — no allocation at startup, cheap to iterate each
/// frame.
#[derive(Debug, Clone, Copy)]
pub struct Shortcut {
    /// Human-readable key expression, e.g. `"Cmd+F"` or `"Space"`. Used both
    /// by the help modal (rendered verbatim) and by the dispatcher (matched
    /// against the incoming key event).
    pub keys: &'static str,
    /// One-line description shown next to the keys in the help modal.
    pub label: &'static str,
    pub group: ShortcutGroup,
    pub action: ShortcutAction,
}

/// The registry. Single source of truth for (keys, label, group, action)
/// across the whole app.
///
/// `B` (View) and `?` (Help) are placeholders per PRD #33 — `B`'s handler
/// lands in issue #35, `?`'s in commit 4 of this PRD.
pub const SHORTCUTS: &[Shortcut] = &[
    Shortcut {
        keys: "Cmd+O",
        label: "Open folder",
        group: ShortcutGroup::Navigation,
        action: ShortcutAction::OpenFolder,
    },
    Shortcut {
        keys: "Cmd+F",
        label: "Toggle fuzzy search",
        group: ShortcutGroup::Navigation,
        action: ShortcutAction::ToggleSearch,
    },
    Shortcut {
        keys: "Cmd+R",
        label: "Re-scan current folder",
        group: ShortcutGroup::Navigation,
        action: ShortcutAction::Rescan,
    },
    Shortcut {
        keys: "Esc",
        label: "Dismiss overlay or clear selection",
        group: ShortcutGroup::Navigation,
        action: ShortcutAction::Dismiss,
    },
    Shortcut {
        keys: "F",
        label: "Fit graph to viewport",
        group: ShortcutGroup::View,
        action: ShortcutAction::FitView,
    },
    Shortcut {
        keys: "Space",
        label: "Pause or resume physics",
        group: ShortcutGroup::View,
        action: ShortcutAction::TogglePhysics,
    },
    // Placeholder — issue #35 wires this to toggle barrel collapse.
    Shortcut {
        keys: "B",
        label: "Toggle barrel collapse",
        group: ShortcutGroup::View,
        action: ShortcutAction::Noop,
    },
    // No keyboard shortcut today — the file-visibility toggles live in the
    // sidebar as per-file checkboxes. Documented here so the help modal has
    // a Filters section that points the user at the right affordance rather
    // than looking broken (empty section). Not dispatched.
    Shortcut {
        keys: "Sidebar click",
        label: "Toggle file visibility (per file in sidebar)",
        group: ShortcutGroup::Filters,
        action: ShortcutAction::Noop,
    },
    // Placeholder — commit 4 of this PRD wires this to toggle the help modal.
    Shortcut {
        keys: "?",
        label: "Show keyboard shortcuts",
        group: ShortcutGroup::Help,
        action: ShortcutAction::Noop,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_group_has_at_least_one_entry() {
        // The help modal renders one section per group and an empty section
        // looks like a rendering bug — enforce every group surface something.
        for group in ShortcutGroup::ORDER {
            assert!(
                SHORTCUTS.iter().any(|s| s.group == *group),
                "group {:?} has no shortcuts registered",
                group
            );
        }
    }

    #[test]
    fn keys_are_unique_across_table() {
        // Two entries sharing a `keys` string would make dispatch ambiguous
        // (which action fires for Cmd+F?). Enforce uniqueness so the registry
        // can be scanned linearly without tiebreaker rules.
        let mut seen = HashSet::new();
        for s in SHORTCUTS {
            assert!(
                seen.insert(s.keys),
                "duplicate `keys` entry in registry: {}",
                s.keys
            );
        }
    }
}
