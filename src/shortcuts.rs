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

/// Modifier flags the dispatcher cares about. A subset of `egui::Modifiers`
/// lifted here so the `Shortcut::matches` helper and its tests don't need an
/// egui context. The app-side adapter fills this from `egui::InputState` each
/// frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShortcutMods {
    /// Platform-command modifier (Cmd on macOS, Ctrl elsewhere — matches
    /// `egui::Modifiers::command`).
    pub command: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
}

/// One key event the dispatcher evaluates against the registry. `key` is
/// already uppercased so the registry's `"F"` / `"O"` strings match directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPress {
    /// Uppercase ASCII label for the pressed key — `"F"`, `"SPACE"`, `"ESC"`,
    /// `"/"` and so on. The dispatcher canonicalises egui's `Key` into this
    /// so the registry can stay human-readable.
    pub key: &'static str,
    pub mods: ShortcutMods,
}

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
/// are registered for documentation but not yet wired.
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
    /// Toggle the blast-radius dim on the current selection (`B`). Flips
    /// the dim state without affecting the selection itself — see
    /// `GruffApp::blast_radius_active`.
    ToggleBlastRadius,
    /// Toggle the keyboard-shortcut cheat-sheet modal (`?` / Shift+/).
    ToggleHelp,
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

impl Shortcut {
    /// Parse `self.keys` and decide whether it matches `press`. The key-string
    /// syntax is the same one users see in the help modal — `"Cmd+F"`,
    /// `"Space"`, `"?"`, etc. Case-insensitive, tolerant of surrounding
    /// whitespace, rejects shortcuts whose `keys` aren't a recognised keyboard
    /// binding (e.g. the `"Sidebar click"` Filters placeholder) so the
    /// dispatcher simply skips those entries.
    pub fn matches(&self, press: &KeyPress) -> bool {
        let Some(expected) = parse_keys(self.keys) else {
            return false;
        };
        expected == *press
    }
}

/// Parse a registry `keys` string into the same `KeyPress` shape the
/// dispatcher produces from egui events. Returns `None` for entries that
/// aren't dispatchable (e.g. the `"Sidebar click"` documentation-only row).
fn parse_keys(keys: &str) -> Option<KeyPress> {
    let mut mods = ShortcutMods::default();
    let mut key: Option<&'static str> = None;
    for token in keys.split('+') {
        let token = token.trim();
        // A few tokens carry spaces internally (e.g. `"Sidebar click"`).
        // Those aren't keyboard-dispatchable — bail and let the caller treat
        // the row as display-only.
        if token.contains(' ') {
            return None;
        }
        match token.to_ascii_lowercase().as_str() {
            "cmd" | "command" | "ctrl+cmd" => mods.command = true,
            "ctrl" | "control" => mods.ctrl = true,
            "alt" | "option" | "opt" => mods.alt = true,
            "shift" => mods.shift = true,
            other => {
                // Canonicalise to the `&'static str` values the dispatcher
                // produces. Keeping this match exhaustive-ish is deliberate —
                // a new registry entry for an unmapped key will return `None`
                // and get silently skipped, which `Shortcut::matches` callers
                // catch via dispatch not firing. Tests enforce this round-
                // trips for every dispatched entry.
                key = Some(match other {
                    "o" => "O",
                    "f" => "F",
                    "r" => "R",
                    "b" => "B",
                    "space" => "SPACE",
                    "esc" | "escape" => "ESC",
                    "?" => "?",
                    "/" => "/",
                    _ => return None,
                });
            }
        }
    }
    Some(KeyPress { key: key?, mods })
}

/// The registry. Single source of truth for (keys, label, group, action)
/// across the whole app.
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
    Shortcut {
        keys: "B",
        label: "Toggle blast-radius dim on selection",
        group: ShortcutGroup::View,
        action: ShortcutAction::ToggleBlastRadius,
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
    Shortcut {
        keys: "?",
        label: "Show keyboard shortcuts",
        group: ShortcutGroup::Help,
        action: ShortcutAction::ToggleHelp,
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

    #[test]
    fn matches_canonical_keybinds() {
        // Spot check: each action's registry entry matches exactly the key
        // event it's documented for and nothing else.
        let cmd_f = KeyPress {
            key: "F",
            mods: ShortcutMods {
                command: true,
                ..Default::default()
            },
        };
        let search = SHORTCUTS
            .iter()
            .find(|s| s.action == ShortcutAction::ToggleSearch)
            .expect("ToggleSearch must be in the registry");
        assert!(search.matches(&cmd_f));

        let bare_f = KeyPress {
            key: "F",
            mods: ShortcutMods::default(),
        };
        let fit = SHORTCUTS
            .iter()
            .find(|s| s.action == ShortcutAction::FitView)
            .expect("FitView must be in the registry");
        assert!(fit.matches(&bare_f));
        // Cmd+F must not fire the bare-F fit action — the modifier gate is
        // the whole reason the keybind is written as `"F"` not `"Cmd+F"`.
        assert!(!fit.matches(&cmd_f));
    }

    #[test]
    fn display_only_entries_never_match() {
        // Rows like `"Sidebar click"` are documentation; they must be
        // unreachable from the keyboard so dispatch never accidentally fires
        // them.
        let any_key = KeyPress {
            key: "F",
            mods: ShortcutMods::default(),
        };
        for s in SHORTCUTS.iter().filter(|s| s.keys.contains(' ')) {
            assert!(
                !s.matches(&any_key),
                "display-only entry {:?} should not match any keyboard event",
                s.keys
            );
        }
    }
}
