//! Keyboard-shortcut cheat-sheet modal.
//!
//! Follows the same pattern as `editor_prompt.rs` — an `egui::Area` anchored
//! center, wrapped in `Frame::popup`, dismissed on Escape or click-outside.
//! The content is rendered straight from `crate::shortcuts::SHORTCUTS`
//! grouped by `ShortcutGroup`, so the modal and the dispatcher stay in sync
//! without any drift: adding a new shortcut to the registry shows up here
//! for free.
//!
//! Toggling the modal is a separate concern — commit 4 of PRD #33 wires the
//! `?` keybind and menu-bar button. This module is trigger-agnostic: flip
//! `self.show_help` and the renderer paints on the next frame.

use eframe::egui;

use crate::colors;
use crate::shortcuts::{SHORTCUTS, ShortcutGroup};

use super::GruffApp;

impl GruffApp {
    /// Render the help modal when `self.show_help` is set. Called last in
    /// the paint order from `ui()` so the popup sits on top of the sidebar,
    /// canvas, and the other overlays.
    ///
    /// Dismissal:
    /// - `Escape` closes it (handled here rather than through the shortcut
    ///   registry because the main dispatcher's `Dismiss` action also
    ///   closes other overlays, and we want "help wins" priority when help
    ///   is visible).
    /// - Clicking outside the popup closes it — egui's `Area` doesn't
    ///   consume pointer events outside its bounds, so we detect a press
    ///   whose position lies outside the response rect and treat that as
    ///   click-outside.
    pub(super) fn draw_help_modal(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }

        let mut close = false;

        // Pull Escape off the input queue *before* the main `Dismiss`
        // handler sees it so closing the help modal doesn't also clear
        // selection. Using `consume_key` yields true once per press and
        // removes it from the event stream.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            close = true;
        }

        let area = egui::Area::new(egui::Id::new("gruff-help-modal"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(egui::Margin::symmetric(16, 14))
                    .show(ui, |ui| {
                        ui.set_min_width(360.0);
                        ui.set_max_width(480.0);
                        ui.heading("Keyboard shortcuts");
                        ui.add_space(8.0);

                        for (idx, group) in ShortcutGroup::ORDER.iter().enumerate() {
                            if idx > 0 {
                                ui.add_space(10.0);
                            }
                            ui.label(
                                egui::RichText::new(group.header())
                                    .color(colors::HINT)
                                    .small(),
                            );
                            ui.add_space(2.0);
                            for shortcut in SHORTCUTS.iter().filter(|s| s.group == *group) {
                                render_entry(ui, shortcut.keys, shortcut.label);
                            }
                        }

                        ui.add_space(12.0);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("Esc or click outside to close")
                                    .color(colors::HINT)
                                    .small(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.small_button("Close").clicked() {
                                        close = true;
                                    }
                                },
                            );
                        });
                    });
            });

        // Click-outside dismisses. A pointer-down anywhere off the popup's
        // rect counts — we check `any_click` against the area response so
        // clicks on scrollbars / menu items inside still work normally.
        let clicked_outside = ctx.input(|i| {
            i.pointer.any_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area.response.rect.contains(p))
        });
        if clicked_outside {
            close = true;
        }

        if close {
            self.show_help = false;
        }
    }
}

/// One row in the modal: keys on the left, description on the right. Kept
/// as a free function rather than a method so the tests can eyeball the
/// rendered-line structure without standing up a `GruffApp`.
fn render_entry(ui: &mut egui::Ui, keys: &str, label: &str) {
    ui.horizontal(|ui| {
        // Fixed-width keys column so labels line up across rows with
        // different key-string lengths. Monospace makes `Cmd+F` and `Space`
        // look like key chords rather than prose.
        ui.add_sized(
            [120.0, 18.0],
            egui::Label::new(egui::RichText::new(keys).monospace()),
        );
        ui.label(label);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shortcuts::SHORTCUTS;

    #[test]
    fn renderer_paints_every_registry_entry_when_open() {
        // Dev-only exercise of the paint path: drive a headless egui context,
        // flip `show_help` on, and run one frame through the modal. We don't
        // assert on pixels — we assert the renderer didn't panic and that the
        // toggle round-trips. Also verifies the registry isn't silently
        // empty for the modal's section loop.
        let ctx = egui::Context::default();
        let mut app = GruffApp {
            show_help: true,
            ..GruffApp::default()
        };

        let _ = ctx.run(Default::default(), |ctx| {
            app.draw_help_modal(ctx);
        });

        // Still open: without an Escape press or outside click, the modal
        // stays visible across frames.
        assert!(
            app.show_help,
            "modal must stay open across frames until dismissed"
        );

        // Every group contributes at least one entry the renderer would
        // paint — mirrors the registry invariant but checked from the
        // modal's perspective so a future refactor that changes how the
        // modal iterates still has a test to catch a blank section.
        for group in ShortcutGroup::ORDER {
            assert!(
                SHORTCUTS.iter().any(|s| s.group == *group),
                "modal would render an empty section for {:?}",
                group
            );
        }
    }

    #[test]
    fn renderer_is_noop_when_closed() {
        // When `show_help` is false the modal must return immediately —
        // no paint, no area. Exercised here because the main `ui()` calls
        // this every frame regardless of state.
        let ctx = egui::Context::default();
        let mut app = GruffApp::default();
        assert!(!app.show_help);

        let _ = ctx.run(Default::default(), |ctx| {
            app.draw_help_modal(ctx);
        });

        assert!(!app.show_help, "closed modal must stay closed");
    }
}
