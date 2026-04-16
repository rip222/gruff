use std::collections::HashSet;

use eframe::egui;

use crate::colors;
use crate::graph::NodeId;
use crate::search;

use super::GruffApp;

/// State for the Cmd+F fuzzy-search overlay. Kept as a cached match set so
/// the render loop does O(1) `contains` lookups per node instead of
/// re-running the matcher against every node every frame.
pub(super) struct SearchState {
    /// Current text in the search input. Empty right after opening.
    pub(super) input: String,
    /// Set of node ids that match `input` under the package-aware rule.
    /// Recomputed when `input` changes; empty when `input` is empty/whitespace.
    pub(super) matches: HashSet<NodeId>,
    /// Snapshot of the last input we computed matches for, used to skip
    /// recomputation when the user is e.g. holding an arrow key in the field.
    last_computed_input: String,
    /// One-shot flag: request keyboard focus on the text field on the first
    /// frame after opening. Consumed by `draw_search_overlay`.
    request_focus: bool,
}

impl GruffApp {
    /// Toggle the search overlay. If it's already open we close it (Cmd+F
    /// as a dismiss shortcut feels natural next to Escape); otherwise open
    /// a fresh empty search. Selection is intentionally preserved either way.
    pub(super) fn toggle_search(&mut self) {
        if self.search.is_some() {
            self.search = None;
        } else {
            self.search = Some(SearchState {
                input: String::new(),
                matches: HashSet::new(),
                last_computed_input: String::new(),
                request_focus: true,
            });
        }
    }

    /// Recompute the cached match set if the query text has changed since
    /// the last recompute. Cheap no-op when nothing changed — keeps the
    /// render loop fast when the user is idle in the overlay.
    fn refresh_search_matches(&mut self) {
        let Some(search) = self.search.as_mut() else {
            return;
        };
        if search.input == search.last_computed_input {
            return;
        }
        search.matches = search::compute_matches(&search.input, &self.graph.nodes);
        search.last_computed_input = search.input.clone();
    }

    pub(super) fn draw_search_overlay(&mut self, ctx: &egui::Context) {
        let Some(search) = self.search.as_ref() else {
            return;
        };
        let mut input = search.input.clone();
        let request_focus = search.request_focus;
        let match_count = search.matches.len();
        let query_empty = search.input.trim().is_empty();

        let mut close = false;

        // Anchored near top-center of the viewport so the overlay doesn't
        // hide the sidebar or cover the status line. `Area` (rather than
        // `Window`) keeps the chrome minimal — no title bar, no resize.
        egui::Area::new(egui::Id::new("gruff-search-overlay"))
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 16.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(egui::Margin::symmetric(12, 10))
                    .show(ui, |ui| {
                        ui.set_min_width(360.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Search").color(colors::HINT).small());
                            let edit = ui.add(
                                egui::TextEdit::singleline(&mut input)
                                    .desired_width(f32::INFINITY)
                                    .hint_text("file or package name"),
                            );
                            if request_focus {
                                edit.request_focus();
                            }
                        });
                        ui.add_space(4.0);
                        let status = if query_empty {
                            "Type to filter · Esc to close".to_string()
                        } else {
                            format!(
                                "{match_count} match{} · Esc to close",
                                if match_count == 1 { "" } else { "es" },
                            )
                        };
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(status).color(colors::HINT).small());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.small_button("×").clicked() {
                                        close = true;
                                    }
                                },
                            );
                        });
                    });
            });

        if let Some(search) = self.search.as_mut() {
            search.input = input;
            // `request_focus` is a one-shot; clear it after the first frame
            // so subsequent frames don't keep stealing focus from clicks
            // into other widgets.
            search.request_focus = false;
        }
        if close {
            self.search = None;
            return;
        }
        self.refresh_search_matches();
    }
}
