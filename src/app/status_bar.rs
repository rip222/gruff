use egui::RichText;

use crate::colors;
use crate::reachability;

use super::GruffApp;

pub(super) const LARGE_GRAPH_WARNING_THRESHOLD: usize = 20_000;

impl GruffApp {
    pub(super) fn draw_status_bar(&mut self, ui: &mut egui::Ui) {
        let root = self.last_root.as_deref();
        let errors = self.current_status_errors();
        let latest_error = errors.last();
        let summary = if self.status.is_empty() {
            "Ready"
        } else {
            &self.status
        };
        // Blast-radius readout takes priority over the index summary when a
        // selection is live — the index counts are already surfaced in the
        // sidebar and the "N files in M packages depend on this" string is
        // the whole point of clicking a node per PRD #35.
        let blast_summary = self.blast_radius_summary();
        let left_text = blast_summary.as_deref().unwrap_or(summary);

        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.label(RichText::new(left_text).small().color(colors::HINT));

            if self.unresolved_dynamic > 0 {
                let label = format!(
                    " {} unresolved dynamic import{} ",
                    self.unresolved_dynamic,
                    if self.unresolved_dynamic == 1 {
                        ""
                    } else {
                        "s"
                    },
                );
                ui.label(
                    RichText::new(label)
                        .small()
                        .color(colors::BG)
                        .background_color(colors::HINT),
                );
            }

            if has_large_graph_warning(self.graph.nodes.len()) {
                ui.label(
                    RichText::new(" 20k+ nodes ")
                        .small()
                        .color(colors::BG)
                        .background_color(colors::SELECTED),
                );
            }

            if errors.len() > 1 {
                let label = format!(" {} issues ", errors.len());
                ui.label(
                    RichText::new(label)
                        .small()
                        .color(colors::BG)
                        .background_color(colors::CYCLE_EDGE),
                );
            }

            if let Some(error) = latest_error {
                ui.separator();
                ui.label(
                    RichText::new(error.short_message(root))
                        .small()
                        .color(colors::BG)
                        .background_color(colors::CYCLE_EDGE),
                );
            }
        });
    }
}

pub(super) fn has_large_graph_warning(node_count: usize) -> bool {
    node_count > LARGE_GRAPH_WARNING_THRESHOLD
}

impl GruffApp {
    /// Build the "N files in M packages depend on this." string the status
    /// bar renders while a selection is live. Returns `None` when there's
    /// nothing to report (no selection, or no cached cone) so the caller
    /// falls back to the regular index summary.
    ///
    /// The format is fixed per PRD #35's decision doc — no pluralisation,
    /// no i18n — so changing the copy here is a single-point edit.
    pub(super) fn blast_radius_summary(&self) -> Option<String> {
        let cone = self.blast_cone.as_ref()?;
        let (files, packages) = reachability::cone_stats(&self.graph, cone);
        Some(format!(
            "{files} files in {packages} packages depend on this."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_graph_warning_triggers_above_threshold() {
        assert!(!has_large_graph_warning(20_000));
        assert!(has_large_graph_warning(20_001));
    }
}
