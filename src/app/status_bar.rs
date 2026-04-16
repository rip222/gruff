use egui::RichText;

use crate::colors;

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

        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.label(RichText::new(summary).small().color(colors::HINT));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_graph_warning_triggers_above_threshold() {
        assert!(!has_large_graph_warning(20_000));
        assert!(has_large_graph_warning(20_001));
    }
}
