use eframe::egui;

use crate::colors;
use crate::graph::{NodeId, NodeKind};

use super::GruffApp;

impl GruffApp {
    pub(super) fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        if self.selected.is_some() {
            self.draw_selection_pane(ui);
            ui.add_space(12.0);
            ui.separator();
        }
        self.draw_cycles_pane(ui);
    }

    fn draw_selection_pane(&mut self, ui: &mut egui::Ui) {
        let Some(selected) = self.selected.clone() else {
            return;
        };
        // Snapshot owned copies of the fields we need so `self` stays free for
        // `&mut` calls (editor prompt, frame requests) below.
        let Some((display_id, display_label, on_disk_path, package, kind)) = self
            .graph
            .nodes
            .get(&selected)
            .map(|n| {
                (
                    n.id.clone(),
                    n.label.clone(),
                    n.path.clone(),
                    n.package.clone(),
                    n.kind,
                )
            })
        else {
            // Selection stale (node removed) — clear it silently next frame.
            self.selected = None;
            return;
        };

        ui.add_space(6.0);
        match kind {
            NodeKind::File => ui.heading("Selected file"),
            NodeKind::External => ui.heading("External dependency"),
            NodeKind::WorkspacePackage => ui.heading("Workspace package"),
        };
        ui.add_space(4.0);

        if matches!(kind, NodeKind::External | NodeKind::WorkspacePackage) {
            // Synthetic nodes have no on-disk file — render just the package
            // name. Skip the "open in editor" affordance so we don't launch
            // an editor on a non-existent path.
            ui.label(egui::RichText::new("Package").color(colors::HINT).small());
            ui.label(egui::RichText::new(&display_label).monospace());
        } else {
            ui.label(egui::RichText::new("Path").color(colors::HINT).small());
            // Clickable path — launches the user's configured editor. The actual
            // path on disk is what we hand the editor, but we show the workspace-
            // relative node id in the UI so it stays short.
            let link = ui
                .add(
                    egui::Label::new(egui::RichText::new(&display_id).monospace().underline())
                        .sense(egui::Sense::click()),
                )
                .on_hover_text("Click to open in your editor");
            if link.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            if link.clicked() {
                self.try_open_in_editor(on_disk_path);
            }
        }

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Owning package")
                .color(colors::HINT)
                .small(),
        );
        match package.as_deref() {
            Some(name) => {
                let swatch = self.node_color(&selected);
                ui.horizontal(|ui| {
                    // Small color chip so the sidebar identity matches the
                    // node's color on the canvas at a glance.
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(10.0, 10.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(rect, 2.0, swatch);
                    ui.label(egui::RichText::new(name).monospace());
                });
            }
            None => {
                ui.label(egui::RichText::new("(no owning package)").italics());
            }
        }

        ui.add_space(8.0);
        let imports = self.imports.get(&selected).cloned().unwrap_or_default();
        let imported_by = self
            .imported_by
            .get(&selected)
            .cloned()
            .unwrap_or_default();

        render_list(ui, "Imports", &imports);
        ui.add_space(4.0);
        render_list(ui, "Imported by", &imported_by);

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Cycles participated in")
                .color(colors::HINT)
                .small(),
        );
        match self.cycle_of.get(&selected).copied() {
            Some(idx) => {
                // The selected file participates in exactly one SCC — render
                // it as a button that frames the whole cycle in the viewport,
                // same affordance as the "Cycles" section below.
                let size = self.cycles[idx].len();
                let label = format!("Cycle {}  ·  {size} files", idx + 1);
                if ui.button(label).clicked() {
                    self.frame_request = Some(idx);
                }
            }
            None => {
                ui.label(egui::RichText::new("(none)").italics().color(colors::HINT));
            }
        }
    }

    fn draw_cycles_pane(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.heading(format!("Cycles ({})", self.cycles.len()));
        ui.add_space(4.0);

        if self.cycles.is_empty() {
            ui.label(
                egui::RichText::new("No circular dependencies detected.")
                    .italics()
                    .color(colors::HINT),
            );
            return;
        }

        // Scroll-contain the cycle list so a repo with many cycles doesn't
        // push the selection pane off-screen.
        egui::ScrollArea::vertical()
            .id_salt("cycles-list")
            .max_height(260.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (idx, cycle) in self.cycles.iter().enumerate() {
                    let label = format!("Cycle {}  ·  {} files", idx + 1, cycle.len());
                    // Button is full-width so the whole row is the click target.
                    let resp = ui.add(egui::Button::new(label).min_size(egui::vec2(
                        ui.available_width(),
                        0.0,
                    )));
                    if resp.clicked() {
                        self.frame_request = Some(idx);
                    }
                    // Small preview of member files under each cycle, capped
                    // to keep long cycles from dominating the sidebar.
                    egui::CollapsingHeader::new("files")
                        .id_salt(idx)
                        .default_open(false)
                        .show(ui, |ui| {
                            for node_id in cycle {
                                ui.label(
                                    egui::RichText::new(node_id).monospace().small(),
                                );
                            }
                        });
                    ui.add_space(2.0);
                }
            });
    }
}

fn render_list(ui: &mut egui::Ui, title: &str, items: &[NodeId]) {
    let header = format!("{} ({})", title, items.len());
    egui::CollapsingHeader::new(header)
        .default_open(false)
        .show(ui, |ui| {
            if items.is_empty() {
                ui.label(egui::RichText::new("(none)").italics().color(colors::HINT));
                return;
            }
            // Cap the rendered height so huge lists don't blow out the sidebar.
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for item in items {
                        ui.label(egui::RichText::new(item).monospace().small());
                    }
                });
        });
}
