use eframe::egui;

use crate::colors;
use crate::graph::{NodeId, NodeKind};
use crate::package_tree::{
    CheckState, EXTERNALS_LABEL, ExternalsBucket, FolderChild, FolderNode, PackageNode,
    PackageTree, UNPACKAGED_LABEL, UnpackagedBucket,
};

use super::GruffApp;

impl GruffApp {
    pub(super) fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        if self.selected.is_some() {
            self.draw_selection_pane(ui);
            ui.add_space(12.0);
            ui.separator();
        }
        // Packages pane sits between Selection and Cycles per #21. Only
        // rendered once we have something to show — otherwise an empty
        // header still reads as clutter.
        if !self.graph.nodes.is_empty() {
            self.draw_packages_pane(ui);
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
        let Some((display_id, display_label, on_disk_path, package, kind)) =
            self.graph.nodes.get(&selected).map(|n| {
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
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
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
        let imported_by = self.imported_by.get(&selected).cloned().unwrap_or_default();

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

    fn draw_packages_pane(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.heading("Packages");
        ui.add_space(4.0);

        // Rebuild on every frame from the live graph + package color map +
        // current filter. The tree is cheap (one pass over nodes) and
        // staying stateless keeps the renderer in sync with incremental
        // indexer diffs and toggle-driven state changes without an explicit
        // invalidation hook.
        let tree = PackageTree::build(
            &self.graph,
            &self.package_indices,
            self.filter_state.hidden(),
        );
        if tree.is_empty() {
            ui.label(
                egui::RichText::new("(no packages)")
                    .italics()
                    .color(colors::HINT),
            );
            return;
        }

        // Collect file-checkbox toggles emitted by the renderer this frame.
        // Applied after the ScrollArea closes so the filter-change flow
        // (layout resync, cycle recompute, camera tween) runs with a fully
        // consistent view of user intent rather than mid-render.
        let mut toggles: Vec<NodeId> = Vec::new();

        // Cap the pane's rendered height so a huge package list doesn't
        // push Cycles off-screen — matches the ScrollArea policy used for
        // the Cycles pane.
        egui::ScrollArea::vertical()
            .id_salt("packages-list")
            .max_height(360.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for pkg in &tree.packages {
                    draw_package_node(ui, pkg, &mut toggles);
                }
                if !tree.unpackaged.files.is_empty() {
                    draw_unpackaged_bucket(ui, &tree.unpackaged, &mut toggles);
                }
                if !tree.externals.externals.is_empty() {
                    draw_externals_bucket(ui, &tree.externals, &mut toggles);
                }
            });

        self.toggle_file_visibility(&toggles);
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
                    let resp = ui.add(
                        egui::Button::new(label).min_size(egui::vec2(ui.available_width(), 0.0)),
                    );
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
                                ui.label(egui::RichText::new(node_id).monospace().small());
                            }
                        });
                    ui.add_space(2.0);
                }
            });
    }
}

/// Render a single workspace-package subtree. The header row combines the
/// collapsing disclosure triangle, the package-level tristate checkbox, the
/// color swatch, and the package name; the body holds nested folders and
/// files. Collapsed by default per the issue's acceptance criteria. File
/// checkbox toggles — including cascades from a parent click — bubble up
/// through `toggles`.
fn draw_package_node(ui: &mut egui::Ui, pkg: &PackageNode, toggles: &mut Vec<NodeId>) {
    let id = ui.make_persistent_id(format!("pkg-tree:pkg:{}", pkg.name));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    let check_id = format!("pkg-check:{}", pkg.name);
    state
        .show_header(ui, |ui| {
            // Package-level checkbox cascades the click over every
            // descendant file leaf. The rule mirrors the tree's fold:
            // fully checked → hide all; unchecked or mixed → show all.
            if draw_tristate_check(ui, &check_id, pkg.check) {
                toggles.extend(collect_package_leaves(pkg));
            }
            draw_color_swatch(ui, pkg.color_index);
            ui.label(egui::RichText::new(&pkg.name).monospace());
        })
        .body(|ui| {
            let salt = format!("pkg:{}", pkg.name);
            for child in &pkg.children {
                draw_folder_child(ui, child, &salt, toggles);
            }
        });
}

/// Render one child of a folder — either a nested folder (collapsible) or a
/// file leaf row.
fn draw_folder_child(
    ui: &mut egui::Ui,
    child: &FolderChild,
    parent_salt: &str,
    toggles: &mut Vec<NodeId>,
) {
    match child {
        FolderChild::Folder(folder) => draw_folder_node(ui, folder, parent_salt, toggles),
        FolderChild::File(file) => {
            let id = format!("{parent_salt}:file:{}", file.id);
            ui.horizontal(|ui| {
                if draw_tristate_check(ui, &id, file.check) {
                    toggles.push(file.id.clone());
                }
                ui.label(egui::RichText::new(&file.label).monospace().small());
            });
        }
    }
}

fn draw_folder_node(
    ui: &mut egui::Ui,
    folder: &FolderNode,
    parent_salt: &str,
    toggles: &mut Vec<NodeId>,
) {
    let salt = format!("{parent_salt}:folder:{}", folder.name);
    let id = ui.make_persistent_id(format!("pkg-tree:{salt}"));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    let check_id = format!("{salt}:check");
    state
        .show_header(ui, |ui| {
            // Folder-level checkbox cascades the same way as a package row.
            if draw_tristate_check(ui, &check_id, folder.check) {
                toggles.extend(collect_folder_leaves(folder));
            }
            ui.label(egui::RichText::new(&folder.name).monospace());
        })
        .body(|ui| {
            for child in &folder.children {
                draw_folder_child(ui, child, &salt, toggles);
            }
        });
}

fn draw_unpackaged_bucket(ui: &mut egui::Ui, bucket: &UnpackagedBucket, toggles: &mut Vec<NodeId>) {
    let salt = "unpackaged";
    let id = ui.make_persistent_id("pkg-tree:unpackaged");
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    state
        .show_header(ui, |ui| {
            // Bucket-level checkbox cascades over the flat file list.
            if draw_tristate_check(ui, "unpackaged:check", bucket.check) {
                for f in &bucket.files {
                    if should_emit_leaf(bucket.check, f.check) {
                        toggles.push(f.id.clone());
                    }
                }
            }
            ui.label(egui::RichText::new(UNPACKAGED_LABEL).italics());
        })
        .body(|ui| {
            for file in &bucket.files {
                let id = format!("{salt}:file:{}", file.id);
                ui.horizontal(|ui| {
                    if draw_tristate_check(ui, &id, file.check) {
                        toggles.push(file.id.clone());
                    }
                    ui.label(egui::RichText::new(&file.label).monospace().small());
                });
            }
        });
}

fn draw_externals_bucket(ui: &mut egui::Ui, bucket: &ExternalsBucket, toggles: &mut Vec<NodeId>) {
    let salt = "externals";
    let id = ui.make_persistent_id("pkg-tree:externals");
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    state
        .show_header(ui, |ui| {
            // Externals bucket cascades the same way — externals share the
            // same `FilterState` hide set as workspace files, so toggling
            // an external id routes through `toggle_file_visibility`
            // uniformly.
            if draw_tristate_check(ui, "externals:check", bucket.check) {
                for e in &bucket.externals {
                    if should_emit_leaf(bucket.check, e.check) {
                        toggles.push(e.id.clone());
                    }
                }
            }
            ui.label(egui::RichText::new(EXTERNALS_LABEL).italics());
        })
        .body(|ui| {
            for ext in &bucket.externals {
                let id = format!("{salt}:ext:{}", ext.id);
                ui.horizontal(|ui| {
                    if draw_tristate_check(ui, &id, ext.check) {
                        toggles.push(ext.id.clone());
                    }
                    ui.label(egui::RichText::new(&ext.label).monospace().small());
                });
            }
        });
}

/// Draw a checkbox that renders `Mixed` as the native tristate/indeterminate
/// style, `Checked` as on, and `Unchecked` as off. Returns `true` on the
/// frame where the user toggled it — the caller decides whether that maps
/// to a single-file toggle or a cascading parent click. The `checked` and
/// `indeterminate` bindings are scratch; the rendered state is re-derived
/// from `PackageTree::build` on every frame, so any mutation here is
/// overwritten next paint.
fn draw_tristate_check(ui: &mut egui::Ui, id: &str, state: CheckState) -> bool {
    let mut toggled = false;
    ui.scope(|ui| {
        ui.push_id(id, |ui| {
            let mut checked = matches!(state, CheckState::Checked);
            let indeterminate = matches!(state, CheckState::Mixed);
            let response =
                ui.add(egui::Checkbox::new(&mut checked, "").indeterminate(indeterminate));
            if response.changed() {
                toggled = true;
            }
        });
    });
    toggled
}

/// Walk a package subtree and emit the ids that a parent click should flip
/// given the current check state. The walk is biased: when the parent is
/// `Checked` we emit every visible leaf (to hide it); when `Unchecked` or
/// `Mixed` we emit every hidden leaf (to show it). Keeping the filter here
/// means the caller's toggle batch never contains a no-op flip.
fn collect_package_leaves(pkg: &PackageNode) -> Vec<NodeId> {
    let mut out = Vec::new();
    for child in &pkg.children {
        collect_child_leaves(child, pkg.check, &mut out);
    }
    out
}

fn collect_folder_leaves(folder: &FolderNode) -> Vec<NodeId> {
    let mut out = Vec::new();
    for child in &folder.children {
        collect_child_leaves(child, folder.check, &mut out);
    }
    out
}

fn collect_child_leaves(child: &FolderChild, parent: CheckState, out: &mut Vec<NodeId>) {
    match child {
        FolderChild::File(f) => {
            if should_emit_leaf(parent, f.check) {
                out.push(f.id.clone());
            }
        }
        FolderChild::Folder(f) => {
            for c in &f.children {
                collect_child_leaves(c, parent, out);
            }
        }
    }
}

/// True if a leaf in state `leaf` should be toggled when the user clicks a
/// parent currently in state `parent`. The rule mirrors the PRD cascade:
/// parent fully on → toggle every currently-on leaf (off), parent off or
/// mixed → toggle every currently-off leaf (on). Skipping no-op flips keeps
/// `FilterState::toggle` calls strictly meaningful.
fn should_emit_leaf(parent: CheckState, leaf: CheckState) -> bool {
    match parent {
        CheckState::Checked => matches!(leaf, CheckState::Checked),
        CheckState::Unchecked | CheckState::Mixed => matches!(leaf, CheckState::Unchecked),
    }
}

/// Small color chip matching the selection pane's swatch style. `None` picks
/// the neutral hint color so packages without a color-map entry still
/// render a placeholder rather than a layout gap.
fn draw_color_swatch(ui: &mut egui::Ui, color_index: Option<usize>) {
    let color = color_index
        .map(colors::package_color)
        .unwrap_or(colors::HINT);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
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
