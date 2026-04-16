use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::colors;
use crate::graph::{Graph, NodeId};
use crate::indexer::index_folder;
use crate::layout::{Layout, Vec2};

pub struct GruffApp {
    graph: Graph,
    layout: Layout,
    imports: HashMap<NodeId, Vec<NodeId>>,
    imported_by: HashMap<NodeId, Vec<NodeId>>,
    /// Stable index per package name, used to look up the node's color. The
    /// index order is whatever the first sweep through `graph.nodes` produced —
    /// good enough for deterministic colors within a single session.
    package_indices: HashMap<String, usize>,
    selected: Option<NodeId>,
    camera: Vec2,
    zoom: f32,
    sim_enabled: bool,
    status: String,
}

impl Default for GruffApp {
    fn default() -> Self {
        Self {
            graph: Graph::new(),
            layout: Layout::new(),
            imports: HashMap::new(),
            imported_by: HashMap::new(),
            package_indices: HashMap::new(),
            selected: None,
            camera: Vec2::new(0.0, 0.0),
            zoom: 1.0,
            sim_enabled: true,
            status: String::new(),
        }
    }
}

impl GruffApp {
    fn load_folder(&mut self, path: PathBuf) {
        let start = Instant::now();
        self.graph = index_folder(&path);

        // Precompute adjacency lists once per load so hit-test + sidebar
        // rendering don't iterate all edges per frame. Sorted for stable UI.
        self.imports.clear();
        self.imported_by.clear();
        for edge in &self.graph.edges {
            self.imports
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
            self.imported_by
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
        }
        for v in self.imports.values_mut() {
            v.sort();
        }
        for v in self.imported_by.values_mut() {
            v.sort();
        }

        // Assign each package a stable color index. Iterate in sorted name
        // order so the same repo reopens with the same colors regardless of
        // HashMap iteration order.
        self.package_indices.clear();
        let mut names: Vec<&str> = self
            .graph
            .nodes
            .values()
            .filter_map(|n| n.package.as_deref())
            .collect();
        names.sort();
        names.dedup();
        for name in names {
            let next = self.package_indices.len();
            self.package_indices.insert(name.to_string(), next);
        }

        self.layout = Layout::new();
        self.layout.sync(&self.graph);
        self.selected = None;
        self.camera = Vec2::new(0.0, 0.0);
        self.zoom = 1.0;
        self.sim_enabled = true;
        self.status = format!(
            "{} files, {} edges — indexed in {:.2}s",
            self.graph.nodes.len(),
            self.graph.edges.len(),
            start.elapsed().as_secs_f32(),
        );
    }

    fn world_to_screen(&self, world: Vec2, screen_center: egui::Pos2) -> egui::Pos2 {
        egui::pos2(
            (world.x - self.camera.x) * self.zoom + screen_center.x,
            (world.y - self.camera.y) * self.zoom + screen_center.y,
        )
    }

    /// Color for an unselected node. Workspace files take their owning
    /// package's color; files outside every package fall back to [`colors::NODE`].
    fn node_color(&self, id: &NodeId) -> egui::Color32 {
        let Some(node) = self.graph.nodes.get(id) else {
            return colors::NODE;
        };
        let Some(pkg) = node.package.as_deref() else {
            return colors::NODE;
        };
        let Some(&idx) = self.package_indices.get(pkg) else {
            return colors::NODE;
        };
        colors::package_color(idx)
    }

    fn node_render_radius(&self, id: &NodeId, zoom_scale: f32) -> f32 {
        let deps = self
            .imported_by
            .get(id)
            .map(|v| v.len())
            .unwrap_or(0) as f32;
        (4.0 + deps.sqrt() * 2.0) * zoom_scale
    }

    /// Find the topmost node at `screen_pos` within its visible radius.
    /// Nodes are compared in screen space so hit tolerance scales with zoom.
    fn pick_node(&self, screen_pos: egui::Pos2, screen_center: egui::Pos2) -> Option<NodeId> {
        let zoom_scale = self.zoom.clamp(0.5, 2.0);
        let mut best: Option<(f32, NodeId)> = None;
        for (id, world) in self.layout.iter() {
            let p = self.world_to_screen(world, screen_center);
            let dx = p.x - screen_pos.x;
            let dy = p.y - screen_pos.y;
            let dist_sq = dx * dx + dy * dy;
            let r = self.node_render_radius(id, zoom_scale);
            // Add a small pixel buffer so tiny nodes remain clickable.
            let hit_r = r + 2.0;
            if dist_sq <= hit_r * hit_r {
                match &best {
                    Some((d, _)) if *d <= dist_sq => {}
                    _ => best = Some((dist_sq, id.clone())),
                }
            }
        }
        best.map(|(_, id)| id)
    }

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        let Some(selected) = self.selected.clone() else {
            return;
        };
        let Some(node) = self.graph.nodes.get(&selected) else {
            // Selection stale (node removed) — clear it silently next frame.
            self.selected = None;
            return;
        };

        ui.add_space(6.0);
        ui.heading("Selected file");
        ui.add_space(4.0);

        ui.label(egui::RichText::new("Path").color(colors::HINT).small());
        ui.label(egui::RichText::new(&node.id).monospace());

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Owning package")
                .color(colors::HINT)
                .small(),
        );
        match node.package.as_deref() {
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
        // Left empty until cycle detection (slice #7).
        ui.label(egui::RichText::new("(pending cycle detection)").italics());
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

impl eframe::App for GruffApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        let [r, g, b, a] = colors::BG.to_array();
        [
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        ]
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Cmd+O (or Ctrl+O on non-mac via `modifiers.command`).
        let open_requested = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::O));
        if open_requested {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                self.load_folder(dir);
            }
        }

        // Space toggles the physics simulation (useful when it settles).
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.sim_enabled = !self.sim_enabled;
        }

        // Escape deselects.
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.selected = None;
        }

        // Drag-and-drop: use first dropped path; if it's a file, use its parent.
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            if let Some(path) = dropped.into_iter().find_map(|f| f.path) {
                let folder = if path.is_dir() {
                    path
                } else {
                    path.parent().map(PathBuf::from).unwrap_or(path)
                };
                self.load_folder(folder);
            }
        }

        // Step physics under an ~8 ms budget so huge graphs don't stall the UI.
        if self.sim_enabled && !self.layout.is_empty() {
            let budget = std::time::Duration::from_millis(8);
            let start = Instant::now();
            self.layout.step(1.0 / 60.0);
            let mut steps = 1;
            while steps < 3 && start.elapsed() < budget {
                self.layout.step(1.0 / 60.0);
                steps += 1;
            }
            ctx.request_repaint();
        }

        // Sidebar only appears when something is selected — it's the surface
        // that later slices will plug more content into.
        if self.selected.is_some() {
            egui::Panel::left("sidebar")
                .resizable(true)
                .default_size(280.0)
                .size_range(220.0..=480.0)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.draw_sidebar(ui);
                    });
                });
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| {
                self.draw_canvas(ui);
            });
    }
}

impl GruffApp {
    fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let rect = ui.max_rect();
        let center = rect.center();

        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.camera.x -= delta.x / self.zoom;
            self.camera.y -= delta.y / self.zoom;
        }

        // Click handling: select a node under the pointer, or deselect on empty.
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                self.selected = self.pick_node(click_pos, center);
            }
        }

        if let Some(hover) = response.hover_pos() {
            let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll_y.abs() > f32::EPSILON {
                let factor = (scroll_y * 0.0015).exp();
                let world_before_x = (hover.x - center.x) / self.zoom + self.camera.x;
                let world_before_y = (hover.y - center.y) / self.zoom + self.camera.y;
                self.zoom = (self.zoom * factor).clamp(0.05, 20.0);
                self.camera.x = world_before_x - (hover.x - center.x) / self.zoom;
                self.camera.y = world_before_y - (hover.y - center.y) / self.zoom;
            }
        }

        let painter = ui.painter_at(rect);
        let edge_stroke = egui::Stroke::new(1.0, colors::EDGE);

        // Edges first so nodes overlap them.
        for edge in &self.graph.edges {
            let (Some(pa), Some(pb)) = (self.layout.get(&edge.from), self.layout.get(&edge.to))
            else {
                continue;
            };
            let a = self.world_to_screen(pa, center);
            let b = self.world_to_screen(pb, center);
            painter.line_segment([a, b], edge_stroke);
        }

        let zoom_scale = self.zoom.clamp(0.5, 2.0);
        for (id, pos) in self.layout.iter() {
            let p = self.world_to_screen(pos, center);
            let radius = self.node_render_radius(id, zoom_scale);
            let is_selected = self.selected.as_ref() == Some(id);
            let color = if is_selected {
                colors::SELECTED
            } else {
                self.node_color(id)
            };
            painter.circle_filled(p, radius, color);
            if is_selected {
                // Outer ring makes the selection pop even at low zoom.
                painter.circle_stroke(
                    p,
                    radius + 3.0,
                    egui::Stroke::new(2.0, colors::SELECTED_RING),
                );
            }
        }

        if self.layout.is_empty() {
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "Drop a folder or press Cmd+O",
                egui::FontId::proportional(18.0),
                colors::HINT,
            );
        } else if !self.status.is_empty() {
            painter.text(
                egui::pos2(rect.left() + 12.0, rect.bottom() - 14.0),
                egui::Align2::LEFT_BOTTOM,
                &self.status,
                egui::FontId::proportional(12.0),
                colors::HINT,
            );
        }
    }
}
