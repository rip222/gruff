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
    dependents: HashMap<NodeId, usize>,
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
            dependents: HashMap::new(),
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

        // Precompute dependents counts once per load so per-frame draw doesn't
        // iterate all edges for every node.
        self.dependents.clear();
        for edge in &self.graph.edges {
            *self.dependents.entry(edge.to.clone()).or_insert(0) += 1;
        }

        self.layout = Layout::new();
        self.layout.sync(&self.graph);
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

        let rect = ui.max_rect();
        let center = rect.center();

        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.camera.x -= delta.x / self.zoom;
            self.camera.y -= delta.y / self.zoom;
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
            let deps = self.dependents.get(id).copied().unwrap_or(0) as f32;
            let radius = (4.0 + deps.sqrt() * 2.0) * zoom_scale;
            painter.circle_filled(p, radius, colors::NODE);
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
