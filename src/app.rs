use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::colors;
use crate::config::{self, Config};
use crate::editor::{self, OpenError};
use crate::graph::{Graph, NodeId, NodeKind};
use crate::indexer::index_folder;
use crate::layout::{Layout, Vec2};

/// Quick-pick editor commands surfaced as buttons in the picker. Ordered by
/// rough popularity on macOS — VS Code first, then the classic terminal
/// editors, then JetBrains.
const QUICK_PICK_EDITORS: &[&str] = &["code", "cursor", "subl", "idea", "nvim", "vim"];

/// Action queued while the editor picker modal is open, resumed after the
/// user picks a valid editor.
#[derive(Debug, Clone)]
enum PendingEditorAction {
    OpenFile(PathBuf),
    RevealConfig,
}

struct EditorPromptState {
    /// Text the user is typing / has picked — pre-seeded with the current
    /// config value (often empty on first launch).
    input: String,
    /// What to do once an editor is chosen. Kept on the prompt so the same
    /// picker handles "open file" and "reveal config" symmetrically.
    pending: PendingEditorAction,
    /// Error from the previous attempt (if any) — e.g. "editor not found".
    error: Option<String>,
}

pub struct GruffApp {
    graph: Graph,
    layout: Layout,
    imports: HashMap<NodeId, Vec<NodeId>>,
    imported_by: HashMap<NodeId, Vec<NodeId>>,
    /// Stable index per package name, used to look up the node's color. The
    /// index order is whatever the first sweep through `graph.nodes` produced —
    /// good enough for deterministic colors within a single session.
    package_indices: HashMap<String, usize>,
    /// Cyclic SCCs reported by `Graph::cycles`. Each entry is the node set of
    /// one circular-dependency region. Recomputed on every load.
    cycles: Vec<Vec<NodeId>>,
    /// Reverse index: for each node participating in a cycle, the index into
    /// `cycles` of the SCC it belongs to. Used to tint cycle edges red in O(1)
    /// and to surface "cycles this file participates in" in the sidebar.
    cycle_of: HashMap<NodeId, usize>,
    /// Deferred "frame this cycle in the viewport" request set by the sidebar
    /// and consumed by `draw_canvas` (which has the screen rect in hand).
    frame_request: Option<usize>,
    selected: Option<NodeId>,
    camera: Vec2,
    zoom: f32,
    sim_enabled: bool,
    status: String,
    /// User settings loaded from `~/.gruff/config.toml`. Mutated when the user
    /// picks an editor through the modal; persisted immediately on change.
    config: Config,
    /// Editor picker modal state; `Some` while the prompt is visible.
    editor_prompt: Option<EditorPromptState>,
}

impl Default for GruffApp {
    fn default() -> Self {
        Self {
            graph: Graph::new(),
            layout: Layout::new(),
            imports: HashMap::new(),
            imported_by: HashMap::new(),
            package_indices: HashMap::new(),
            cycles: Vec::new(),
            cycle_of: HashMap::new(),
            frame_request: None,
            selected: None,
            camera: Vec2::new(0.0, 0.0),
            zoom: 1.0,
            sim_enabled: true,
            status: String::new(),
            config: config::load(),
            editor_prompt: None,
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

        // Detect cycles once per load. Tarjan's is O(V+E) so this is cheap
        // even on large graphs, and lets every frame look up cycle membership
        // in O(1) via `cycle_of` rather than re-running SCC.
        self.cycles = self.graph.cycles();
        // Sort cycle members for stable sidebar presentation. Tarjan's returns
        // them in stack-pop order, which is deterministic per-run but visually
        // arbitrary — alphabetical reads better in a list.
        for cycle in &mut self.cycles {
            cycle.sort();
        }
        // Sort cycles themselves by their smallest member so the list order
        // stays stable across reruns of the same graph.
        self.cycles
            .sort_by(|a, b| a.first().cmp(&b.first()).then(a.len().cmp(&b.len())));
        self.cycle_of.clear();
        for (idx, cycle) in self.cycles.iter().enumerate() {
            for node in cycle {
                self.cycle_of.insert(node.clone(), idx);
            }
        }
        self.frame_request = None;

        self.layout = Layout::new();
        self.layout.sync(&self.graph);
        self.selected = None;
        self.camera = Vec2::new(0.0, 0.0);
        self.zoom = 1.0;
        self.sim_enabled = true;
        self.status = format!(
            "{} files, {} edges, {} cycle{} — indexed in {:.2}s",
            self.graph.nodes.len(),
            self.graph.edges.len(),
            self.cycles.len(),
            if self.cycles.len() == 1 { "" } else { "s" },
            start.elapsed().as_secs_f32(),
        );
    }

    /// True when this edge lies inside a cyclic SCC. An edge `(u,v)` is on a
    /// cycle iff `u` and `v` sit in the same SCC *and* that SCC is itself
    /// cyclic (multi-node or self-loop) — which is exactly the condition
    /// `cycle_of` encodes (it only maps nodes that participate in a cycle).
    fn edge_in_cycle(&self, from: &str, to: &str) -> bool {
        match (self.cycle_of.get(from), self.cycle_of.get(to)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// Center and zoom the viewport so cycle `idx`'s members fit inside the
    /// canvas rect with a comfortable margin. Self-loop cycles (single node,
    /// zero-size bbox) are handled by falling back to a generous fixed zoom.
    fn frame_cycle(&mut self, idx: usize, rect: egui::Rect) {
        let Some(cycle) = self.cycles.get(idx) else {
            return;
        };
        let mut positions = cycle.iter().filter_map(|id| self.layout.get(id));
        let Some(first) = positions.next() else {
            return;
        };
        let (mut min_x, mut max_x, mut min_y, mut max_y) =
            (first.x, first.x, first.y, first.y);
        for p in positions {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        self.camera = Vec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);

        // Padding leaves breathing room around the framed cycle so nodes
        // aren't jammed against the viewport edge. Additive (in world units)
        // rather than multiplicative so a single self-loop doesn't collapse
        // to zero padding.
        const PAD: f32 = 60.0;
        let bbox_w = (max_x - min_x) + PAD * 2.0;
        let bbox_h = (max_y - min_y) + PAD * 2.0;
        let fit_x = rect.width() / bbox_w.max(1.0);
        let fit_y = rect.height() / bbox_h.max(1.0);
        self.zoom = fit_x.min(fit_y).clamp(0.05, 20.0);
    }

    fn world_to_screen(&self, world: Vec2, screen_center: egui::Pos2) -> egui::Pos2 {
        egui::pos2(
            (world.x - self.camera.x) * self.zoom + screen_center.x,
            (world.y - self.camera.y) * self.zoom + screen_center.y,
        )
    }

    /// Color for an unselected node. External leaves render neutral gray so
    /// `node_modules` packages read as "not our code" at a glance; workspace
    /// files take their owning package's color; fallbacks use [`colors::NODE`].
    fn node_color(&self, id: &NodeId) -> egui::Color32 {
        let Some(node) = self.graph.nodes.get(id) else {
            return colors::NODE;
        };
        if node.kind == NodeKind::External {
            return colors::EXTERNAL_NODE;
        }
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

    /// Launch the user's editor on `path`. Opens the editor picker modal when
    /// no editor is configured or the configured one isn't on PATH.
    fn try_open_in_editor(&mut self, path: PathBuf) {
        match editor::open_in_editor(&self.config.editor.name, &path) {
            Ok(()) => {
                self.status = format!("Opened {} in {}", path.display(), self.config.editor.name);
            }
            Err(OpenError::NotConfigured) => {
                self.open_editor_prompt(PendingEditorAction::OpenFile(path), None);
            }
            Err(OpenError::NotFound(name)) => {
                self.open_editor_prompt(
                    PendingEditorAction::OpenFile(path),
                    Some(format!("Couldn't find `{name}` on your PATH. Pick another editor:")),
                );
            }
            Err(OpenError::Io(e)) => {
                self.status = format!("Failed to launch editor: {e}");
            }
        }
    }

    /// Open `~/.gruff/config.toml` in the user's editor, creating it with
    /// sensible defaults first if it doesn't exist yet. Backs the "Reveal
    /// config file" menu item.
    fn reveal_config_file(&mut self) {
        let path = match config::ensure_exists() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("Failed to prepare config file: {e}");
                return;
            }
        };
        match editor::open_in_editor(&self.config.editor.name, &path) {
            Ok(()) => self.status = format!("Opened {}", path.display()),
            Err(OpenError::NotConfigured) => {
                self.open_editor_prompt(PendingEditorAction::RevealConfig, None);
            }
            Err(OpenError::NotFound(name)) => {
                self.open_editor_prompt(
                    PendingEditorAction::RevealConfig,
                    Some(format!("Couldn't find `{name}` on your PATH. Pick another editor:")),
                );
            }
            Err(OpenError::Io(e)) => {
                self.status = format!("Failed to launch editor: {e}");
            }
        }
    }

    fn open_editor_prompt(&mut self, pending: PendingEditorAction, error: Option<String>) {
        // Seed the input with the current config value so users who already
        // have (say) a broken `code` config can tweak it rather than retype.
        self.editor_prompt = Some(EditorPromptState {
            input: self.config.editor.name.clone(),
            pending,
            error,
        });
    }

    /// Persist the new editor name to config and resume the pending action.
    /// Leaves the modal open with an error message if the editor still isn't
    /// found so the user can pick another without reopening the prompt.
    fn apply_picked_editor(&mut self, name: String) {
        let trimmed = name.trim().to_string();
        if trimmed.is_empty() {
            if let Some(prompt) = &mut self.editor_prompt {
                prompt.error = Some("Please enter an editor command (e.g. `code` or `vim`).".to_string());
            }
            return;
        }
        self.config.editor.name = trimmed.clone();
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Saved editor choice in-memory only — couldn't write config: {e}");
        }

        let Some(prompt) = self.editor_prompt.take() else { return; };
        let pending = prompt.pending;
        match pending {
            PendingEditorAction::OpenFile(path) => {
                match editor::open_in_editor(&self.config.editor.name, &path) {
                    Ok(()) => {
                        self.status = format!("Opened {} in {}", path.display(), trimmed);
                    }
                    Err(OpenError::NotFound(n)) => {
                        // Reopen the prompt with the new error so the user can
                        // try yet another editor without an extra click.
                        self.editor_prompt = Some(EditorPromptState {
                            input: trimmed,
                            pending: PendingEditorAction::OpenFile(path),
                            error: Some(format!("Couldn't find `{n}` on your PATH. Pick another editor:")),
                        });
                    }
                    Err(OpenError::Io(e)) => {
                        self.status = format!("Failed to launch editor: {e}");
                    }
                    Err(OpenError::NotConfigured) => {
                        // Shouldn't happen — we just saved a non-empty name.
                    }
                }
            }
            PendingEditorAction::RevealConfig => {
                match config::ensure_exists() {
                    Ok(path) => match editor::open_in_editor(&self.config.editor.name, &path) {
                        Ok(()) => self.status = format!("Opened {}", path.display()),
                        Err(OpenError::NotFound(n)) => {
                            self.editor_prompt = Some(EditorPromptState {
                                input: trimmed,
                                pending: PendingEditorAction::RevealConfig,
                                error: Some(format!("Couldn't find `{n}` on your PATH. Pick another editor:")),
                            });
                        }
                        Err(OpenError::Io(e)) => {
                            self.status = format!("Failed to launch editor: {e}");
                        }
                        Err(OpenError::NotConfigured) => {}
                    },
                    Err(e) => self.status = format!("Failed to prepare config file: {e}"),
                }
            }
        }
    }

    fn draw_editor_prompt(&mut self, ctx: &egui::Context) {
        let Some(prompt) = &self.editor_prompt else { return; };
        // Lift fields out — the modal body wants `&mut self` access to apply
        // the user's choice, so we operate on a snapshot and write back at the
        // end via `self.editor_prompt`.
        let mut input = prompt.input.clone();
        let error = prompt.error.clone();
        let pending_label = match &prompt.pending {
            PendingEditorAction::OpenFile(p) => {
                format!("to open {}", p.display())
            }
            PendingEditorAction::RevealConfig => "to open your config file".to_string(),
        };

        let mut cancel = false;
        let mut submit: Option<String> = None;

        egui::Modal::new(egui::Id::new("gruff-editor-prompt")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.heading("Pick your editor");
            ui.add_space(6.0);
            ui.label(format!("Gruff needs an editor command {pending_label}."));
            ui.add_space(8.0);

            if let Some(msg) = &error {
                ui.colored_label(colors::CYCLE_EDGE, msg);
                ui.add_space(6.0);
            }

            ui.label(
                egui::RichText::new("Editor command")
                    .color(colors::HINT)
                    .small(),
            );
            let edit = ui.add(
                egui::TextEdit::singleline(&mut input)
                    .desired_width(f32::INFINITY)
                    .hint_text("e.g. code, vim, subl"),
            );
            // Submit on Enter from the text field — standard keyboard flow.
            let enter_pressed =
                edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Quick picks")
                    .color(colors::HINT)
                    .small(),
            );
            ui.horizontal_wrapped(|ui| {
                for name in QUICK_PICK_EDITORS {
                    if ui.button(*name).clicked() {
                        input = (*name).to_string();
                    }
                }
            });

            ui.add_space(10.0);
            ui.label(
                egui::RichText::new(
                    "You can change this later in ~/.gruff/config.toml.",
                )
                .italics()
                .color(colors::HINT)
                .small(),
            );

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
                let open_clicked = ui.button("Use this editor").clicked();
                if open_clicked || enter_pressed {
                    submit = Some(input.clone());
                }
            });
        });

        // Write the edited input back so keystrokes persist across frames.
        if let Some(prompt) = &mut self.editor_prompt {
            prompt.input = input;
        }
        if cancel {
            self.editor_prompt = None;
            return;
        }
        if let Some(name) = submit {
            self.apply_picked_editor(name);
        }
    }

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
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

        // Escape deselects, or dismisses the editor prompt if it's open.
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.editor_prompt.is_some() {
                self.editor_prompt = None;
            } else {
                self.selected = None;
            }
        }

        egui::Panel::top("menu_bar").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open folder…").clicked() {
                        ui.close();
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            self.load_folder(dir);
                        }
                    }
                    ui.separator();
                    if ui.button("Reveal config file").clicked() {
                        ui.close();
                        self.reveal_config_file();
                    }
                });
            });
        });

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

        // Sidebar appears whenever there's content worth showing: a current
        // selection or at least one detected cycle. Keeping it hidden on an
        // empty canvas preserves the onboarding-only initial screen.
        if self.selected.is_some() || !self.cycles.is_empty() {
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

        // Drawn last so the modal paints over the sidebar and canvas.
        self.draw_editor_prompt(&ctx);
    }
}

impl GruffApp {
    fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let rect = ui.max_rect();
        let center = rect.center();

        // Consume any pending "frame this cycle" request before drawing, so
        // the updated camera applies to this frame rather than lagging by one.
        if let Some(idx) = self.frame_request.take() {
            self.frame_cycle(idx, rect);
        }

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
        let normal_stroke = egui::Stroke::new(1.0, colors::EDGE);
        // Cycle edges get a slightly thicker stroke on top of the red tint so
        // they pop even when the canvas is busy.
        let cycle_stroke = egui::Stroke::new(1.6, colors::CYCLE_EDGE);

        // Edges first so nodes overlap them.
        for edge in &self.graph.edges {
            let (Some(pa), Some(pb)) = (self.layout.get(&edge.from), self.layout.get(&edge.to))
            else {
                continue;
            };
            let a = self.world_to_screen(pa, center);
            let b = self.world_to_screen(pb, center);
            let stroke = if self.edge_in_cycle(&edge.from, &edge.to) {
                cycle_stroke
            } else {
                normal_stroke
            };
            painter.line_segment([a, b], stroke);
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
