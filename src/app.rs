use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::colors;
use crate::config::{self, Config};
use crate::editor::{self, OpenError};
use crate::graph::{Edge, Graph, NodeId, NodeKind};
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

/// Snapshot of the dependency chain a user clicked into — the clicked edge
/// plus every edge that lies on some path passing through it. Lets the
/// renderer highlight/dim in O(1) per edge instead of recomputing chains
/// every frame.
#[derive(Debug, Clone, Default)]
struct PathHighlight {
    /// `(from, to)` pairs for every edge on the chain, including the clicked
    /// edge itself. Used by the edge render loop to decide highlight vs dim.
    edges: HashSet<(NodeId, NodeId)>,
    /// Every node touched by an edge in `edges`. Used to decide which nodes
    /// stay full-opacity and which fade out.
    nodes: HashSet<NodeId>,
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
    /// Currently-highlighted dependency chain (set by clicking an edge),
    /// `None` when no chain is highlighted. Selecting a node or clicking
    /// empty canvas clears it.
    highlight: Option<PathHighlight>,
    selected: Option<NodeId>,
    camera: Vec2,
    zoom: f32,
    sim_enabled: bool,
    status: String,
    /// Count of fully-unresolvable dynamic imports (e.g. `import(modName)`)
    /// found during the last index. Surfaced in the status bar when nonzero
    /// so users know the graph isn't showing those edges.
    unresolved_dynamic: usize,
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
            highlight: None,
            selected: None,
            camera: Vec2::new(0.0, 0.0),
            zoom: 1.0,
            sim_enabled: true,
            status: String::new(),
            unresolved_dynamic: 0,
            config: config::load(),
            editor_prompt: None,
        }
    }
}

impl GruffApp {
    fn load_folder(&mut self, path: PathBuf) {
        let start = Instant::now();
        let result = index_folder(&path);
        self.graph = result.graph;
        self.unresolved_dynamic = result.unresolved_dynamic;

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
        self.highlight = None;

        self.layout = Layout::new();
        self.layout.sync(&self.graph);
        self.selected = None;
        self.camera = Vec2::new(0.0, 0.0);
        self.zoom = 1.0;
        self.sim_enabled = true;
        let mut status = format!(
            "{} files, {} edges, {} cycle{} — indexed in {:.2}s",
            self.graph.nodes.len(),
            self.graph.edges.len(),
            self.cycles.len(),
            if self.cycles.len() == 1 { "" } else { "s" },
            start.elapsed().as_secs_f32(),
        );
        if self.unresolved_dynamic > 0 {
            // Append rather than overwrite so the user still sees indexing
            // metrics; the dynamic-imports note is supplementary context.
            status.push_str(&format!(
                "  ·  {} unresolved dynamic import{}",
                self.unresolved_dynamic,
                if self.unresolved_dynamic == 1 { "" } else { "s" },
            ));
        }
        self.status = status;
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
        // Sqrt keeps the growth gentle; capping the dependents bonus prevents
        // a single megahub (1000+ dependents) from dwarfing every other node
        // into visual noise. The cap is world-units; the full radius is
        // zoomed afterwards so hotspots still scale with viewport zoom.
        let deps_bonus = (deps.sqrt() * 2.0).min(24.0);
        (4.0 + deps_bonus) * zoom_scale
    }

    /// Find the edge whose on-screen line segment is closest to `screen_pos`,
    /// within a pixel-distance tolerance. Returns `None` if nothing is close
    /// enough. Works purely in screen space so the hit tolerance is a fixed
    /// pixel distance regardless of zoom.
    fn pick_edge(
        &self,
        screen_pos: egui::Pos2,
        screen_center: egui::Pos2,
    ) -> Option<(NodeId, NodeId)> {
        // 5 px is comfortable on a trackpad without making edges impossible
        // to miss when the user actually meant to click empty canvas.
        const EDGE_HIT_PX: f32 = 5.0;
        let mut best: Option<(f32, (NodeId, NodeId))> = None;
        for edge in &self.graph.edges {
            let (Some(pa), Some(pb)) = (self.layout.get(&edge.from), self.layout.get(&edge.to))
            else {
                continue;
            };
            let a = self.world_to_screen(pa, screen_center);
            let b = self.world_to_screen(pb, screen_center);
            let d = point_to_segment_distance(screen_pos, a, b);
            if d <= EDGE_HIT_PX {
                match &best {
                    Some((bd, _)) if *bd <= d => {}
                    _ => best = Some((d, (edge.from.clone(), edge.to.clone()))),
                }
            }
        }
        best.map(|(_, e)| e)
    }

    /// Build a [`PathHighlight`] for the clicked edge `u -> v`: walk backward
    /// from `u` collecting ancestors and forward from `v` collecting
    /// descendants, then gather every edge whose endpoints both lie in one
    /// of those reachable sets. The result represents every simple path
    /// through the graph that passes through the clicked edge. BFS uses a
    /// visited set, so cyclic regions don't cause infinite loops.
    fn build_path_highlight(&self, from: &NodeId, to: &NodeId) -> PathHighlight {
        compute_path_highlight(
            from,
            to,
            &self.graph.edges,
            &self.imports,
            &self.imported_by,
        )
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

/// Shortest distance from a point to a line segment in 2D (screen space).
/// Used by edge hit-testing so clicking near an edge — not just exactly on
/// its 1 px stroke — still selects it.
fn point_to_segment_distance(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let len_sq = abx * abx + aby * aby;
    if len_sq < f32::EPSILON {
        // Degenerate segment (a == b): fall back to point-to-point distance.
        let dx = p.x - a.x;
        let dy = p.y - a.y;
        return (dx * dx + dy * dy).sqrt();
    }
    let apx = p.x - a.x;
    let apy = p.y - a.y;
    // Clamp to [0, 1] so we only measure against the segment, not the
    // infinite line it sits on — near-colinear clicks past the endpoints
    // correctly measure to the endpoint.
    let t = ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0);
    let projx = a.x + t * abx;
    let projy = a.y + t * aby;
    let dx = p.x - projx;
    let dy = p.y - projy;
    (dx * dx + dy * dy).sqrt()
}

/// Pure path-highlight builder: collect every edge that lies on a path
/// passing through the clicked edge `(from -> to)`. Factored out of
/// [`GruffApp`] so it can be unit-tested without standing up an egui
/// context.
fn compute_path_highlight(
    from: &NodeId,
    to: &NodeId,
    edges: &[Edge],
    imports: &HashMap<NodeId, Vec<NodeId>>,
    imported_by: &HashMap<NodeId, Vec<NodeId>>,
) -> PathHighlight {
    // Backward reach from `from` along reverse edges — every ancestor whose
    // imports can eventually land on `from`. Forward reach from `to` along
    // forward edges — every descendant `to` can reach. Together these bound
    // the set of nodes that lie on any path through the clicked edge.
    let upstream = bfs_visit(from, imported_by);
    let downstream = bfs_visit(to, imports);

    let mut hl_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
    hl_edges.insert((from.clone(), to.clone()));
    for e in edges {
        // An edge is on some path through (from -> to) iff both endpoints
        // live in the same reach set. Crossing sets (e.g. an ancestor
        // pointing into a descendant without going through the clicked
        // edge) are intentionally excluded — they aren't a chain through
        // the clicked edge.
        let in_up = upstream.contains(&e.from) && upstream.contains(&e.to);
        let in_down = downstream.contains(&e.from) && downstream.contains(&e.to);
        if in_up || in_down {
            hl_edges.insert((e.from.clone(), e.to.clone()));
        }
    }

    let mut nodes: HashSet<NodeId> = HashSet::new();
    nodes.extend(upstream);
    nodes.extend(downstream);
    PathHighlight {
        edges: hl_edges,
        nodes,
    }
}

/// Iterative BFS over `adj` starting from `start`, returning every visited
/// node (including `start`). The visited set is what makes this safe on
/// cyclic graphs: a node only ever gets pushed once.
fn bfs_visit(start: &NodeId, adj: &HashMap<NodeId, Vec<NodeId>>) -> HashSet<NodeId> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = Vec::new();
    seen.insert(start.clone());
    stack.push(start.clone());
    while let Some(n) = stack.pop() {
        if let Some(neighbors) = adj.get(&n) {
            for nb in neighbors {
                if seen.insert(nb.clone()) {
                    stack.push(nb.clone());
                }
            }
        }
    }
    seen
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
        // Also clears any active edge-path highlight so `Escape` is a
        // universal "go back to a clean canvas" key.
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.editor_prompt.is_some() {
                self.editor_prompt = None;
            } else {
                self.selected = None;
                self.highlight = None;
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

        // Click handling: nodes take priority over edges (the hit radius is
        // tighter than the edge tolerance, and the user almost always meant
        // the node when both are under the cursor). An empty-canvas click
        // clears both the selection and any active path highlight.
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                if let Some(node) = self.pick_node(click_pos, center) {
                    self.selected = Some(node);
                    self.highlight = None;
                } else if let Some((from, to)) = self.pick_edge(click_pos, center) {
                    self.highlight = Some(self.build_path_highlight(&from, &to));
                    self.selected = None;
                } else {
                    self.selected = None;
                    self.highlight = None;
                }
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
        // Dim non-highlighted elements so the highlighted chain reads clearly.
        // Alpha roughly preserves the layout shape in the background without
        // competing with the highlighted edges for attention.
        const DIM_ALPHA: f32 = 0.18;
        let highlight_active = self.highlight.is_some();

        // Edges first so nodes overlap them.
        for edge in &self.graph.edges {
            let (Some(pa), Some(pb)) = (self.layout.get(&edge.from), self.layout.get(&edge.to))
            else {
                continue;
            };
            let a = self.world_to_screen(pa, center);
            let b = self.world_to_screen(pb, center);
            let is_cycle = self.edge_in_cycle(&edge.from, &edge.to);
            let on_path = self
                .highlight
                .as_ref()
                .is_some_and(|h| h.edges.contains(&(edge.from.clone(), edge.to.clone())));

            // Base color: cycle edges keep their red identity even when
            // highlighted — the highlight manifests as increased width /
            // opacity rather than overwriting the cycle signal.
            let base = if is_cycle {
                colors::CYCLE_EDGE
            } else if on_path {
                colors::PATH_EDGE
            } else {
                colors::EDGE
            };
            let (width, color) = if on_path {
                (2.0, base)
            } else if highlight_active {
                (1.0, base.gamma_multiply(DIM_ALPHA))
            } else if is_cycle {
                (1.6, base)
            } else {
                (1.0, base)
            };
            painter.line_segment([a, b], egui::Stroke::new(width, color));
        }

        let zoom_scale = self.zoom.clamp(0.5, 2.0);
        for (id, pos) in self.layout.iter() {
            let p = self.world_to_screen(pos, center);
            let radius = self.node_render_radius(id, zoom_scale);
            let is_selected = self.selected.as_ref() == Some(id);
            let on_path = self
                .highlight
                .as_ref()
                .is_some_and(|h| h.nodes.contains(id));
            let base = if is_selected {
                colors::SELECTED
            } else {
                self.node_color(id)
            };
            // Fade nodes that aren't on the highlighted chain; keep the
            // selected node fully opaque even when no chain is active.
            let color = if highlight_active && !on_path && !is_selected {
                base.gamma_multiply(DIM_ALPHA)
            } else {
                base
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;

    fn id(s: &str) -> NodeId {
        s.to_string()
    }

    /// Build forward/reverse adjacency lists from a flat edge list, matching
    /// the shape `GruffApp` caches at load time.
    fn adj(edges: &[Edge]) -> (HashMap<NodeId, Vec<NodeId>>, HashMap<NodeId, Vec<NodeId>>) {
        let mut fwd: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut rev: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for e in edges {
            fwd.entry(e.from.clone()).or_default().push(e.to.clone());
            rev.entry(e.to.clone()).or_default().push(e.from.clone());
        }
        (fwd, rev)
    }

    fn edge(from: &str, to: &str) -> Edge {
        Edge {
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    #[test]
    fn linear_chain_click_middle_highlights_everything() {
        // a -> b -> c -> d : clicking (b -> c) should highlight every edge
        // and every node — the chain fills the graph.
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("c", "d")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("b"), &id("c"), &edges, &fwd, &rev);
        assert!(hl.edges.contains(&(id("a"), id("b"))));
        assert!(hl.edges.contains(&(id("b"), id("c"))));
        assert!(hl.edges.contains(&(id("c"), id("d"))));
        assert_eq!(hl.edges.len(), 3);
        for n in ["a", "b", "c", "d"] {
            assert!(hl.nodes.contains(&id(n)), "expected node {n} in highlight");
        }
    }

    #[test]
    fn branching_graph_excludes_sibling_branches() {
        //        /-> x
        //   a -> b -> c -> d
        //        \-> y
        // Clicking (b -> c) should include a and d (on the chain) but NOT
        // x or y — they branch away from the clicked edge.
        let edges = vec![
            edge("a", "b"),
            edge("b", "c"),
            edge("c", "d"),
            edge("b", "x"),
            edge("b", "y"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("b"), &id("c"), &edges, &fwd, &rev);
        assert!(hl.nodes.contains(&id("a")));
        assert!(hl.nodes.contains(&id("d")));
        assert!(!hl.nodes.contains(&id("x")));
        assert!(!hl.nodes.contains(&id("y")));
        // And the sibling-branch edges are excluded from the highlight.
        assert!(!hl.edges.contains(&(id("b"), id("x"))));
        assert!(!hl.edges.contains(&(id("b"), id("y"))));
    }

    #[test]
    fn cycle_does_not_cause_infinite_loop() {
        // a -> b -> c -> a forms a cycle. Clicking any edge inside the cycle
        // should terminate and highlight every member edge.
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("c", "a")];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("a"), &id("b"), &edges, &fwd, &rev);
        assert_eq!(hl.edges.len(), 3);
        assert_eq!(hl.nodes.len(), 3);
        for from_to in [("a", "b"), ("b", "c"), ("c", "a")] {
            assert!(
                hl.edges.contains(&(id(from_to.0), id(from_to.1))),
                "missing cycle edge {from_to:?}"
            );
        }
    }

    #[test]
    fn cycle_with_external_tail_and_head() {
        // root -> a -> b -> a (cycle a<->b) -> c -> leaf
        // Clicking (a -> b) inside the cycle should reach root (upstream of
        // the cycle) and leaf (downstream of the cycle) — the full chain —
        // without looping.
        let edges = vec![
            edge("root", "a"),
            edge("a", "b"),
            edge("b", "a"),
            edge("b", "c"),
            edge("c", "leaf"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("a"), &id("b"), &edges, &fwd, &rev);
        for n in ["root", "a", "b", "c", "leaf"] {
            assert!(hl.nodes.contains(&id(n)), "expected {n} in path highlight");
        }
    }

    #[test]
    fn unrelated_component_is_not_highlighted() {
        // Two disjoint chains: clicking one must leave the other dim.
        let edges = vec![
            edge("a", "b"),
            edge("b", "c"),
            edge("x", "y"),
            edge("y", "z"),
        ];
        let (fwd, rev) = adj(&edges);
        let hl = compute_path_highlight(&id("a"), &id("b"), &edges, &fwd, &rev);
        assert!(hl.nodes.contains(&id("a")));
        assert!(hl.nodes.contains(&id("c")));
        assert!(!hl.nodes.contains(&id("x")));
        assert!(!hl.nodes.contains(&id("y")));
        assert!(!hl.nodes.contains(&id("z")));
    }

    #[test]
    fn point_to_segment_distance_clamps_to_endpoints() {
        let a = egui::pos2(0.0, 0.0);
        let b = egui::pos2(10.0, 0.0);

        // Midpoint perpendicular.
        let d_mid = point_to_segment_distance(egui::pos2(5.0, 3.0), a, b);
        assert!((d_mid - 3.0).abs() < 0.001);

        // Past the endpoint — must measure to the endpoint, not the line.
        let d_past = point_to_segment_distance(egui::pos2(20.0, 0.0), a, b);
        assert!((d_past - 10.0).abs() < 0.001);

        // On the segment — zero distance.
        let d_on = point_to_segment_distance(egui::pos2(5.0, 0.0), a, b);
        assert!(d_on < 0.001);

        // Degenerate zero-length segment falls back to point distance.
        let d_degen = point_to_segment_distance(egui::pos2(3.0, 4.0), a, a);
        assert!((d_degen - 5.0).abs() < 0.001);
    }
}
