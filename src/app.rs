use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::colors;
use crate::config::{self, Config};
use crate::graph::{Graph, NodeId};
use crate::indexer::index_folder;
use crate::layout::{Layout, Vec2};

mod canvas;
mod editor_prompt;
mod highlight;
mod search;
mod sidebar;

use editor_prompt::EditorPromptState;
use highlight::PathHighlight;
use search::SearchState;

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
    /// Cmd+F fuzzy-search overlay state; `Some` while the overlay is open.
    /// `None` closes the overlay and clears any search-driven dim state —
    /// the renderer keys off `is_some` rather than a separate flag.
    search: Option<SearchState>,
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
            search: None,
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
        // Reset the search overlay on a new folder load — the cached match
        // set references node ids from the previous graph and would be
        // nonsensical against the new one.
        self.search = None;

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

        // Cmd+F opens (or closes) the fuzzy-search overlay. We gate it on
        // having a loaded graph so the shortcut doesn't surface a useless
        // empty overlay on the initial onboarding screen.
        let search_requested =
            ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F));
        if search_requested && !self.graph.nodes.is_empty() {
            self.toggle_search();
        }

        // Space toggles the physics simulation (useful when it settles).
        // Skip while the search overlay is open so typing a space in the
        // search box doesn't pause physics.
        if self.search.is_none() && ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.sim_enabled = !self.sim_enabled;
        }

        // Escape deselects, or dismisses whichever overlay is open. Priority:
        // editor prompt → search overlay → clear selection/highlight. Each
        // stage handles exactly one pressed Escape so the user can unwind
        // state layer by layer.
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.editor_prompt.is_some() {
                self.editor_prompt = None;
            } else if self.search.is_some() {
                // Closing the overlay clears the dim state per the issue's
                // acceptance criteria — the search struct owns both.
                self.search = None;
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

        // Drawn before the editor prompt so a modal still paints on top of
        // the search overlay — the modal is strictly more blocking.
        self.draw_search_overlay(&ctx);
        // Drawn last so the modal paints over the sidebar and canvas.
        self.draw_editor_prompt(&ctx);
    }
}
