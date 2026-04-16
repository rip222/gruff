use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::colors;
use crate::config::{self, Config};
use crate::error::{self, GruffError};
use crate::export;
use crate::graph::{Graph, GraphDiff, NodeId};
use crate::indexer::Indexer;
use crate::layout::{Layout, Vec2};
use crate::watcher::{ChangeEvent, Watcher};

mod canvas;
mod editor_prompt;
mod highlight;
mod search;
mod sidebar;
mod status_bar;

use editor_prompt::EditorPromptState;
use highlight::PathHighlight;
use search::SearchState;

pub struct GruffApp {
    graph: Graph,
    /// Mirrors the indexer's `include_tests` option so the menu checkbox has
    /// somewhere to bind without reaching into `self.indexer` through a
    /// closure. Kept in sync by [`GruffApp::set_include_tests`].
    include_tests: bool,
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
    /// Session-scoped runtime failures (panic hook, watcher startup, etc.)
    /// that don't belong to one specific file snapshot.
    runtime_errors: VecDeque<GruffError>,
    /// User settings loaded from `~/.gruff/config.toml`. Mutated when the user
    /// picks an editor through the modal; persisted immediately on change.
    config: Config,
    /// Editor picker modal state; `Some` while the prompt is visible.
    editor_prompt: Option<EditorPromptState>,
    /// Cmd+F fuzzy-search overlay state; `Some` while the overlay is open.
    /// `None` closes the overlay and clears any search-driven dim state —
    /// the renderer keys off `is_some` rather than a separate flag.
    search: Option<SearchState>,
    /// Stateful indexer owned by the app for the currently-loaded folder.
    /// `None` until the first folder is opened. Stored so incremental
    /// updates (watcher events, Cmd+R) can patch the graph in place
    /// without re-running the full scan each time.
    indexer: Option<Indexer>,
    /// Filesystem watcher for the currently-loaded folder. `None` before
    /// the first folder is opened. Dropped when a new folder replaces it.
    watcher: Option<Watcher>,
    /// Root of the currently-loaded folder, kept so Cmd+R can rescan the
    /// same folder without a file picker round-trip.
    last_root: Option<PathBuf>,
}

impl Default for GruffApp {
    fn default() -> Self {
        Self {
            graph: Graph::new(),
            include_tests: false,
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
            runtime_errors: VecDeque::new(),
            config: config::load(),
            editor_prompt: None,
            search: None,
            indexer: None,
            watcher: None,
            last_root: None,
        }
    }
}

impl GruffApp {
    /// Build a fresh app and, if the config records a `last_repo` that still
    /// points at a readable directory, auto-open it so the user doesn't have
    /// to re-drop the folder every session. A missing or stale path silently
    /// falls back to the onboarding hint — same state as a first launch.
    pub fn with_autoload() -> Self {
        let mut app = Self::default();
        if let Some(path) = app.config.last_repo.clone() {
            if is_readable_dir(&path) {
                app.load_folder(path);
            }
        }
        app
    }

    fn load_folder(&mut self, path: PathBuf) {
        let start = Instant::now();
        let mut indexer = Indexer::build(&path);
        // Carry the toggle across folder changes so the user doesn't have to
        // re-enable "include test files" every time they open a new repo.
        if indexer.options.include_tests != self.include_tests {
            indexer.set_include_tests(self.include_tests);
        }
        self.graph = indexer.graph.clone();
        self.unresolved_dynamic = indexer.unresolved_dynamic;
        let root = indexer.ws.root.clone();
        self.last_root = Some(root.clone());
        self.indexer = Some(indexer);

        // Remember this repo so the next launch re-opens it. Per the PRD this
        // is the *only* thing persisted across sessions — layout, selection,
        // and filter state are deliberately session-scoped. A write failure
        // just means the user won't get auto-reopen; don't surface it.
        if self.config.last_repo.as_ref() != Some(&root) {
            self.config.last_repo = Some(root);
            let _ = config::save(&self.config);
        }

        self.rebuild_derived_indexes();
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

        // Start (or replace) the filesystem watcher. Drop happens first so
        // the old watcher's thread shuts down before we spawn a new one for
        // the just-loaded folder.
        self.watcher = None;
        let debounce = Duration::from_millis(self.config.watch.debounce_ms);
        if let Some(root) = self.last_root.clone() {
            match Watcher::new(root, debounce) {
                Ok(w) => self.watcher = Some(w),
                Err(e) => self.push_runtime_error(GruffError::WatcherStartup {
                    message: e.to_string(),
                }),
            }
        }

        self.set_status_after_index(start.elapsed());
    }

    /// Flip the "include test files" toggle. Rescans the current folder so
    /// the test nodes (re)appear or disappear immediately — same end state
    /// as pressing Cmd+R after editing the config.
    fn set_include_tests(&mut self, include: bool) {
        if self.include_tests == include {
            return;
        }
        self.include_tests = include;
        let Some(indexer) = self.indexer.as_mut() else {
            // No folder loaded yet — nothing to re-index. The preference
            // still persists so the next `load_folder` picks it up.
            return;
        };
        let start = Instant::now();
        indexer.set_include_tests(include);
        self.graph = indexer.graph.clone();
        self.unresolved_dynamic = indexer.unresolved_dynamic;
        self.rebuild_derived_indexes();
        self.layout.sync(&self.graph);
        self.frame_request = None;
        self.highlight = None;
        // A node that was just filtered out of the graph can't stay
        // selected — clear rather than render a dangling sidebar.
        if let Some(selected) = self.selected.clone() {
            if !self.graph.nodes.contains_key(&selected) {
                self.selected = None;
            }
        }
        self.set_status_after_index(start.elapsed());
    }

    /// Force a full re-scan of the currently-loaded folder. The recovery
    /// path for any drift between the live graph and disk (mirrors Cmd+R).
    fn rescan_folder(&mut self) {
        let Some(root) = self.last_root.clone() else {
            return;
        };
        let start = Instant::now();
        if let Some(indexer) = self.indexer.as_mut() {
            indexer.rescan();
            self.graph = indexer.graph.clone();
            self.unresolved_dynamic = indexer.unresolved_dynamic;
        } else {
            // No indexer yet (shouldn't happen with a non-empty last_root),
            // but fall back to a fresh build rather than a silent no-op.
            self.load_folder(root);
            return;
        }
        self.rebuild_derived_indexes();
        self.layout.sync(&self.graph);
        self.frame_request = None;
        self.highlight = None;
        self.selected = None;
        self.set_status_after_index(start.elapsed());
    }

    /// Rebuild adjacency lists, package color indices, and cycles from the
    /// current `self.graph`. Called after full scans and after every
    /// non-empty incremental diff.
    fn rebuild_derived_indexes(&mut self) {
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

        self.cycles = self.graph.cycles();
        for cycle in &mut self.cycles {
            cycle.sort();
        }
        self.cycles
            .sort_by(|a, b| a.first().cmp(&b.first()).then(a.len().cmp(&b.len())));
        self.cycle_of.clear();
        for (idx, cycle) in self.cycles.iter().enumerate() {
            for node in cycle {
                self.cycle_of.insert(node.clone(), idx);
            }
        }
    }

    fn set_status_after_index(&mut self, elapsed: Duration) {
        let mut status = format!(
            "{} files, {} edges, {} cycle{} — indexed in {:.2}s",
            self.graph.nodes.len(),
            self.graph.edges.len(),
            self.cycles.len(),
            if self.cycles.len() == 1 { "" } else { "s" },
            elapsed.as_secs_f32(),
        );
        if self.unresolved_dynamic > 0 {
            status.push_str(&format!(
                "  ·  {} unresolved dynamic import{}",
                self.unresolved_dynamic,
                if self.unresolved_dynamic == 1 {
                    ""
                } else {
                    "s"
                },
            ));
        }
        self.status = status;
    }

    fn push_runtime_error(&mut self, error: GruffError) {
        const MAX_STATUS_ERRORS: usize = 32;

        if self.runtime_errors.len() == MAX_STATUS_ERRORS {
            self.runtime_errors.pop_front();
        }
        self.runtime_errors.push_back(error);
    }

    fn poll_runtime_errors(&mut self) {
        for error in error::drain_runtime_errors() {
            self.push_runtime_error(error);
        }
    }

    fn current_status_errors(&self) -> Vec<GruffError> {
        let mut errors = self
            .indexer
            .as_ref()
            .map(Indexer::current_errors)
            .unwrap_or_default();
        errors.extend(self.runtime_errors.iter().cloned());
        errors
    }

    /// Write the current graph out as JSON at a user-picked path. Backs the
    /// "Export graph as JSON…" menu item. Errors from the save dialog (user
    /// cancelled) are silent; write failures surface in the status bar.
    fn export_graph_json(&mut self) {
        let default_name = self
            .last_root
            .as_deref()
            .and_then(|p| p.file_name())
            .map(|n| format!("{}.graph.json", n.to_string_lossy()))
            .unwrap_or_else(|| "graph.json".to_string());

        let Some(target) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };

        let exported = export::build_export(&self.graph, &self.cycles, self.last_root.as_deref());
        match export::write_json(&target, &exported) {
            Ok(()) => {
                self.status = format!(
                    "Exported {} nodes, {} edges, {} cycle{} to {}",
                    exported.nodes.len(),
                    exported.edges.len(),
                    exported.cycles.len(),
                    if exported.cycles.len() == 1 { "" } else { "s" },
                    target.display(),
                );
            }
            Err(e) => {
                self.status = format!("Failed to export: {e}");
            }
        }
    }

    /// Drain every debounced watcher event and apply the resulting diffs to
    /// the indexer and the live graph. Returns true if any diff was applied,
    /// so callers can decide to refresh derived indexes / request a repaint.
    fn pump_watcher(&mut self) -> bool {
        let Some(watcher) = self.watcher.as_ref() else {
            return false;
        };
        let events = watcher.drain();
        if events.is_empty() {
            return false;
        }
        let Some(indexer) = self.indexer.as_mut() else {
            return false;
        };

        let mut combined = GraphDiff::default();
        for event in events {
            let diff = match event {
                ChangeEvent::Touched(path) => indexer.update_file(&path),
                ChangeEvent::Removed(path) => indexer.remove_file(&path),
            };
            merge_diff(&mut combined, diff);
        }
        if combined.is_empty() {
            return false;
        }
        self.graph.apply(&combined);
        self.unresolved_dynamic = indexer.unresolved_dynamic;

        // Layout's `sync` preserves existing node positions and seeds new
        // ones on a spiral, so the simulation absorbs the diff in place
        // without restarting from scratch.
        self.layout.sync(&self.graph);
        self.rebuild_derived_indexes();

        // Clear selection/highlight if the affected node is gone — stale
        // state in the sidebar would point at nothing.
        if let Some(selected) = self.selected.clone() {
            if !self.graph.nodes.contains_key(&selected) {
                self.selected = None;
                self.highlight = None;
            }
        }

        self.status = format!(
            "{} files, {} edges, {} cycle{} — live",
            self.graph.nodes.len(),
            self.graph.edges.len(),
            self.cycles.len(),
            if self.cycles.len() == 1 { "" } else { "s" },
        );
        if self.unresolved_dynamic > 0 {
            self.status.push_str(&format!(
                "  ·  {} unresolved dynamic import{}",
                self.unresolved_dynamic,
                if self.unresolved_dynamic == 1 {
                    ""
                } else {
                    "s"
                },
            ));
        }
        true
    }
}

/// True if `path` is a directory we can list. Used to decide whether a
/// `last_repo` entry is still live enough to auto-open — a successful
/// `read_dir` implies existence, readability, and that it isn't a plain file.
fn is_readable_dir(path: &std::path::Path) -> bool {
    std::fs::read_dir(path).is_ok()
}

/// Append `src` into `dst`, preserving ordering (removals first, additions
/// last so a burst of events still applies cleanly to the graph).
fn merge_diff(dst: &mut GraphDiff, src: GraphDiff) {
    dst.removed_edges.extend(src.removed_edges);
    dst.removed_nodes.extend(src.removed_nodes);
    dst.added_nodes.extend(src.added_nodes);
    dst.added_edges.extend(src.added_edges);
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
        self.poll_runtime_errors();

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
        let search_requested = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::F));
        if search_requested && !self.graph.nodes.is_empty() {
            self.toggle_search();
        }

        // Cmd+R forces a full re-scan of the currently-loaded folder.
        // Watcher events patch the graph incrementally; Cmd+R is the
        // recovery path that reconciles any drift (files moved while the
        // watcher was paused, rename storms we missed, etc.).
        let refresh_requested = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::R));
        if refresh_requested && self.last_root.is_some() {
            self.rescan_folder();
        }

        // Drain and apply any filesystem changes since the previous frame.
        // Non-blocking — returns immediately when the repo is quiet.
        if self.pump_watcher() {
            ctx.request_repaint();
        }
        if self.watcher.is_some() {
            // The watcher and panic queue are polled from the UI thread.
            // Keep a light heartbeat even when physics is paused.
            ctx.request_repaint_after(Duration::from_millis(100));
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
                    // Only enable export when there's an indexed graph — an
                    // empty export would be misleading rather than useful.
                    let can_export = !self.graph.nodes.is_empty();
                    if ui
                        .add_enabled(can_export, egui::Button::new("Export graph as JSON…"))
                        .clicked()
                    {
                        ui.close();
                        self.export_graph_json();
                    }
                    ui.separator();
                    if ui.button("Reveal config file").clicked() {
                        ui.close();
                        self.reveal_config_file();
                    }
                });
                ui.menu_button("View", |ui| {
                    // Checkbox-style menu item so the current state is
                    // always visible at a glance; flipping it triggers an
                    // immediate rescan via `set_include_tests`.
                    let mut include_tests = self.include_tests;
                    if ui
                        .checkbox(&mut include_tests, "Include test files")
                        .changed()
                    {
                        self.set_include_tests(include_tests);
                    }
                });
            });
        });

        egui::Panel::bottom("status_bar")
            .exact_size(28.0)
            .show_inside(ui, |ui| {
                self.draw_status_bar(ui);
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
