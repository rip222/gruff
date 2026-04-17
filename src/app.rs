use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::aggregation::{self, PassthroughAggregator, TsBarrelAggregator};
use crate::camera::Camera;
use crate::colors;
use crate::config::{self, Config};
use crate::error::{self, GruffError};
use crate::export;
use crate::filter_state::FilterState;
use crate::graph::{Graph, GraphDiff, NodeId};
use crate::indexer::Indexer;
use crate::layout::Layout;
use crate::lib_detect::{self, LibDetector, TsLibDetector};
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

/// How the pending "fit all visible nodes" request should land — snap
/// immediately (folder load, where there's nothing to animate from) or
/// tween smoothly (the `F` shortcut, where the user has a prior view).
#[derive(Clone, Copy, Debug, PartialEq)]
enum FitMode {
    Snap,
    Tween,
}

/// Cap on synchronous physics ticks run during `load_folder`'s settle pass.
/// Paired with [`SETTLE_TIME_BUDGET`] — whichever bound trips first ends the
/// pass. 500 ticks is enough for small-to-medium graphs to visibly settle;
/// the time budget keeps huge graphs from stalling the UI.
const SETTLE_MAX_TICKS: u32 = 500;

/// Wall-clock cap on the settle pass. Matches PRD #16's "~300 ms" target so
/// the app stays responsive on large folders — once this trips we rely on
/// auto-refit to keep framing the graph while physics continues in the
/// normal per-frame loop.
const SETTLE_TIME_BUDGET: Duration = Duration::from_millis(300);

/// Max per-node speed below which the graph is considered visually at rest.
/// Used after the load-time settle pass to decide whether to arm auto-refit
/// and each frame thereafter to decide whether to disarm it. Chosen small
/// relative to `Layout::max_speed` (≈400) so damping-driven drift doesn't
/// look "still moving" to the camera.
const SETTLED_VELOCITY: f32 = 1.0;

/// Approximate world-space width of a single glyph at the canvas's label font
/// size (see `LABEL_WORLD_FONT_SIZE` in `app/canvas.rs`). Used only for the
/// pre-render heuristic that sizes `Layout::k` — the renderer measures each
/// label exactly, but the spring constant has to be calibrated before we have
/// an `egui::Ui` in hand during `load_folder`. Proportional fonts average
/// ≈ 0.55 × font-size per glyph; we round up so the resulting `k` errs toward
/// too-much whitespace rather than too-little.
const APPROX_CHAR_WIDTH_WORLD: f32 = 7.2;

/// Additive padding on the estimated widest rect, matching the renderer's
/// `RECT_H_PADDING` × 2. Keeps the heuristic consistent with how wide rects
/// actually draw so the calibrated `k` tracks real rendered width.
const APPROX_RECT_H_PADDING_TOTAL: f32 = 12.0;

/// Additive slack on top of the largest rendered node extent when deriving
/// `Layout::k`. Edge-attraction equilibrium sits at `d ≈ k`, so without a
/// margin two max-width rects connected by an edge would settle exactly
/// touching. A small margin opens visible whitespace between them without
/// blowing the overall layout up.
const K_SPACING_MARGIN: f32 = 24.0;

/// Floor for the calibrated spring constant. On graphs whose widest label is
/// tiny (single-char filenames, empty labels), the derived `k` could collapse
/// below the pre-calibration default and make the layout feel cramped — the
/// floor pins it to the old `Layout::new` value so those graphs look identical.
const K_SPACING_FLOOR: f32 = 55.0;

pub struct GruffApp {
    graph: Graph,
    /// Mirrors the indexer's `include_tests` option so the menu checkbox has
    /// somewhere to bind without reaching into `self.indexer` through a
    /// closure. Kept in sync by [`GruffApp::set_include_tests`].
    include_tests: bool,
    /// Mirrors the indexer's `collapse_barrels` option for the same reason.
    /// Default `true` preserves the long-standing barrel-collapse behaviour;
    /// flipping the menu checkbox calls [`GruffApp::set_collapse_barrels`].
    collapse_barrels: bool,
    layout: Layout,
    imports: HashMap<NodeId, Vec<NodeId>>,
    imported_by: HashMap<NodeId, Vec<NodeId>>,
    /// Stable index per package name, used to look up the node's color. The
    /// index order is whatever the first sweep through `graph.nodes` produced —
    /// good enough for deterministic colors within a single session.
    package_indices: HashMap<String, usize>,
    /// Per-node lib-shade key: `(lib_index_within_package, lib_count_in_package)`.
    /// Populated by `rebuild_derived_indexes` from the current workspace's
    /// lib roots. Nodes without a lib assignment (no enclosing `tsconfig.json`
    /// or a single-lib package) are absent — `node_color` falls back to
    /// `package_color`, which matches the pre-#26 palette bit-identically.
    node_lib_shade: HashMap<NodeId, (usize, usize)>,
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
    /// Deferred "fit the full graph into the viewport" request — produced by
    /// folder-load (snap) and the `F` shortcut (tween), consumed by
    /// `draw_canvas` once the canvas rect is known.
    fit_request: Option<FitMode>,
    /// Currently-highlighted dependency chain (set by clicking an edge),
    /// `None` when no chain is highlighted. Selecting a node or clicking
    /// empty canvas clears it.
    highlight: Option<PathHighlight>,
    selected: Option<NodeId>,
    /// Current + target viewport transform + tween state. All camera math
    /// lives in the `camera` module; this struct just carries the state.
    camera: Camera,
    /// When true, the canvas re-runs `Camera::fit` every frame so the
    /// viewport keeps framing the whole graph while physics continues to
    /// converge after load. Flipped on at folder load when the settle pass
    /// didn't fully quiesce the sim, and flipped off the first time the
    /// user pans or zooms (either ends auto-refit for the rest of the
    /// session, or until the next folder open). See PRD #16's "Camera
    /// settle + fit flow at load" for the full state machine.
    auto_refit: bool,
    /// True once the overlap resolver has run against the current settled
    /// layout. Cleared every time the layout is re-synced (new graph, filter
    /// toggle, watcher-driven diff) so the next settle re-runs the resolver.
    /// Without this flag the resolver would run every frame the sim happened
    /// to be quiet, wastefully mutating positions that are already clean.
    overlap_resolved: bool,
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
    /// Session-scoped visibility filter. Nodes whose id is in the hide set
    /// are excluded from rendering, physics, and cycle detection. Reset to
    /// empty on every folder open per PRD #16 — the filter is memory-only
    /// and never persisted.
    filter_state: FilterState,
}

impl Default for GruffApp {
    fn default() -> Self {
        Self {
            graph: Graph::new(),
            include_tests: false,
            collapse_barrels: true,
            layout: Layout::new(),
            imports: HashMap::new(),
            imported_by: HashMap::new(),
            package_indices: HashMap::new(),
            node_lib_shade: HashMap::new(),
            cycles: Vec::new(),
            cycle_of: HashMap::new(),
            frame_request: None,
            fit_request: None,
            highlight: None,
            selected: None,
            camera: Camera::new(),
            auto_refit: false,
            overlap_resolved: false,
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
            filter_state: FilterState::new(),
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
        // Carry the toggles across folder changes so the user doesn't have
        // to re-set them every time they open a new repo.
        if indexer.options.include_tests != self.include_tests {
            indexer.set_include_tests(self.include_tests);
        }
        // Barrel collapse doesn't need an indexer rescan — it only changes
        // the aggregator. Set the flag in place; `aggregated_graph` reads
        // it on the next call.
        indexer.options.collapse_barrels = self.collapse_barrels;
        self.graph = aggregated_graph(&indexer);
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

        // Filter state is session-scoped: opening a new folder starts fresh
        // with every node visible. Must run *before* `rebuild_derived_indexes`
        // so cycles are computed against the full (unfiltered) new graph.
        self.filter_state.clear();

        self.rebuild_derived_indexes();
        self.frame_request = None;
        self.highlight = None;
        // Reset the search overlay on a new folder load — the cached match
        // set references node ids from the previous graph and would be
        // nonsensical against the new one.
        self.search = None;

        self.layout = Layout::new();
        self.layout.sync(&self.visible_graph());
        // Scale the spring constant to the widest label before settling so
        // the settle pass equilibrates at a spacing that already accounts
        // for rendered rect widths — otherwise the first view frames the
        // graph with rects stacked on one another.
        self.calibrate_layout_spring_constant();
        self.overlap_resolved = false;
        self.selected = None;
        self.camera = Camera::new();
        // Run physics synchronously up to ~500 ticks / ~300 ms so the first
        // fit frames a meaningful bounding box instead of the seed spiral.
        // On small graphs this finishes well before the budget; on large
        // ones we exit on the time bound and rely on `auto_refit` to keep
        // re-framing while physics continues to converge.
        self.layout.settle(SETTLE_MAX_TICKS, SETTLE_TIME_BUDGET);
        // Fit to the full graph on first frame after load. Snap rather than
        // tween — there's no prior view to animate from.
        self.fit_request = Some(FitMode::Snap);
        // If the graph is still visibly moving after the settle pass, keep
        // the camera refitting each frame until the user pans or zooms.
        // Above the settled threshold we opt in; below it we trust the
        // snap-fit above and leave auto-refit off.
        self.auto_refit = self.layout.max_velocity() > SETTLED_VELOCITY;
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
        self.graph = aggregated_graph(indexer);
        self.unresolved_dynamic = indexer.unresolved_dynamic;
        self.rebuild_derived_indexes();
        self.resync_layout();
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

    /// Flip the "collapse barrels into one node" toggle. No indexer rescan
    /// — barrels are an aggregation-time concern — so this just re-runs
    /// `aggregated_graph` and rebuilds derived indexes / layout. Mirrors
    /// the structure of [`Self::set_include_tests`] including the no-op
    /// guard so menu redraws don't trigger spurious rebuilds.
    fn set_collapse_barrels(&mut self, collapse: bool) {
        if self.collapse_barrels == collapse {
            return;
        }
        self.collapse_barrels = collapse;
        let Some(indexer) = self.indexer.as_mut() else {
            // No folder loaded yet — preference still persists for the
            // next `load_folder` to pick up.
            return;
        };
        let start = Instant::now();
        indexer.options.collapse_barrels = collapse;
        self.graph = aggregated_graph(indexer);
        self.unresolved_dynamic = indexer.unresolved_dynamic;
        self.rebuild_derived_indexes();
        self.resync_layout();
        self.frame_request = None;
        self.highlight = None;
        // Selection might point at a barrel display node that just
        // disappeared (or a file node that just got swallowed by a
        // barrel) — drop it so the sidebar doesn't dangle.
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
            self.graph = aggregated_graph(indexer);
            self.unresolved_dynamic = indexer.unresolved_dynamic;
        } else {
            // No indexer yet (shouldn't happen with a non-empty last_root),
            // but fall back to a fresh build rather than a silent no-op.
            self.load_folder(root);
            return;
        }
        self.rebuild_derived_indexes();
        self.resync_layout();
        self.frame_request = None;
        self.highlight = None;
        self.selected = None;
        self.set_status_after_index(start.elapsed());
    }

    /// Rebuild adjacency lists, package color indices, and cycles from the
    /// current `self.graph`. Called after full scans and after every
    /// non-empty incremental diff.
    ///
    /// Adjacency and package color indices cover the *full* graph — they back
    /// sidebar sections and the package-color map that must keep a stable
    /// identity across filter toggles. Cycles, by contrast, run against the
    /// *visible subgraph* per PRD #16's "Visibility semantics": hiding a node
    /// that bridged a cycle must drop that cycle from the sidebar, and
    /// un-hiding it must bring it back.
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

        self.rebuild_lib_shades();
        self.recompute_cycles();
    }

    /// Populate `self.node_lib_shade` by running the lib detector against the
    /// current workspace and assigning every file-kind node to its deepest
    /// enclosing lib. Packages with only one lib skip assignment entirely so
    /// `node_color` collapses to `package_color` for them (bit-identical to
    /// pre-#26 rendering, per the acceptance criteria).
    fn rebuild_lib_shades(&mut self) {
        self.node_lib_shade.clear();
        let Some(indexer) = self.indexer.as_ref() else {
            return;
        };
        let libs = TsLibDetector::new().detect(&indexer.ws);
        if libs.is_empty() {
            return;
        }
        let grouped = lib_detect::libs_by_package(&libs);

        // Global lib-idx -> (within-package idx, package's lib count).
        let mut shade_by_lib: HashMap<usize, (usize, usize)> = HashMap::new();
        for (_pkg, indices) in &grouped {
            let count = indices.len();
            if count <= 1 {
                // Single-lib packages collapse to `package_color`; leaving
                // them out of `node_lib_shade` makes that fallback automatic.
                continue;
            }
            for (within, &global) in indices.iter().enumerate() {
                shade_by_lib.insert(global, (within, count));
            }
        }

        for (id, node) in &self.graph.nodes {
            // Barrel display nodes carry the folder path; file nodes carry
            // the file path. `deepest_lib_for` handles both — `starts_with`
            // is equally valid for a file inside a lib folder and for the
            // folder itself being (or living under) the lib.
            let Some(global) = lib_detect::deepest_lib_for(&node.path, &libs) else {
                continue;
            };
            if let Some(&shade) = shade_by_lib.get(&global) {
                self.node_lib_shade.insert(id.clone(), shade);
            }
        }
    }

    /// Recompute the visible-subgraph cycle set and its reverse index.
    /// Isolated so filter toggles can refresh cycles without touching the
    /// full-graph adjacency/color indexes above.
    fn recompute_cycles(&mut self) {
        let source = if self.filter_state.is_empty() {
            // Fast path: nothing hidden, so the visible subgraph equals the
            // full graph and we can skip the clone.
            self.graph.cycles()
        } else {
            self.visible_graph().cycles()
        };
        self.cycles = source;
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

    /// Build a clone of `self.graph` with hidden nodes and incident edges
    /// removed. Used by layout-sync and cycle detection so hidden nodes are
    /// absent from both the physics sim and the cycle report. Cheap enough
    /// (one pass, no heavy data) to rebuild on every toggle; we avoid the
    /// clone entirely when the filter is empty via [`Self::recompute_cycles`]
    /// and [`Self::resync_layout`].
    fn visible_graph(&self) -> Graph {
        let mut g = Graph::new();
        for (id, node) in &self.graph.nodes {
            if self.filter_state.is_hidden(id) {
                continue;
            }
            g.add_node(node.clone());
        }
        for edge in &self.graph.edges {
            if self.filter_state.is_hidden(&edge.from) || self.filter_state.is_hidden(&edge.to) {
                continue;
            }
            g.add_edge(&edge.from, &edge.to);
        }
        g
    }

    /// Re-sync the layout against the current visible subgraph. Hidden nodes
    /// drop out of the simulation (freeing their space for remaining nodes
    /// to redistribute into); previously-hidden nodes that are unhidden get
    /// a fresh spiral seed. Uses the full graph directly when nothing is
    /// hidden to skip the clone.
    fn resync_layout(&mut self) {
        if self.filter_state.is_empty() {
            self.layout.sync(&self.graph);
        } else {
            self.layout.sync(&self.visible_graph());
        }
        self.calibrate_layout_spring_constant();
        // Any re-sync invalidates the previous "no overlap" guarantee —
        // positions just shifted (new nodes, removed nodes, or a freshly
        // re-kicked sim), so the resolver must run again on the next settle.
        self.overlap_resolved = false;
    }

    /// Scale `Layout::k` to the widest rendered node so the force-sim's
    /// equilibrium distance stays larger than any single node's rectangle.
    /// Without this, `k = 55` pulls edge-connected nodes to `d ≈ 55` world
    /// units — smaller than most label widths (`business-hours-card`,
    /// `customer-portal-step`) — so rects end up stacked at rest. Called
    /// after every `layout.sync` so new nodes / filter toggles recalibrate.
    ///
    /// Uses a character-count heuristic rather than real text measurement
    /// because it runs during `load_folder`, before the first `draw_canvas`
    /// frame has a `ui` in hand. The approximation overestimates slightly,
    /// which is the safe direction — "too much whitespace" beats "overlap."
    fn calibrate_layout_spring_constant(&mut self) {
        let mut max_chars = 0usize;
        for (id, node) in &self.graph.nodes {
            if self.filter_state.is_hidden(id) {
                continue;
            }
            let len = crate::node_label::display_label(node).chars().count();
            if len > max_chars {
                max_chars = len;
            }
        }
        let approx_extent =
            max_chars as f32 * APPROX_CHAR_WIDTH_WORLD + APPROX_RECT_H_PADDING_TOTAL;
        self.layout.k = (approx_extent + K_SPACING_MARGIN).max(K_SPACING_FLOOR);
    }

    /// React to a file-level filter toggle: drop hidden nodes from the
    /// simulation so remaining nodes redistribute, recompute cycles on the
    /// visible subgraph, and request the camera to tween to the new bbox
    /// over ~200-300 ms. Selection / highlight that refer to a now-hidden
    /// node are cleared so stale sidebar content doesn't linger.
    ///
    /// Callers must update `self.filter_state` *before* invoking this — the
    /// helper reads the post-toggle state to decide what's visible.
    fn apply_filter_change(&mut self) {
        self.resync_layout();
        self.recompute_cycles();

        // Drop stale selection / highlight pointing at a now-hidden node.
        if let Some(selected) = self.selected.clone() {
            if self.filter_state.is_hidden(&selected) {
                self.selected = None;
                self.highlight = None;
            }
        }

        // Tween the camera to the new visible-subgraph bbox. The canvas
        // consumes this request on the next frame, which is where it has
        // the rect dimensions needed for `Camera::fit`. Uses the `Tween`
        // variant so the transition reads as motion rather than a jump cut
        // — exactly the "~200-300 ms" target from PRD #16's
        // "Camera fit-on-filter-change flow".
        self.fit_request = Some(FitMode::Tween);
    }

    /// Apply a batch of file-level visibility toggles produced by the
    /// sidebar's file checkboxes. No-op for an empty batch so the sidebar
    /// can call unconditionally each frame without worrying about
    /// triggering a tween on idle frames.
    pub(super) fn toggle_file_visibility(&mut self, ids: &[NodeId]) {
        if ids.is_empty() {
            return;
        }
        for id in ids {
            self.filter_state.toggle(id);
        }
        self.apply_filter_change();
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

    /// Called when the user does something that should take the camera out
    /// of auto-refit (pan, zoom, or any future interaction where they've
    /// picked their own view). Intentionally a no-op when auto-refit isn't
    /// active so callers don't need to check. Extracted from the canvas so
    /// the behavior is unit-testable without an egui context.
    fn disable_auto_refit_on_user_interaction(&mut self) {
        self.auto_refit = false;
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

        // Export reflects the *full* graph — filtered export is explicitly
        // out of scope for PRD #16. Since `self.cycles` now tracks the
        // visible-subgraph cycles (per #22), recompute against the full
        // graph here so the exported JSON stays consistent with its nodes
        // and edges.
        let full_cycles = self.graph.cycles();
        let exported = export::build_export(&self.graph, &full_cycles, self.last_root.as_deref());
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
        // Barrel aggregation has to run over the full post-diff graph — a
        // single file change can promote a folder to a barrel (new index.ts)
        // or demote it, which the per-edge raw diff can't express on its own.
        // Re-aggregating is cheap relative to the parse work the indexer just
        // did, and `Layout::sync` below preserves positions for nodes whose
        // id didn't change so the view stays stable.
        self.graph = aggregated_graph(indexer);
        self.unresolved_dynamic = indexer.unresolved_dynamic;

        // Layout's `sync` preserves existing node positions and seeds new
        // ones on a spiral, so the simulation absorbs the diff in place
        // without restarting from scratch. Watcher-driven updates preserve
        // the current filter state (resets only happen on explicit folder
        // open), so sync against the visible subgraph.
        self.resync_layout();
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

/// Run the configured aggregator over the indexer's raw graph and return
/// the display-level graph the UI actually renders. Called at every point
/// where the app refreshes its live graph from the indexer — full load,
/// rescan, test-toggle, barrel-toggle, and watcher pumps. The aggregator
/// choice follows `indexer.options.collapse_barrels`: `true` runs the
/// existing `TsBarrelAggregator`; `false` runs a passthrough so the
/// graph stays at file-level granularity (madge's shape).
fn aggregated_graph(indexer: &Indexer) -> Graph {
    let ctx = aggregation::context_from_workspace(&indexer.ws, &indexer.ctx);
    if indexer.options.collapse_barrels {
        aggregation::apply_aggregation(&indexer.graph, &ctx, &TsBarrelAggregator::new())
    } else {
        aggregation::apply_aggregation(&indexer.graph, &ctx, &PassthroughAggregator::new())
    }
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

        // Bare `F` refits the viewport to the currently-visible graph.
        // Gated on no overlays being open so typing `f` in the search box
        // doesn't pull the rug out. The `!command && !ctrl` guards keep
        // this shortcut from swallowing Cmd+F (search) or Ctrl+F.
        let fit_requested = self.search.is_none()
            && self.editor_prompt.is_none()
            && ctx.input(|i| {
                !i.modifiers.command
                    && !i.modifiers.ctrl
                    && !i.modifiers.alt
                    && i.key_pressed(egui::Key::F)
            });
        if fit_requested && !self.layout.is_empty() {
            self.fit_request = Some(FitMode::Tween);
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
                    let mut collapse_barrels = self.collapse_barrels;
                    if ui
                        .checkbox(&mut collapse_barrels, "Collapse barrels into one node")
                        .changed()
                    {
                        self.set_collapse_barrels(collapse_barrels);
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
        // selection, a detected cycle, or a non-empty graph (which drives
        // the `Packages` pane introduced for issue #21). Keeping it hidden
        // on an empty canvas preserves the onboarding-only initial screen.
        if self.selected.is_some() || !self.cycles.is_empty() || !self.graph.nodes.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_interaction_breaks_auto_refit() {
        // The core PRD #16 guarantee for this issue: the first pan or zoom
        // tears down auto-refit for the rest of the session. Drive the
        // state transition directly so the test doesn't need an egui ctx
        // or a real folder load.
        let mut app = GruffApp {
            auto_refit: true,
            ..GruffApp::default()
        };
        app.disable_auto_refit_on_user_interaction();
        assert!(!app.auto_refit);
    }

    #[test]
    fn disable_auto_refit_is_idempotent() {
        // Calling the hook while already disabled must be a no-op, so
        // frames with no interaction don't flip state spuriously.
        let mut app = GruffApp {
            auto_refit: false,
            ..GruffApp::default()
        };
        app.disable_auto_refit_on_user_interaction();
        assert!(!app.auto_refit);
    }

    #[test]
    fn fresh_app_has_auto_refit_off() {
        // Default state is "not auto-refitting" — load_folder is the only
        // place that arms it, based on post-settle velocity.
        let app = GruffApp::default();
        assert!(!app.auto_refit);
    }

    // --- Filter plumbing (#22) --------------------------------------------

    use crate::graph::{Node, NodeKind};
    use std::path::PathBuf;

    fn file_node(id: &str) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: None,
            kind: NodeKind::File,
        }
    }

    /// Seed a fresh app with `graph`. Mirrors the subset of `load_folder`
    /// that matters for filter-flow tests — graph install, derived indexes,
    /// layout sync, fresh filter — without touching the filesystem, the
    /// watcher, or the indexer.
    fn app_with_graph(graph: Graph) -> GruffApp {
        let mut app = GruffApp {
            graph,
            ..GruffApp::default()
        };
        app.filter_state.clear();
        app.rebuild_derived_indexes();
        app.resync_layout();
        app
    }

    fn abc_cycle_graph() -> Graph {
        // a -> b -> c -> a — the PRD's canonical "hide B breaks the cycle"
        // fixture.
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "a");
        g
    }

    #[test]
    fn hiding_middle_of_cycle_drops_cycle_from_sidebar() {
        // A→B→C→A with B hidden leaves A→C and C→A dangling as unrelated
        // edges once B is out — the SCC collapses and the cycles list
        // empties. Acceptance criterion from #22.
        let mut app = app_with_graph(abc_cycle_graph());
        assert_eq!(app.cycles.len(), 1, "precondition: one cycle before hide");

        app.toggle_file_visibility(&["b".to_string()]);
        assert!(app.filter_state.is_hidden("b"));
        assert!(
            app.cycles.is_empty(),
            "hiding B must drop the A->B->C->A cycle from the visible subgraph"
        );
    }

    #[test]
    fn unhiding_brings_cycle_back() {
        // Reverse of the above: toggling B off, then on, restores the
        // cycle. The filter round-trip is fully reversible in memory per
        // PRD #16's "session-scoped, reset only on new folder open".
        let mut app = app_with_graph(abc_cycle_graph());
        app.toggle_file_visibility(&["b".to_string()]);
        assert!(app.cycles.is_empty());

        app.toggle_file_visibility(&["b".to_string()]);
        assert!(!app.filter_state.is_hidden("b"));
        assert_eq!(
            app.cycles.len(),
            1,
            "unhiding B must bring the cycle back on the visible subgraph"
        );
    }

    #[test]
    fn hiding_one_of_two_disjoint_cycles_leaves_the_other() {
        // Two independent SCCs: {a,b} and {c,d,e} joined by a one-way
        // bridge. Hiding `a` kills the {a,b} cycle; {c,d,e} must survive
        // untouched.
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d", "e"] {
            g.add_node(file_node(id));
        }
        g.add_edge("a", "b");
        g.add_edge("b", "a");
        g.add_edge("b", "c"); // one-way bridge
        g.add_edge("c", "d");
        g.add_edge("d", "e");
        g.add_edge("e", "c");

        let mut app = app_with_graph(g);
        assert_eq!(app.cycles.len(), 2, "precondition: two cycles");

        app.toggle_file_visibility(&["a".to_string()]);
        assert_eq!(
            app.cycles.len(),
            1,
            "hiding `a` must leave exactly the {{c,d,e}} cycle"
        );
        let remaining: std::collections::BTreeSet<_> = app.cycles[0].iter().cloned().collect();
        let expected: std::collections::BTreeSet<_> =
            ["c", "d", "e"].iter().map(|s| s.to_string()).collect();
        assert_eq!(remaining, expected);
    }

    #[test]
    fn filter_change_requests_camera_tween() {
        // Acceptance criterion: Camera::fit tweens on every filter change.
        // The canvas consumes `fit_request` on the next frame; asserting
        // the request type here is the observable proxy for "tween fires"
        // without standing up an egui context.
        let mut app = app_with_graph(abc_cycle_graph());
        assert!(app.fit_request.is_none(), "precondition: no pending fit");
        app.toggle_file_visibility(&["b".to_string()]);
        assert_eq!(
            app.fit_request,
            Some(FitMode::Tween),
            "filter change must request a camera tween, not a snap"
        );
    }

    #[test]
    fn hidden_nodes_leave_physics_simulation() {
        // Layout must drop hidden nodes entirely so remaining nodes can
        // redistribute into the freed space. We don't run the sim; we
        // just assert the hidden node is no longer present in the layout
        // after a toggle.
        let mut app = app_with_graph(abc_cycle_graph());
        assert_eq!(app.layout.len(), 3);
        app.toggle_file_visibility(&["b".to_string()]);
        assert!(
            !app.layout.contains("b"),
            "hidden node must be absent from the physics simulation"
        );
        assert_eq!(
            app.layout.len(),
            2,
            "remaining nodes stay in the simulation"
        );
    }

    #[test]
    fn unhiding_a_node_reseeds_it_in_the_layout() {
        // Reversibility for the layout: a hidden-then-unhidden node
        // reappears in the simulation at a fresh spiral seed. No edge-
        // state checks here (that's the cycle test); just that the node
        // rejoins the flat layout arrays.
        let mut app = app_with_graph(abc_cycle_graph());
        app.toggle_file_visibility(&["b".to_string()]);
        assert!(!app.layout.contains("b"));
        app.toggle_file_visibility(&["b".to_string()]);
        assert!(
            app.layout.contains("b"),
            "unhidden node must reappear in the physics simulation"
        );
    }

    #[test]
    fn clearing_selection_when_selected_node_is_hidden() {
        // A stale selection pointing at a now-hidden node would render as
        // dangling sidebar content. The filter-change flow must drop it.
        let mut app = app_with_graph(abc_cycle_graph());
        app.selected = Some("b".to_string());
        app.toggle_file_visibility(&["b".to_string()]);
        assert!(
            app.selected.is_none(),
            "hiding the selected node must clear the selection"
        );
    }

    #[test]
    fn empty_toggle_batch_is_a_noop() {
        // An empty toggle batch (e.g. a frame with no checkbox clicks)
        // must not kick off a camera tween or disturb cycle state. Sidebar
        // calls `toggle_file_visibility` unconditionally, so the guard
        // matters for perceived smoothness.
        let mut app = app_with_graph(abc_cycle_graph());
        assert!(app.fit_request.is_none());
        app.toggle_file_visibility(&[]);
        assert!(app.fit_request.is_none());
        assert_eq!(app.cycles.len(), 1);
    }

    // --- collapse_barrels toggle (#29) ------------------------------------

    /// Build the canonical barrel-toggle workspace: a single package with
    /// `src/index.ts` + `src/util.ts` (so `src/` looks like a barrel) and
    /// a sibling `bar.ts` that imports from `./src/util`.
    fn write_barrel_toggle_fixture(root: &std::path::Path) {
        std::fs::write(root.join("package.json"), r#"{"name":"app"}"#).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/index.ts"), "export * from \"./util\";\n").unwrap();
        std::fs::write(root.join("src/util.ts"), "export const x = 1;\n").unwrap();
        std::fs::write(root.join("bar.ts"), "import { x } from \"./src/util\";\n").unwrap();
    }

    #[test]
    fn collapse_barrels_default_collapses_src_into_one_node() {
        // Default flow: `src/` collapses into a barrel display node and
        // bar.ts → src/util becomes bar.ts → barrel:src.
        let dir = tempfile::tempdir().unwrap();
        write_barrel_toggle_fixture(dir.path());

        let indexer = crate::indexer::Indexer::build(dir.path());
        assert!(indexer.options.collapse_barrels, "default must be true");
        let g = aggregated_graph(&indexer);

        // Exactly one barrel node, no separate src/index or src/util nodes.
        let barrel_count = g.nodes.values().filter(|n| n.id.starts_with("barrel:")).count();
        assert_eq!(barrel_count, 1, "expected exactly one barrel display node");
        assert!(
            g.nodes.values().all(|n| n.label != "util.ts" && n.label != "index.ts"),
            "src/util.ts and src/index.ts must not appear as separate nodes when collapsed"
        );

        // The bar.ts → src/util edge must rewrite onto bar.ts → barrel:src.
        let bar_to_barrel = g
            .edges
            .iter()
            .filter(|e| e.from.ends_with("bar.ts") && e.to.starts_with("barrel:"))
            .count();
        assert_eq!(
            bar_to_barrel, 1,
            "expected one rewritten edge from bar.ts to the barrel"
        );
    }

    #[test]
    fn collapse_barrels_off_keeps_src_files_separate() {
        // Same fixture, flag flipped: src/ stays expanded, bar.ts →
        // src/util.ts is the literal file-level edge madge would produce.
        let dir = tempfile::tempdir().unwrap();
        write_barrel_toggle_fixture(dir.path());

        let mut indexer = crate::indexer::Indexer::build(dir.path());
        indexer.options.collapse_barrels = false;
        let g = aggregated_graph(&indexer);

        // No barrel nodes — every src/* file shows up as itself.
        assert!(
            g.nodes.values().all(|n| !n.id.starts_with("barrel:")),
            "no barrel nodes when collapse_barrels = false"
        );
        let labels: std::collections::BTreeSet<_> =
            g.nodes.values().map(|n| n.label.clone()).collect();
        for expected in ["bar.ts", "index.ts", "util.ts"] {
            assert!(
                labels.contains(expected),
                "expected file node {expected} in expanded graph, got {labels:?}"
            );
        }

        // The original file-level edge bar.ts → src/util.ts must survive
        // unrewritten.
        let bar_to_util = g
            .edges
            .iter()
            .filter(|e| e.from.ends_with("bar.ts") && e.to.ends_with("src/util.ts"))
            .count();
        assert_eq!(
            bar_to_util, 1,
            "expected the literal bar.ts -> src/util.ts edge when expanded"
        );
    }
}
