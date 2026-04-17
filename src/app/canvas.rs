use std::collections::{HashMap, HashSet};

use eframe::egui;

use crate::camera::Bbox;
use crate::colors;
use crate::graph::{NodeId, NodeKind};
use crate::layout::Vec2;
use crate::node_label;

use super::{FitMode, GruffApp};

/// World-space margin around a fit's bounding box. Additive (not
/// multiplicative) so a single-node or self-loop bbox still gets visible
/// breathing room instead of collapsing to zero padding.
const FIT_PADDING: f32 = 60.0;

/// Half-life of the camera tween animation in seconds. Short enough to feel
/// immediate, long enough to read as motion rather than a jump cut.
const CAMERA_TWEEN_HALF_LIFE: f32 = 0.12;

/// Base world-space font size for node labels. The on-screen size is this
/// times the camera zoom, so labels shrink when you zoom out and grow when
/// you zoom in — exactly how a label painted on the world itself would behave.
const LABEL_WORLD_FONT_SIZE: f32 = 12.0;

/// Horizontal padding inside the node rect, in world units. Leaves visual
/// breathing room between the label and the rect's rounded edge.
const RECT_H_PADDING: f32 = 6.0;

/// Vertical padding inside the node rect, in world units. Controls the
/// minimum rect height and provides the residual "more-dependents = taller"
/// growth channel — a hub's padding adds a few extra world-units of height.
const RECT_V_PADDING: f32 = 4.0;

/// Cap on the world-space bonus height a node earns from having many
/// dependents. Sqrt-scaled below this cap so a megahub (1000+ dependents)
/// doesn't dwarf every other node into visual noise.
const RECT_DEPENDENTS_BONUS_CAP: f32 = 12.0;

/// Corner radius of the node rect, in world units. Picked to read as
/// unambiguously rounded at typical zoom without softening into a blob.
const RECT_CORNER_RADIUS: f32 = 4.0;

/// Below this on-screen font size (in pixels), text would be sub-readable and
/// pile into a blur. Hide labels when the effective size drops under this —
/// the rects themselves still draw.
const LABEL_MIN_SCREEN_FONT_PX: f32 = 7.0;

/// Minimum world-space rect width when the label is empty or we can't measure
/// it. Prevents truly degenerate zero-width rects from producing an invisible
/// click target.
const RECT_MIN_WIDTH: f32 = 14.0;

/// Arrowhead length in screen pixels, measured along the edge direction.
/// Sized so the triangle reads as a direction cue at default zoom without
/// dominating a short edge between two neighboring nodes.
const ARROWHEAD_LEN_PX: f32 = 8.0;

/// Arrowhead half-width in screen pixels, measured perpendicular to the edge.
/// Narrower than `ARROWHEAD_LEN_PX` gives a slender triangle that reads as
/// "arrow" rather than "blob".
const ARROWHEAD_HALF_WIDTH_PX: f32 = 3.5;

impl GruffApp {
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
    /// canvas rect with a comfortable margin. Delegates the math to
    /// `Camera::fit` and snaps — cycle framing is an instant jump, not a
    /// tween, to match the established behavior.
    fn frame_cycle(&mut self, idx: usize, rect: egui::Rect) {
        let Some(cycle) = self.cycles.get(idx) else {
            return;
        };
        let Some(bbox) = Bbox::from_points(cycle.iter().filter_map(|id| self.layout.get(id)))
        else {
            return;
        };
        self.camera
            .fit(bbox, Vec2::new(rect.width(), rect.height()), FIT_PADDING);
        self.camera.snap_to_target();
    }

    /// Fit the camera to every visible node's bounding box. Used by the
    /// initial folder-load fit (snap), the `F` shortcut (tween), and the
    /// filter-change flow (tween). A layout with no positions is a no-op —
    /// there's nothing meaningful to fit to. Layout already omits hidden
    /// nodes after `apply_filter_change`, so iterating its positions yields
    /// the visible-subgraph bbox directly.
    fn fit_all(&mut self, rect: egui::Rect, mode: FitMode) {
        let Some(bbox) = Bbox::from_points(self.layout.iter().map(|(_, p)| p)) else {
            return;
        };
        self.camera
            .fit(bbox, Vec2::new(rect.width(), rect.height()), FIT_PADDING);
        if matches!(mode, FitMode::Snap) {
            self.camera.snap_to_target();
        }
    }

    fn world_to_screen(&self, world: Vec2, screen_center: egui::Pos2) -> egui::Pos2 {
        let s = self
            .camera
            .world_to_screen(world, Vec2::new(screen_center.x, screen_center.y));
        egui::pos2(s.x, s.y)
    }

    /// Color for an unselected node. External leaves render neutral gray so
    /// `node_modules` packages read as "not our code" at a glance; workspace
    /// files take a shade of their owning package's color that identifies
    /// which lib (folder with its own `tsconfig.json`) the node belongs to.
    /// Packages with zero or one libs fall through to `package_color`, which
    /// is bit-identical to the pre-#26 rendering.
    pub(super) fn node_color(&self, id: &NodeId) -> egui::Color32 {
        let Some(node) = self.graph.nodes.get(id) else {
            return colors::NODE;
        };
        if node.kind == NodeKind::External {
            return colors::EXTERNAL_NODE;
        }
        let Some(pkg) = node.package.as_deref() else {
            return colors::NODE;
        };
        let Some(&pkg_idx) = self.package_indices.get(pkg) else {
            return colors::NODE;
        };
        if let Some(&(lib_idx, lib_count)) = self.node_lib_shade.get(id) {
            return colors::lib_color(pkg_idx, lib_idx, lib_count);
        }
        colors::package_color(pkg_idx)
    }

    /// World-space size of a node's rounded-rect body. Width is driven by the
    /// text label (measured in world-unit font-size so the rect fits the
    /// text at any zoom). Height carries the residual "more-dependents =
    /// bigger" signal via extra world-unit vertical padding, sqrt-capped so
    /// a 1000-dependent hub doesn't dwarf ordinary nodes.
    fn node_render_size_world(&self, id: &NodeId, ui: &egui::Ui) -> Vec2 {
        let label = self
            .graph
            .nodes
            .get(id)
            .map(node_label::display_label)
            .unwrap_or_default();
        let text_width = measure_text_width(ui, &label, LABEL_WORLD_FONT_SIZE);

        let deps = self.imported_by.get(id).map(|v| v.len()).unwrap_or(0) as f32;
        let deps_bonus = (deps.sqrt() * 2.0).min(RECT_DEPENDENTS_BONUS_CAP);

        let width = (text_width + RECT_H_PADDING * 2.0).max(RECT_MIN_WIDTH);
        let height = LABEL_WORLD_FONT_SIZE + RECT_V_PADDING * 2.0 + deps_bonus;
        Vec2::new(width, height)
    }

    /// Find the topmost node at `screen_pos` whose drawn rect covers it.
    /// Hit-test is rect-based to match the new rounded-rect node shape.
    /// Ties (two overlapping rects both contain the point) resolve by
    /// smallest area, so the more specific target wins — a tiny leaf node
    /// stacked on a hub's rect still receives the click.
    fn pick_node(
        &self,
        screen_pos: egui::Pos2,
        screen_center: egui::Pos2,
        ui: &egui::Ui,
    ) -> Option<NodeId> {
        let zoom = self.camera.zoom();
        let mut best: Option<(f32, NodeId)> = None;
        for (id, world) in self.layout.iter() {
            let p = self.world_to_screen(world, screen_center);
            let size = self.node_render_size_world(id, ui);
            // Convert world-space rect dimensions to screen-space so the hit
            // area tracks what the user actually sees.
            let half_w_px = size.x * zoom * 0.5;
            let half_h_px = size.y * zoom * 0.5;
            // Floor the half-extents so a node that drew as 2–3 px at low
            // zoom isn't an impossibly small click target.
            let half_w_hit = half_w_px.max(3.0);
            let half_h_hit = half_h_px.max(3.0);
            let dx = (screen_pos.x - p.x).abs();
            let dy = (screen_pos.y - p.y).abs();
            if dx <= half_w_hit && dy <= half_h_hit {
                let area = half_w_hit * half_h_hit;
                match &best {
                    Some((a, _)) if *a <= area => {}
                    _ => best = Some((area, id.clone())),
                }
            }
        }
        best.map(|(_, id)| id)
    }

    pub(super) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let rect = ui.max_rect();
        let center = rect.center();

        // Consume any pending camera requests before drawing so the updated
        // transform applies this frame instead of lagging by one. Cycle
        // framing wins over fit-all since it's the more specific request;
        // in practice they're never both set at once.
        if let Some(idx) = self.frame_request.take() {
            self.frame_cycle(idx, rect);
        }
        if let Some(mode) = self.fit_request.take() {
            self.fit_all(rect, mode);
        }

        // Auto-refit mode: while the physics sim is still visibly moving
        // after folder load, keep re-fitting the viewport each frame so the
        // user always sees the whole graph. Once the sim quiesces we stop
        // requesting new fits (the last fit's target is still in flight and
        // the tween will land naturally). Any user pan / zoom below will
        // disable this for the rest of the session.
        if self.auto_refit {
            if self.layout.max_velocity() > super::SETTLED_VELOCITY {
                self.fit_all(rect, FitMode::Snap);
                ctx.request_repaint();
            } else {
                self.auto_refit = false;
            }
        }

        // Overlap resolution: once the simulation has come to rest, iteratively
        // separate overlapping rectangles so no two labels collide. Runs once
        // per settle — any `resync_layout` clears `overlap_resolved` so the
        // next re-kick (watcher diff, filter toggle, fresh folder) triggers
        // another pass after the sim re-settles. Positions shift as a side
        // effect; a final fit-all keeps the camera framed on the expanded
        // bounds. We deliberately don't fight the running sim — the physics
        // step above may push nodes back into overlap on the very next tick,
        // which is fine: we'll re-run once things quiet down again after the
        // next re-kick.
        if !self.overlap_resolved && self.layout.max_velocity() < super::SETTLED_VELOCITY {
            let sizes: Vec<Vec2> = self
                .layout
                .iter()
                .map(|(id, _)| self.node_render_size_world(id, ui))
                .collect();
            self.layout.resolve_overlaps(&sizes);
            self.overlap_resolved = true;
            // Graph bounds may have expanded; re-fit so the user sees the
            // post-resolution layout instead of a cropped view. Tween so the
            // motion reads as "relaxation" rather than a jump cut.
            self.fit_request = Some(FitMode::Tween);
            ctx.request_repaint();
        }

        // Advance any in-flight tween. `is_settled` short-circuits when
        // current == target so this is free on frames with no animation.
        if !self.camera.is_settled() {
            let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1);
            self.camera.step(dt, CAMERA_TWEEN_HALF_LIFE);
            ctx.request_repaint();
        }

        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_delta();
            self.camera.pan_pixels(Vec2::new(delta.x, delta.y));
            // User took the wheel — stop fighting them with auto-refit for
            // the rest of the session (or until the next folder open).
            self.disable_auto_refit_on_user_interaction();
        }

        // Click handling: a node click selects the node and highlights its
        // direct neighbours in both directions — one hop, no transitive
        // walk. Anything else (empty canvas, an edge) clears both the
        // selection and any active highlight: edges are no longer pick
        // targets per #29.
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                if let Some(node) = self.pick_node(click_pos, center, ui) {
                    self.highlight = Some(self.build_node_highlight(&node));
                    self.selected = Some(node);
                    // Every fresh selection arms the blast-radius dim per
                    // PRD #35's "dim is on-by-default when selecting"
                    // decision. A prior `B` press that toggled the dim off
                    // was scoped to the previous selection — re-clicking
                    // re-arms without the user having to press `B` again.
                    self.blast_radius_active = true;
                    self.recompute_blast_cone();
                } else {
                    self.selected = None;
                    self.highlight = None;
                    self.blast_cone = None;
                }
            }
        }

        if let Some(hover) = response.hover_pos() {
            let scroll_y = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll_y.abs() > f32::EPSILON {
                let factor = (scroll_y * 0.0015).exp();
                self.camera.zoom_at_pixel_offset(
                    Vec2::new(hover.x - center.x, hover.y - center.y),
                    factor,
                );
                // Same reasoning as pan: an explicit zoom input means the
                // user wants their view, not an auto-framed one.
                self.disable_auto_refit_on_user_interaction();
            }

            // Pointer-hand cursor over nodes signals clickability. Edges
            // aren't pick targets after #29, so they don't get the cursor
            // change — the cursor stays the default arrow over them.
            if self.pick_node(hover, center, ui).is_some() {
                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
            }
        }

        let painter = ui.painter_at(rect);
        // Dim non-highlighted elements so the highlighted chain reads clearly.
        // Alpha roughly preserves the layout shape in the background without
        // competing with the highlighted edges for attention.
        const DIM_ALPHA: f32 = 0.18;
        let highlight_active = self.highlight.is_some();
        // Search-driven dimming is independent of highlight-driven dimming;
        // a node/edge is dimmed if *either* mode says dim. `search_matches`
        // is `None` when no search is active, which short-circuits the
        // per-element check to "not dimmed by search".
        let search_matches: Option<&HashSet<NodeId>> = self
            .search
            .as_ref()
            .filter(|s| !s.input.trim().is_empty())
            .map(|s| &s.matches);
        // Blast-radius cone: when active, non-dependents are dimmed. Only
        // consulted when a selection is live AND `blast_radius_active` is on
        // (the `B` keybind flips the flag without deselecting, so the cone
        // cache stays populated but the render path skips it). `None` here
        // short-circuits the per-element check exactly like `search_matches`
        // above.
        let cone: Option<&HashSet<NodeId>> = self
            .blast_cone
            .as_ref()
            .filter(|_| self.blast_radius_active && self.selected.is_some());

        // Precompute each visible node's screen-space half-extents so the
        // edge loop can trim lines and anchor arrowheads at the same AABB
        // the node body and overlap resolver use. The label loop below
        // reuses the same map instead of re-measuring.
        let zoom = self.camera.zoom();
        let node_half_extents: HashMap<NodeId, (f32, f32)> = self
            .layout
            .iter()
            .map(|(id, _)| {
                let s = self.node_render_size_world(id, ui);
                (id.clone(), (s.x * zoom * 0.5, s.y * zoom * 0.5))
            })
            .collect();

        // Edges first so nodes overlap them. Edges with any hidden endpoint
        // drop out of rendering entirely — PRD #16's "no dashed short-
        // circuit edges" rule. `layout.get` already returns `None` for
        // hidden nodes, but checking the filter first makes the intent
        // explicit and avoids allocating a position that we'd immediately
        // throw away.
        for edge in &self.graph.edges {
            if self.filter_state.is_hidden(&edge.from) || self.filter_state.is_hidden(&edge.to) {
                continue;
            }
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
            // Search dims an edge when either endpoint isn't a match —
            // otherwise the edge visually anchors to a dim node and reads
            // like a loose thread. `selected` keeps its edges lit so the
            // current selection stays informative under search.
            let search_dim = search_matches.is_some_and(|m| {
                let from_lit = m.contains(&edge.from) || self.selected.as_ref() == Some(&edge.from);
                let to_lit = m.contains(&edge.to) || self.selected.as_ref() == Some(&edge.to);
                !(from_lit && to_lit)
            });
            // Blast-radius dim parallels the search-dim rule: an edge stays
            // lit only when both endpoints are either in the cone or the
            // selected node itself. A node-in-cone / node-out-of-cone edge
            // would visually anchor to a dim node and read as a loose
            // thread.
            let cone_dim = cone.is_some_and(|c| {
                let from_lit = c.contains(&edge.from) || self.selected.as_ref() == Some(&edge.from);
                let to_lit = c.contains(&edge.to) || self.selected.as_ref() == Some(&edge.to);
                !(from_lit && to_lit)
            });
            let (width, color) = if on_path {
                (2.0, base)
            } else if highlight_active || search_dim || cone_dim {
                (1.0, base.gamma_multiply(DIM_ALPHA))
            } else if is_cycle {
                (1.6, base)
            } else {
                (1.0, base)
            };

            // Trim the segment against each endpoint's rect AABB so the line
            // stops at the target's boundary (leaves room for the arrowhead
            // flush on the edge) and doesn't bleed out from behind the
            // source's fill. If either trim fails — tiny rect, coincident
            // centers, etc. — fall back to the raw segment rather than
            // dropping the edge entirely.
            let src_half = node_half_extents.get(&edge.from);
            let tgt_half = node_half_extents.get(&edge.to);
            let (line_start, line_end, tip) = match (src_half, tgt_half) {
                (Some(&(sw, sh)), Some(&(tw, th))) => {
                    let start = clip_ray_from_center(a, b, sw, sh).unwrap_or(a);
                    let tip = clip_ray_from_center(b, a, tw, th).unwrap_or(b);
                    (start, tip, Some(tip))
                }
                _ => (a, b, None),
            };
            painter.line_segment([line_start, line_end], egui::Stroke::new(width, color));
            if let Some(tip) = tip {
                draw_arrowhead(&painter, line_start, tip, color);
            }
        }

        // Labels render in world space — font size scales with zoom — and
        // hide entirely when they'd be sub-readable. The rect itself always
        // draws, so the graph shape stays legible at low zoom.
        let screen_font_px = LABEL_WORLD_FONT_SIZE * zoom;
        let labels_visible = screen_font_px >= LABEL_MIN_SCREEN_FONT_PX;
        let label_font = egui::FontId::proportional(screen_font_px);

        for (id, pos) in self.layout.iter() {
            let p = self.world_to_screen(pos, center);
            let (half_w_px, half_h_px) = node_half_extents
                .get(id)
                .copied()
                .unwrap_or_else(|| {
                    let s = self.node_render_size_world(id, ui);
                    (s.x * zoom * 0.5, s.y * zoom * 0.5)
                });
            let rect = egui::Rect::from_min_max(
                egui::pos2(p.x - half_w_px, p.y - half_h_px),
                egui::pos2(p.x + half_w_px, p.y + half_h_px),
            );

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
            // Same rule for search: selected stays lit so the user can
            // always see which file they had focused.
            let dim_by_highlight = highlight_active && !on_path && !is_selected;
            let dim_by_search = search_matches.is_some_and(|m| !m.contains(id) && !is_selected);
            // Blast-radius dim: nodes outside the cone are dim. Selected
            // stays lit always (it's the subject of the cone); barrel-member
            // raw ids present in the cone also count as "in cone" even if
            // they aren't rendered — see the edge rule above. Rendered
            // display nodes not in the cone get dimmed.
            let dim_by_cone =
                cone.is_some_and(|c| !c.contains(id) && !is_selected);
            let color = if dim_by_highlight || dim_by_search || dim_by_cone {
                base.gamma_multiply(DIM_ALPHA)
            } else {
                base
            };

            // Corner radius also scales with zoom so the rounded look holds
            // at any scale — a constant pixel radius would wash out to a
            // square at low zoom.
            let corner_px = (RECT_CORNER_RADIUS * zoom).clamp(1.0, 10.0);
            painter.rect_filled(rect, corner_px, color);

            if is_selected {
                // Outer ring makes the selection pop even at low zoom.
                let ring_rect = rect.expand(3.0);
                painter.rect_stroke(
                    ring_rect,
                    corner_px + 2.0,
                    egui::Stroke::new(2.0, colors::SELECTED_RING),
                    egui::StrokeKind::Outside,
                );
            }

            if labels_visible {
                let label = self
                    .graph
                    .nodes
                    .get(id)
                    .map(node_label::display_label)
                    .unwrap_or_default();
                if !label.is_empty() {
                    // Text color: foreground that reads on the dim/fill alpha
                    // combination. Dim nodes get a dim label too so the pair
                    // reads as a unit.
                    let text_color = if dim_by_highlight || dim_by_search || dim_by_cone {
                        colors::HINT.gamma_multiply(DIM_ALPHA * 2.0)
                    } else {
                        egui::Color32::from_rgb(0xF4, 0xF6, 0xF8)
                    };
                    painter.text(
                        p,
                        egui::Align2::CENTER_CENTER,
                        &label,
                        label_font.clone(),
                        text_color,
                    );
                }
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
        }
    }
}

/// Measure the rendered width of `text` at `font_size`, using the current
/// `ui`'s font stack. Returns a world-unit width when `font_size` is the
/// world-space font size — the pixel width of text rendered at that exact
/// size, which happens to equal its world width at zoom = 1 and scales
/// linearly with zoom everywhere else.
fn measure_text_width(ui: &egui::Ui, text: &str, font_size: f32) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let font_id = egui::FontId::proportional(font_size);
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_string(), font_id, egui::Color32::WHITE);
    galley.size().x
}

/// Intersect the ray from `center` toward `toward` with an axis-aligned
/// rectangle centered at `center` with half-extents `(half_w, half_h)`, and
/// return the boundary hit point. Uses the slab method: each axis yields a
/// parametric `t` at which the ray crosses the rect's near/far slab; the
/// smallest positive `t` whose perpendicular coordinate stays within the
/// opposite slab is the boundary hit. Returns `None` for degenerate input
/// (zero-length ray or zero-size rect) so the caller can fall back to the
/// raw endpoint instead of drawing a nonsense segment.
fn clip_ray_from_center(
    center: egui::Pos2,
    toward: egui::Pos2,
    half_w: f32,
    half_h: f32,
) -> Option<egui::Pos2> {
    let dx = toward.x - center.x;
    let dy = toward.y - center.y;
    if half_w <= 0.0 || half_h <= 0.0 {
        return None;
    }
    if dx.abs() < f32::EPSILON && dy.abs() < f32::EPSILON {
        return None;
    }
    // Parametric t along the ray where it crosses each slab boundary. We
    // want the smaller of |t_x|, |t_y| — whichever axis the ray exits
    // first is the axis whose slab boundary is the true hit.
    let tx = if dx.abs() > f32::EPSILON {
        half_w / dx.abs()
    } else {
        f32::INFINITY
    };
    let ty = if dy.abs() > f32::EPSILON {
        half_h / dy.abs()
    } else {
        f32::INFINITY
    };
    let t = tx.min(ty);
    Some(egui::pos2(center.x + dx * t, center.y + dy * t))
}

/// Draw a filled arrowhead triangle with its tip anchored at `tip` and its
/// base two points `ARROWHEAD_LEN_PX` back along the line from `tip` toward
/// `from`. `ARROWHEAD_HALF_WIDTH_PX` sets the spread. If the edge is shorter
/// than the arrowhead length the triangle is suppressed — drawing a
/// full-size arrow on a 3 px edge would eat the entire line.
fn draw_arrowhead(
    painter: &egui::Painter,
    from: egui::Pos2,
    tip: egui::Pos2,
    color: egui::Color32,
) {
    let dx = tip.x - from.x;
    let dy = tip.y - from.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < ARROWHEAD_LEN_PX {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    // Base center: step back from the tip along the edge direction by the
    // arrowhead length. Perpendicular axis (-uy, ux) spreads the base.
    let bx = tip.x - ux * ARROWHEAD_LEN_PX;
    let by = tip.y - uy * ARROWHEAD_LEN_PX;
    let px = -uy * ARROWHEAD_HALF_WIDTH_PX;
    let py = ux * ARROWHEAD_HALF_WIDTH_PX;
    let p0 = tip;
    let p1 = egui::pos2(bx + px, by + py);
    let p2 = egui::pos2(bx - px, by - py);
    painter.add(egui::Shape::convex_polygon(
        vec![p0, p1, p2],
        color,
        egui::Stroke::NONE,
    ));
}

