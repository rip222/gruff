use std::collections::HashSet;

use eframe::egui;

use crate::colors;
use crate::graph::{NodeId, NodeKind};
use crate::layout::Vec2;

use super::GruffApp;

/// Click tolerance for edge hit-testing, in screen pixels. Set wider than the
/// edge stroke width so clicks near a thin 1 px line still land — trackpads
/// and mice jitter a few pixels per click.
const EDGE_HIT_PX: f32 = 6.0;

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

    /// Find the topmost node at `screen_pos` within its visible radius.
    /// Nodes are compared in screen space so hit tolerance scales with zoom.
    /// Hit radius tracks the drawn radius (with a 6 px floor for tiny nodes)
    /// so large hubs don't swallow clicks meant for attached edges — that
    /// previously made edges entering hubs effectively unclickable.
    fn pick_node(&self, screen_pos: egui::Pos2, screen_center: egui::Pos2) -> Option<NodeId> {
        let zoom_scale = self.zoom.clamp(0.5, 2.0);
        let mut best: Option<(f32, NodeId)> = None;
        for (id, world) in self.layout.iter() {
            let p = self.world_to_screen(world, screen_center);
            let dx = p.x - screen_pos.x;
            let dy = p.y - screen_pos.y;
            let dist_sq = dx * dx + dy * dy;
            let r = self.node_render_radius(id, zoom_scale);
            // Match the drawn radius with a modest floor so 2–4 px nodes at
            // low zoom aren't impossibly small targets. No extra buffer on
            // top of the drawn radius — the buffer used to eat edge clicks.
            let hit_r = r.max(6.0);
            if dist_sq <= hit_r * hit_r {
                match &best {
                    Some((d, _)) if *d <= dist_sq => {}
                    _ => best = Some((dist_sq, id.clone())),
                }
            }
        }
        best.map(|(_, id)| id)
    }

    pub(super) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
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

            // Pointer-hand cursor over nodes and edges signals clickability
            // — essential feedback now that edges are click targets too.
            // Node check first so a node under the cursor wins regardless of
            // whether an edge also passes nearby.
            if self.pick_node(hover, center).is_some() || self.pick_edge(hover, center).is_some() {
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
            // Search dims an edge when either endpoint isn't a match —
            // otherwise the edge visually anchors to a dim node and reads
            // like a loose thread. `selected` keeps its edges lit so the
            // current selection stays informative under search.
            let search_dim = search_matches.is_some_and(|m| {
                let from_lit =
                    m.contains(&edge.from) || self.selected.as_ref() == Some(&edge.from);
                let to_lit =
                    m.contains(&edge.to) || self.selected.as_ref() == Some(&edge.to);
                !(from_lit && to_lit)
            });
            let (width, color) = if on_path {
                (2.0, base)
            } else if highlight_active || search_dim {
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
            // Same rule for search: selected stays lit so the user can
            // always see which file they had focused.
            let dim_by_highlight = highlight_active && !on_path && !is_selected;
            let dim_by_search =
                search_matches.is_some_and(|m| !m.contains(id) && !is_selected);
            let color = if dim_by_highlight || dim_by_search {
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

#[cfg(test)]
mod tests {
    use super::*;

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
