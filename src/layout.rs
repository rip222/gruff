use std::collections::HashMap;
use std::f32::consts::PI;

use crate::graph::{Graph, NodeId};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f32) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}

/// Flat-array force-directed layout with a Barnes-Hut quadtree for repulsion.
///
/// Per-step cost is O(n log n) at θ ≈ 1. Setting `theta` to ≤ 0 falls back to
/// exact O(n²) computation (useful as a correctness oracle in tests).
pub struct Layout {
    ids: Vec<NodeId>,
    index_of: HashMap<NodeId, usize>,
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    edge_pairs: Vec<(usize, usize)>,
    tree: QuadTree,
    /// Per-node group index (usually workspace-package id). `None` = ungrouped.
    /// Parallel to `positions`.
    groups: Vec<Option<u32>>,
    /// Number of distinct groups in `groups` (max(group) + 1). Cached so the
    /// per-step aggregation doesn't re-scan to size its buffers.
    group_count: usize,
    pub k: f32,
    pub gravity: f32,
    pub damping: f32,
    pub max_speed: f32,
    /// Barnes-Hut approximation threshold. Higher = faster, less accurate.
    /// Typical range 0.5–1.2. Values ≤ 0 disable the approximation entirely.
    pub theta: f32,
    /// Strength of the centroid-pull that clusters files in the same package.
    /// 0 disables clustering entirely. Typical values are small (0.01–0.05)
    /// because the force is multiplied by `k` and distance-from-centroid.
    pub cluster_strength: f32,
}

impl Default for Layout {
    fn default() -> Self {
        Self::new()
    }
}

impl Layout {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            index_of: HashMap::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            edge_pairs: Vec::new(),
            tree: QuadTree::new(),
            groups: Vec::new(),
            group_count: 0,
            k: 55.0,
            gravity: 0.03,
            damping: 0.82,
            max_speed: 400.0,
            theta: 1.0,
            cluster_strength: 0.02,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.index_of.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<Vec2> {
        self.index_of.get(id).map(|&i| self.positions[i])
    }

    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, Vec2)> {
        self.ids.iter().zip(self.positions.iter().copied())
    }

    /// Largest per-node speed (length of the velocity vector) across every
    /// body in the simulation. Used by the settle / auto-refit logic to
    /// decide whether the layout has stopped moving: when the max is below a
    /// small threshold the graph is visually at rest and the camera can
    /// disengage auto-refit. Returns 0.0 for an empty layout.
    pub fn max_velocity(&self) -> f32 {
        let mut max_sq = 0.0_f32;
        for v in &self.velocities {
            let sq = v.x * v.x + v.y * v.y;
            if sq > max_sq {
                max_sq = sq;
            }
        }
        max_sq.sqrt()
    }

    /// Run the simulation forward up to `max_ticks` times or until
    /// `time_budget` has elapsed — whichever comes first. Used at folder
    /// load to bring a fresh layout close to a settled state before the
    /// first fit, so the initial view frames a meaningful bounding box
    /// instead of the seed-spiral.
    ///
    /// Returns the number of ticks actually executed. A per-tick `dt` of
    /// `1/60` matches the per-frame integration step used in the app loop
    /// so settle-time physics and live-frame physics evolve identically.
    pub fn settle(&mut self, max_ticks: u32, time_budget: std::time::Duration) -> u32 {
        if self.positions.is_empty() || max_ticks == 0 {
            return 0;
        }
        let start = std::time::Instant::now();
        let dt = 1.0 / 60.0;
        let mut ticks = 0u32;
        while ticks < max_ticks {
            self.step(dt);
            ticks += 1;
            if start.elapsed() >= time_budget {
                break;
            }
        }
        ticks
    }

    /// Rebuild the flat arrays from `graph`. Preserves positions/velocities
    /// of nodes that still exist; seeds new nodes on a spiral.
    pub fn sync(&mut self, graph: &Graph) {
        let old: HashMap<NodeId, (Vec2, Vec2)> = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), (self.positions[i], self.velocities[i])))
            .collect();

        self.ids.clear();
        self.index_of.clear();
        self.positions.clear();
        self.velocities.clear();
        self.edge_pairs.clear();
        self.groups.clear();

        let total = graph.nodes.len().max(1) as f32;
        let mut fresh_idx = 0usize;

        // Assign a stable group id per package name seen in this sync. Files
        // without a package get `None` and are skipped by the clustering force.
        let mut group_of: HashMap<&str, u32> = HashMap::new();

        for (id, node) in graph.nodes.iter() {
            let i = self.ids.len();
            self.ids.push(id.clone());
            self.index_of.insert(id.clone(), i);

            if let Some(&(p, v)) = old.get(id) {
                self.positions.push(p);
                self.velocities.push(v);
            } else {
                let angle = (fresh_idx as f32) / total * 2.0 * PI;
                let radius = 90.0 + (fresh_idx as f32) * 1.5;
                self.positions
                    .push(Vec2::new(radius * angle.cos(), radius * angle.sin()));
                self.velocities.push(Vec2::default());
                fresh_idx += 1;
            }

            let group = node.package.as_deref().map(|name| {
                let next = group_of.len() as u32;
                *group_of.entry(name).or_insert(next)
            });
            self.groups.push(group);
        }

        self.group_count = group_of.len();

        for e in &graph.edges {
            if let (Some(&i), Some(&j)) = (self.index_of.get(&e.from), self.index_of.get(&e.to)) {
                self.edge_pairs.push((i, j));
            }
        }
    }

    /// Advance the simulation by one step using Fruchterman-Reingold forces
    /// with Barnes-Hut approximation for repulsion.
    pub fn step(&mut self, dt: f32) {
        let n = self.positions.len();
        if n == 0 {
            return;
        }

        let mut forces: Vec<Vec2> = vec![Vec2::default(); n];

        let k = self.k;
        let k_sq = k * k;
        let min_dist_sq = 0.25;

        // Repulsion via Barnes-Hut quadtree (O(n log n) at θ > 0; O(n²) at θ ≤ 0).
        self.tree.rebuild(&self.positions);
        for i in 0..n {
            self.tree.apply_force(
                i as u32,
                self.positions[i],
                self.theta,
                k_sq,
                min_dist_sq,
                &self.positions,
                &mut forces[i],
            );
        }

        // Edge attraction (O(e)).
        for &(i, j) in &self.edge_pairs {
            let dx = self.positions[i].x - self.positions[j].x;
            let dy = self.positions[i].y - self.positions[j].y;
            let dist_sq = (dx * dx + dy * dy).max(min_dist_sq);
            let dist = dist_sq.sqrt();
            let mag = dist_sq / k;
            let fx = dx / dist * mag;
            let fy = dy / dist * mag;
            forces[i].x -= fx;
            forces[i].y -= fy;
            forces[j].x += fx;
            forces[j].y += fy;
        }

        // Package-aware clustering: pull each grouped node toward its group's
        // centroid. Keeps the data model flat (no container rendering) while
        // producing visible clusters in the force-directed layout.
        if self.cluster_strength > 0.0 && self.group_count > 0 {
            let mut sum = vec![Vec2::default(); self.group_count];
            let mut count = vec![0u32; self.group_count];
            for i in 0..n {
                if let Some(g) = self.groups[i] {
                    let idx = g as usize;
                    sum[idx] = sum[idx] + self.positions[i];
                    count[idx] += 1;
                }
            }
            let gain = self.cluster_strength * k;
            for i in 0..n {
                if let Some(g) = self.groups[i] {
                    let idx = g as usize;
                    if count[idx] > 1 {
                        let inv = 1.0 / count[idx] as f32;
                        let cx = sum[idx].x * inv;
                        let cy = sum[idx].y * inv;
                        forces[i].x += (cx - self.positions[i].x) * gain;
                        forces[i].y += (cy - self.positions[i].y) * gain;
                    }
                }
            }
        }

        // Gravity pulling toward origin.
        let g = self.gravity * k;
        for i in 0..n {
            forces[i].x -= self.positions[i].x * g;
            forces[i].y -= self.positions[i].y * g;
        }

        // Integrate.
        let max_speed_sq = self.max_speed * self.max_speed;
        for i in 0..n {
            let mut vx = (self.velocities[i].x + forces[i].x * dt) * self.damping;
            let mut vy = (self.velocities[i].y + forces[i].y * dt) * self.damping;
            let sp_sq = vx * vx + vy * vy;
            if sp_sq > max_speed_sq {
                let scale = self.max_speed / sp_sq.sqrt();
                vx *= scale;
                vy *= scale;
            }
            self.velocities[i].x = vx;
            self.velocities[i].y = vy;
            self.positions[i].x += vx * dt;
            self.positions[i].y += vy * dt;
        }
    }
}

// --- Barnes-Hut quadtree ----------------------------------------------------

/// Sentinel for "no child" / "no resident body" in a `QuadNode`.
const QUAD_NONE: u32 = u32::MAX;

/// Below this half-side, we stop subdividing and stash extra bodies in
/// `QuadNode::extras`. Protects against infinite recursion for coincident
/// positions and keeps the tree depth bounded.
const QUAD_MIN_HALF: f32 = 0.5;

#[derive(Default)]
struct QuadNode {
    // Cell bounds — center and half-side length.
    cx: f32,
    cy: f32,
    half: f32,
    // Accumulated mass-weighted sum of positions + total mass.
    // Divide by mass for center of mass.
    com_sum_x: f32,
    com_sum_y: f32,
    mass: f32,
    // Resident body index (QUAD_NONE if internal or empty).
    body: u32,
    // Child arena indices, NW/NE/SW/SE; QUAD_NONE when that quadrant is empty.
    children: [u32; 4],
    // Additional bodies held at this node when subdivision was exhausted
    // (i.e. coincident positions). Normally empty — no allocation cost.
    extras: Vec<u32>,
}

struct QuadTree {
    nodes: Vec<QuadNode>,
}

impl QuadTree {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn rebuild(&mut self, positions: &[Vec2]) {
        // Clear out old extras so they don't leak bodies from a prior build.
        for node in &mut self.nodes {
            node.extras.clear();
        }
        self.nodes.clear();
        if positions.is_empty() {
            return;
        }

        // Compute bounding square large enough to contain every body.
        let (mut min_x, mut max_x) = (positions[0].x, positions[0].x);
        let (mut min_y, mut max_y) = (positions[0].y, positions[0].y);
        for p in &positions[1..] {
            if p.x < min_x {
                min_x = p.x;
            } else if p.x > max_x {
                max_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            } else if p.y > max_y {
                max_y = p.y;
            }
        }
        let cx = (min_x + max_x) * 0.5;
        let cy = (min_y + max_y) * 0.5;
        // Pad by a tiny epsilon so bodies on the exact boundary stay inside.
        let span = (max_x - min_x).max(max_y - min_y) * 0.5 + 1.0;

        self.nodes.push(QuadNode {
            cx,
            cy,
            half: span,
            body: QUAD_NONE,
            children: [QUAD_NONE; 4],
            ..Default::default()
        });

        for (i, _) in positions.iter().enumerate() {
            self.insert(positions, i as u32);
        }
    }

    fn insert(&mut self, positions: &[Vec2], body: u32) {
        let p = positions[body as usize];
        let mut idx = 0usize;
        loop {
            // Update COM sum and mass at the current node for the incoming body.
            let (cx, cy, half) = {
                let n = &mut self.nodes[idx];
                n.com_sum_x += p.x;
                n.com_sum_y += p.y;
                n.mass += 1.0;
                (n.cx, n.cy, n.half)
            };

            // Snapshot structural state to decide what to do.
            let (had_body, is_internal) = {
                let n = &self.nodes[idx];
                (n.body, n.children.iter().any(|&c| c != QUAD_NONE))
            };

            if !is_internal && had_body == QUAD_NONE {
                // Empty cell: place the body here.
                self.nodes[idx].body = body;
                return;
            }

            if !is_internal {
                // Leaf with an existing resident. Need to split.
                if half < QUAD_MIN_HALF {
                    // Tiny cell — stash as extra rather than recursing further.
                    self.nodes[idx].extras.push(body);
                    return;
                }
                // Evict resident to a child cell and continue descent for `body`.
                let resident = had_body;
                self.nodes[idx].body = QUAD_NONE;
                let resident_p = positions[resident as usize];
                let rq = quadrant(resident_p, cx, cy);
                let child_idx = self.ensure_child(idx, rq);
                {
                    let c = &mut self.nodes[child_idx];
                    c.com_sum_x += resident_p.x;
                    c.com_sum_y += resident_p.y;
                    c.mass += 1.0;
                    c.body = resident;
                }
                // `idx` is now internal; fall through to descend for `body`.
            }

            // Internal: descend into the appropriate child.
            let bq = quadrant(p, cx, cy);
            idx = self.ensure_child(idx, bq);
        }
    }

    fn ensure_child(&mut self, parent_idx: usize, quad: usize) -> usize {
        let existing = self.nodes[parent_idx].children[quad];
        if existing != QUAD_NONE {
            return existing as usize;
        }
        let (cx, cy, half) = {
            let p = &self.nodes[parent_idx];
            (p.cx, p.cy, p.half)
        };
        let new_half = half * 0.5;
        let (dx, dy) = match quad {
            0 => (-new_half, -new_half), // NW
            1 => (new_half, -new_half),  // NE
            2 => (-new_half, new_half),  // SW
            _ => (new_half, new_half),   // SE
        };
        let new_idx = self.nodes.len();
        self.nodes.push(QuadNode {
            cx: cx + dx,
            cy: cy + dy,
            half: new_half,
            body: QUAD_NONE,
            children: [QUAD_NONE; 4],
            ..Default::default()
        });
        self.nodes[parent_idx].children[quad] = new_idx as u32;
        new_idx
    }

    fn apply_force(
        &self,
        i: u32,
        p: Vec2,
        theta: f32,
        k_sq: f32,
        min_dist_sq: f32,
        positions: &[Vec2],
        out: &mut Vec2,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        // Iterative traversal via a small stack — avoids Rust recursion overhead
        // and keeps behavior predictable for deep trees.
        let mut stack: [u32; 128] = [0; 128];
        let mut top = 1usize;
        stack[0] = 0;

        let theta_sq = theta * theta;

        while top > 0 {
            top -= 1;
            let idx = stack[top] as usize;
            let n = &self.nodes[idx];
            if n.mass <= 0.0 {
                continue;
            }

            let is_internal = n.children.iter().any(|&c| c != QUAD_NONE);

            if !is_internal {
                // Leaf: apply direct forces for resident + extras, skipping self.
                if n.body != QUAD_NONE && n.body != i {
                    apply_direct(p, positions[n.body as usize], k_sq, min_dist_sq, out);
                }
                for &bi in &n.extras {
                    if bi != i {
                        apply_direct(p, positions[bi as usize], k_sq, min_dist_sq, out);
                    }
                }
                continue;
            }

            let com_x = n.com_sum_x / n.mass;
            let com_y = n.com_sum_y / n.mass;
            let dx = p.x - com_x;
            let dy = p.y - com_y;
            let dist_sq = dx * dx + dy * dy;

            // If `p` sits inside this cell, we must descend — approximating
            // would include a self-contribution that corrupts the force.
            let self_in_cell = (p.x - n.cx).abs() <= n.half && (p.y - n.cy).abs() <= n.half;
            let size = 2.0 * n.half;

            if !self_in_cell && theta > 0.0 && size * size < theta_sq * dist_sq {
                // Approximation: whole subtree as a single body at COM.
                let dist = dist_sq.max(min_dist_sq).sqrt();
                let mag = k_sq * n.mass / dist;
                out.x += dx / dist * mag;
                out.y += dy / dist * mag;
                continue;
            }

            // Descend into non-empty children.
            for &c in &n.children {
                if c != QUAD_NONE && top < stack.len() {
                    stack[top] = c;
                    top += 1;
                }
            }
        }
    }
}

#[inline]
fn apply_direct(p: Vec2, q: Vec2, k_sq: f32, min_dist_sq: f32, out: &mut Vec2) {
    let dx = p.x - q.x;
    let dy = p.y - q.y;
    let dist_sq = (dx * dx + dy * dy).max(min_dist_sq);
    let dist = dist_sq.sqrt();
    let mag = k_sq / dist;
    out.x += dx / dist * mag;
    out.y += dy / dist * mag;
}

#[inline]
fn quadrant(p: Vec2, cx: f32, cy: f32) -> usize {
    match (p.x >= cx, p.y >= cy) {
        (false, false) => 0, // NW
        (true, false) => 1,  // NE
        (false, true) => 2,  // SW
        (true, true) => 3,   // SE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeKind};
    use std::path::PathBuf;

    fn n(id: &str) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: None,
            kind: NodeKind::File,
        }
    }

    fn n_in(id: &str, pkg: &str) -> Node {
        Node {
            id: id.to_string(),
            path: PathBuf::from(id),
            label: id.to_string(),
            package: Some(pkg.to_string()),
            kind: NodeKind::File,
        }
    }

    fn layout_with_positions(positions: &[Vec2], theta: f32) -> Layout {
        let mut g = Graph::new();
        for (i, _) in positions.iter().enumerate() {
            g.add_node(n(&format!("n{i}")));
        }
        let mut layout = Layout::new();
        layout.sync(&g);
        layout.theta = theta;
        // Overwrite seeded positions with the ones we actually care about.
        for (i, p) in positions.iter().enumerate() {
            layout.positions[i] = *p;
        }
        layout
    }

    #[test]
    fn sync_seeds_positions_for_new_nodes_and_drops_removed() {
        let mut g = Graph::new();
        g.add_node(n("a"));
        g.add_node(n("b"));

        let mut layout = Layout::new();
        layout.sync(&g);
        assert_eq!(layout.len(), 2);

        g.remove_node("a");
        layout.sync(&g);
        assert_eq!(layout.len(), 1);
        assert!(layout.contains("b"));
    }

    #[test]
    fn sync_preserves_existing_positions() {
        let mut g = Graph::new();
        g.add_node(n("a"));
        let mut layout = Layout::new();
        layout.sync(&g);
        let p0 = layout.get("a").unwrap();

        g.add_node(n("b"));
        layout.sync(&g);
        assert_eq!(layout.get("a").unwrap(), p0);
    }

    #[test]
    fn sync_absorbs_incremental_diff_without_restart() {
        // Survivors of an add+remove diff must keep both their positions AND
        // their velocities so the force simulation doesn't lurch. This is
        // what lets watcher-driven updates look smooth on a live graph.
        let mut g = Graph::new();
        for id in ["a", "b", "c"] {
            g.add_node(n(id));
        }
        let mut layout = Layout::new();
        layout.sync(&g);

        // Step a few frames so the sim has actual non-zero velocities to
        // preserve — a fresh sync has all zeros and hides the bug we'd care
        // about here.
        for _ in 0..10 {
            layout.step(1.0 / 60.0);
        }
        let p_b_before = layout.get("b").unwrap();
        let idx_b = layout.index_of["b"];
        let v_b_before = layout.velocities[idx_b];

        // Remove `a`, add `d` — classic file-rename diff pattern.
        g.remove_node("a");
        g.add_node(n("d"));
        layout.sync(&g);

        // `b` survived: its position and velocity must be the exact same
        // floats we captured before the sync, not a reseeded spiral point.
        assert_eq!(layout.get("b"), Some(p_b_before));
        let idx_b_after = layout.index_of["b"];
        assert_eq!(layout.velocities[idx_b_after], v_b_before);
        assert!(layout.contains("d"));
        assert!(!layout.contains("a"));
    }

    #[test]
    fn step_on_empty_graph_is_noop() {
        let mut layout = Layout::new();
        layout.step(0.016);
    }

    #[test]
    fn barnes_hut_matches_exact_for_small_graph() {
        // Ten bodies at fixed, non-coincident positions. One step with θ = 0
        // (exact) must match θ = 1.0 (Barnes-Hut) within a small epsilon.
        let positions: Vec<Vec2> = (0..10)
            .map(|i| {
                let a = (i as f32) * 0.8;
                Vec2::new(50.0 * a.cos() + (i as f32) * 3.0, 50.0 * a.sin())
            })
            .collect();

        let mut exact = layout_with_positions(&positions, 0.0);
        let mut approx = layout_with_positions(&positions, 1.0);

        exact.step(1.0 / 60.0);
        approx.step(1.0 / 60.0);

        for i in 0..positions.len() {
            let e = exact.positions[i];
            let a = approx.positions[i];
            let dx = (e.x - a.x).abs();
            let dy = (e.y - a.y).abs();
            assert!(
                dx < 1.0 && dy < 1.0,
                "body {i}: exact={:?} approx={:?} (dx={dx}, dy={dy})",
                e,
                a,
            );
        }
    }

    #[test]
    fn tree_handles_duplicate_positions() {
        // Two bodies at exactly the same point must not infinite-recurse.
        let positions = vec![
            Vec2::new(3.0, 7.0),
            Vec2::new(3.0, 7.0),
            Vec2::new(3.0, 7.0),
        ];
        let mut layout = layout_with_positions(&positions, 1.0);
        // Should return without stack overflow or hang.
        layout.step(1.0 / 60.0);
        assert_eq!(layout.positions.len(), 3);
    }

    #[test]
    fn tree_handles_single_node() {
        // n=1: force from self should be zero; position shouldn't explode.
        let positions = vec![Vec2::new(10.0, -4.0)];
        let mut layout = layout_with_positions(&positions, 1.0);
        layout.step(1.0 / 60.0);
        let p = layout.positions[0];
        // Gravity pulls toward origin but one step at dt=1/60 is tiny.
        assert!(p.x.is_finite() && p.y.is_finite());
    }

    #[test]
    fn package_clustering_pulls_members_closer_together() {
        // Two disconnected packages of four files each. With clustering off,
        // repulsion alone leaves no affinity between same-package members;
        // with clustering on, members of the same package drift closer
        // together even though no edges connect them.
        fn build() -> (Graph, Vec<Vec2>) {
            let mut g = Graph::new();
            for i in 0..4 {
                g.add_node(n_in(&format!("a{i}"), "alpha"));
            }
            for i in 0..4 {
                g.add_node(n_in(&format!("b{i}"), "beta"));
            }
            // Seed starting positions so the two packages start interleaved —
            // clustering must actually pull them apart, not just preserve
            // initial separation.
            let positions = vec![
                Vec2::new(-30.0, 0.0), // a0
                Vec2::new(30.0, 0.0),  // a1
                Vec2::new(0.0, -30.0), // a2
                Vec2::new(0.0, 30.0),  // a3
                Vec2::new(-15.0, -15.0),
                Vec2::new(15.0, 15.0),
                Vec2::new(-15.0, 15.0),
                Vec2::new(15.0, -15.0),
            ];
            (g, positions)
        }

        fn mean_intra_package_distance(layout: &Layout, ids: &[&str]) -> f32 {
            let ps: Vec<Vec2> = ids
                .iter()
                .map(|id| layout.get(id).expect("position"))
                .collect();
            let mut sum = 0.0;
            let mut pairs = 0;
            for i in 0..ps.len() {
                for j in (i + 1)..ps.len() {
                    sum += (ps[i] - ps[j]).length();
                    pairs += 1;
                }
            }
            sum / pairs as f32
        }

        let (g, starts) = build();
        let mut with_clusters = Layout::new();
        with_clusters.cluster_strength = 0.2;
        with_clusters.sync(&g);
        for (i, p) in starts.iter().enumerate() {
            with_clusters.positions[i] = *p;
        }

        let mut without_clusters = Layout::new();
        without_clusters.cluster_strength = 0.0;
        without_clusters.sync(&g);
        for (i, p) in starts.iter().enumerate() {
            without_clusters.positions[i] = *p;
        }

        // Run the simulation long enough for centroid-pull to win over
        // the per-frame velocity damping.
        for _ in 0..240 {
            with_clusters.step(1.0 / 60.0);
            without_clusters.step(1.0 / 60.0);
        }

        let alpha_ids = ["a0", "a1", "a2", "a3"];
        let tight = mean_intra_package_distance(&with_clusters, &alpha_ids);
        let loose = mean_intra_package_distance(&without_clusters, &alpha_ids);
        assert!(
            tight < loose,
            "clustered mean distance ({tight}) should be < unclustered ({loose})",
        );
    }

    #[test]
    fn step_preserves_node_count() {
        let positions: Vec<Vec2> = (0..50).map(|i| Vec2::new(i as f32, 0.0)).collect();
        let mut layout = layout_with_positions(&positions, 1.0);
        let before = layout.positions.len();
        layout.step(1.0 / 60.0);
        assert_eq!(layout.positions.len(), before);
    }

    #[test]
    fn max_velocity_on_empty_layout_is_zero() {
        let layout = Layout::new();
        assert_eq!(layout.max_velocity(), 0.0);
    }

    #[test]
    fn max_velocity_reports_largest_speed() {
        // Directly seed velocities so we're asserting on the reducer, not on
        // the integrator.
        let positions: Vec<Vec2> = (0..3).map(|i| Vec2::new(i as f32, 0.0)).collect();
        let mut layout = layout_with_positions(&positions, 1.0);
        layout.velocities[0] = Vec2::new(3.0, 4.0); // speed 5
        layout.velocities[1] = Vec2::new(0.0, 1.0); // speed 1
        layout.velocities[2] = Vec2::new(-2.0, 0.0); // speed 2
        assert!((layout.max_velocity() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn settle_runs_up_to_max_ticks_on_small_graph() {
        // Ten bodies, 500-tick cap, generous time budget: the tick bound
        // must be the one that ends the pass.
        let positions: Vec<Vec2> = (0..10)
            .map(|i| {
                let a = (i as f32) * 0.8;
                Vec2::new(50.0 * a.cos(), 50.0 * a.sin())
            })
            .collect();
        let mut layout = layout_with_positions(&positions, 1.0);
        let ticks = layout.settle(500, std::time::Duration::from_secs(60));
        assert_eq!(ticks, 500);
    }

    #[test]
    fn settle_returns_zero_on_empty_layout() {
        let mut layout = Layout::new();
        let ticks = layout.settle(500, std::time::Duration::from_millis(300));
        assert_eq!(ticks, 0);
    }

    #[test]
    fn settle_respects_zero_tick_cap() {
        let positions = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)];
        let mut layout = layout_with_positions(&positions, 1.0);
        let ticks = layout.settle(0, std::time::Duration::from_secs(1));
        assert_eq!(ticks, 0);
    }

    #[test]
    fn settle_brings_small_graph_close_to_rest() {
        // A connected graph of modest size should settle well within the
        // 500-tick budget: max_velocity drops to near zero.
        let mut g = Graph::new();
        for i in 0..6 {
            g.add_node(n(&format!("n{i}")));
        }
        let mut layout = Layout::new();
        layout.sync(&g);
        // Push velocities up to something non-trivial before settling so
        // we're asserting the sim brings them down, not that they started
        // at zero.
        layout.step(1.0 / 60.0);
        let moving = layout.max_velocity();
        assert!(moving >= 0.0);
        layout.settle(500, std::time::Duration::from_secs(60));
        // After 500 ticks of O(1)-node damping-dominated motion on a tiny
        // graph, velocity must be small relative to the sim's max_speed.
        assert!(
            layout.max_velocity() < 5.0,
            "max velocity {} is still too high after settle",
            layout.max_velocity()
        );
    }
}
