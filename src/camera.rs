//! Pure camera/viewport math.
//!
//! Owns a current view + a target view plus the per-frame tween between them.
//! Callers poke the target (via `fit` or `set_target`) and step the current
//! view toward it each frame (`step`). User interactions like pan and
//! scroll-wheel zoom mutate both views so an in-flight tween doesn't
//! snap the user's input back.
//!
//! Intentionally no egui dependency — `Vec2` is our own type so the whole
//! module is unit-testable with pure math.
//!
//! Complements and is intended to replace the inline camera math previously
//! scattered through `app/canvas.rs` (world/screen conversion, fit-to-bbox,
//! pan, zoom-around-pointer).

use crate::layout::Vec2;

/// Zoom bounds. Match the existing scroll-wheel clamp in the canvas so
/// fit/pan/zoom/tween all share one definition of "too small" / "too big".
pub const MIN_ZOOM: f32 = 0.05;
pub const MAX_ZOOM: f32 = 20.0;

/// Zoom returned by `view_fitting` when the input is degenerate (empty
/// bbox, zero-size viewport). Any finite value avoids NaN/∞ downstream; 1.0
/// matches the fresh-Camera default so a no-op fit leaves the view alone.
pub const DEGENERATE_ZOOM: f32 = 1.0;

/// A complete view: where the camera is pointed in world coordinates and
/// how many screen pixels represent one world unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub center: Vec2,
    pub zoom: f32,
}

impl View {
    pub const fn new(center: Vec2, zoom: f32) -> Self {
        Self { center, zoom }
    }
}

impl Default for View {
    fn default() -> Self {
        Self::new(Vec2::new(0.0, 0.0), 1.0)
    }
}

/// Axis-aligned world-space bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bbox {
    pub min: Vec2,
    pub max: Vec2,
}

impl Bbox {
    /// Smallest bbox covering every point. `None` for an empty iterator —
    /// callers decide whether that means "skip the fit" or "fall back".
    pub fn from_points(points: impl IntoIterator<Item = Vec2>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut bb = Self {
            min: first,
            max: first,
        };
        for p in iter {
            bb.min.x = bb.min.x.min(p.x);
            bb.min.y = bb.min.y.min(p.y);
            bb.max.x = bb.max.x.max(p.x);
            bb.max.y = bb.max.y.max(p.y);
        }
        Some(bb)
    }

    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Camera {
    current: View,
    target: View,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    pub fn new() -> Self {
        Self {
            current: View::default(),
            target: View::default(),
        }
    }

    pub fn view(&self) -> View {
        self.current
    }

    pub fn target(&self) -> View {
        self.target
    }

    pub fn center(&self) -> Vec2 {
        self.current.center
    }

    pub fn zoom(&self) -> f32 {
        self.current.zoom
    }

    /// Compute the view that frames `bbox` inside a `viewport` screen-sized
    /// rectangle with `padding` *world-space* margin on each side.
    ///
    /// Padding is in world units so a single-node bbox still gets a real
    /// margin instead of collapsing to zero — that mirrors the fallback in
    /// `frame_cycle` for self-loops.
    pub fn view_fitting(bbox: Bbox, viewport: Vec2, padding: f32) -> View {
        let center = bbox.center();
        let bbox_w = bbox.width() + padding * 2.0;
        let bbox_h = bbox.height() + padding * 2.0;

        // Degenerate: zero-size viewport or a zero-area bbox with zero
        // padding. Dividing would produce ∞/NaN; instead fall back to the
        // identity zoom, centered on the bbox center. Callers that want a
        // real fit must supply a non-zero viewport AND either a non-empty
        // bbox or non-zero padding.
        if viewport.x <= 0.0 || viewport.y <= 0.0 || bbox_w <= 0.0 || bbox_h <= 0.0 {
            return View::new(center, DEGENERATE_ZOOM);
        }

        let fit_x = viewport.x / bbox_w;
        let fit_y = viewport.y / bbox_h;
        let zoom = fit_x.min(fit_y).clamp(MIN_ZOOM, MAX_ZOOM);
        View::new(center, zoom)
    }

    /// Set the target view to fit `bbox`. Current view is untouched — call
    /// `snap_to_target` to finalize instantly, or `step` each frame to tween.
    pub fn fit(&mut self, bbox: Bbox, viewport: Vec2, padding: f32) {
        self.target = Self::view_fitting(bbox, viewport, padding);
    }

    /// Direct target override. Clamps zoom so callers that build a view by
    /// hand can't push the camera past the shared zoom bounds.
    pub fn set_target(&mut self, view: View) {
        self.target = View {
            center: view.center,
            zoom: view.zoom.clamp(MIN_ZOOM, MAX_ZOOM),
        };
    }

    /// Finish the tween immediately: current := target.
    pub fn snap_to_target(&mut self) {
        self.current = self.target;
    }

    /// Jump both current and target to an exact state. Used to reset on
    /// folder load and in tests that need a deterministic starting view.
    pub fn jump_to(&mut self, view: View) {
        let clamped = View {
            center: view.center,
            zoom: view.zoom.clamp(MIN_ZOOM, MAX_ZOOM),
        };
        self.current = clamped;
        self.target = clamped;
    }

    /// Advance the tween toward the target by `dt` seconds, with `half_life`
    /// seconds until the distance halves.
    ///
    /// Exponential decay is used deliberately: it converges asymptotically
    /// so no step ever overshoots, which matters for zoom (overshooting
    /// past 0 flips the world inside-out). At the end of the decay curve we
    /// snap exactly to target so `is_settled` stops returning false due to
    /// float drift.
    pub fn step(&mut self, dt: f32, half_life: f32) {
        if dt <= 0.0 {
            return;
        }
        let half_life = half_life.max(1e-6);
        let alpha = 1.0 - (0.5_f32).powf(dt / half_life);
        self.current.center.x += (self.target.center.x - self.current.center.x) * alpha;
        self.current.center.y += (self.target.center.y - self.current.center.y) * alpha;
        self.current.zoom += (self.target.zoom - self.current.zoom) * alpha;

        let dx = self.target.center.x - self.current.center.x;
        let dy = self.target.center.y - self.current.center.y;
        let dz = self.target.zoom - self.current.zoom;
        if dx.abs() < 1e-3 && dy.abs() < 1e-3 && dz.abs() < 1e-4 {
            self.current = self.target;
        }
    }

    pub fn is_settled(&self) -> bool {
        self.current == self.target
    }

    /// Pan by a screen-space pixel delta. Applied to current *and* target
    /// so any tween in flight doesn't fight the user's drag.
    pub fn pan_pixels(&mut self, delta: Vec2) {
        let zoom = self.current.zoom;
        if zoom.abs() < f32::EPSILON {
            return;
        }
        let dx = delta.x / zoom;
        let dy = delta.y / zoom;
        self.current.center.x -= dx;
        self.current.center.y -= dy;
        self.target.center.x -= dx;
        self.target.center.y -= dy;
    }

    /// Scroll-wheel zoom "around the cursor": the world point under
    /// `pointer_offset` (pixel distance from screen center) stays put as
    /// zoom changes. Applied to both current and target — zooming always
    /// wins against an in-flight tween.
    pub fn zoom_at_pixel_offset(&mut self, pointer_offset: Vec2, factor: f32) {
        let zoom = self.current.zoom;
        if zoom.abs() < f32::EPSILON {
            return;
        }
        let world_before_x = pointer_offset.x / zoom + self.current.center.x;
        let world_before_y = pointer_offset.y / zoom + self.current.center.y;
        let new_zoom = (zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let new_center = Vec2::new(
            world_before_x - pointer_offset.x / new_zoom,
            world_before_y - pointer_offset.y / new_zoom,
        );
        self.current = View::new(new_center, new_zoom);
        self.target = View::new(new_center, new_zoom);
    }

    /// Project a world-space point into screen-space, given the screen
    /// center (caller knows the panel rect).
    pub fn world_to_screen(&self, world: Vec2, screen_center: Vec2) -> Vec2 {
        Vec2::new(
            (world.x - self.current.center.x) * self.current.zoom + screen_center.x,
            (world.y - self.current.center.y) * self.current.zoom + screen_center.y,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_from_points_tracks_extremes() {
        let bb = Bbox::from_points([
            Vec2::new(1.0, -5.0),
            Vec2::new(-3.0, 2.0),
            Vec2::new(4.0, 4.0),
        ])
        .unwrap();
        assert_eq!(bb.min, Vec2::new(-3.0, -5.0));
        assert_eq!(bb.max, Vec2::new(4.0, 4.0));
        assert_eq!(bb.center(), Vec2::new(0.5, -0.5));
        assert!((bb.width() - 7.0).abs() < 1e-5);
        assert!((bb.height() - 9.0).abs() < 1e-5);
    }

    #[test]
    fn bbox_from_empty_iter_is_none() {
        let bb = Bbox::from_points(std::iter::empty::<Vec2>());
        assert!(bb.is_none());
    }

    #[test]
    fn fit_known_bbox_centers_and_scales() {
        // 200×100 bbox at center (50, -30), 400×400 viewport, 20 padding
        // -> bbox+pad = 240×140 -> fit zoom = min(400/240, 400/140) = 1.666...
        let bbox = Bbox {
            min: Vec2::new(-50.0, -80.0),
            max: Vec2::new(150.0, 20.0),
        };
        let v = Camera::view_fitting(bbox, Vec2::new(400.0, 400.0), 20.0);
        assert!((v.center.x - 50.0).abs() < 1e-4);
        assert!((v.center.y - (-30.0)).abs() < 1e-4);
        let expected = 400.0_f32 / 240.0;
        assert!(
            (v.zoom - expected).abs() < 1e-4,
            "got zoom {} want {}",
            v.zoom,
            expected
        );
    }

    #[test]
    fn fit_empty_bbox_falls_back_to_finite_view() {
        // Zero-area bbox + zero padding hits the degenerate branch; we
        // still get a finite, centered, non-zero zoom.
        let bbox = Bbox {
            min: Vec2::new(0.0, 0.0),
            max: Vec2::new(0.0, 0.0),
        };
        let v = Camera::view_fitting(bbox, Vec2::new(400.0, 400.0), 0.0);
        assert_eq!(v.center, Vec2::new(0.0, 0.0));
        assert_eq!(v.zoom, DEGENERATE_ZOOM);
        assert!(v.zoom.is_finite());
    }

    #[test]
    fn fit_single_node_with_padding_produces_real_view() {
        // Single-point bbox, non-zero padding -> zoom is defined entirely
        // by the padded square against the viewport. No infinity, no NaN.
        let bbox = Bbox {
            min: Vec2::new(5.0, 7.0),
            max: Vec2::new(5.0, 7.0),
        };
        let v = Camera::view_fitting(bbox, Vec2::new(300.0, 300.0), 25.0);
        assert_eq!(v.center, Vec2::new(5.0, 7.0));
        assert!(v.zoom.is_finite());
        // padded = 50x50, viewport 300x300 -> zoom 6.0
        assert!((v.zoom - 6.0).abs() < 1e-4);
    }

    #[test]
    fn fit_zero_viewport_does_not_produce_nan() {
        let bbox = Bbox {
            min: Vec2::new(-10.0, -10.0),
            max: Vec2::new(10.0, 10.0),
        };
        let v = Camera::view_fitting(bbox, Vec2::new(0.0, 400.0), 5.0);
        assert!(v.zoom.is_finite());
        assert_eq!(v.zoom, DEGENERATE_ZOOM);
    }

    #[test]
    fn fit_clamps_zoom_to_bounds() {
        // Tiny bbox in a huge viewport would otherwise produce zoom ≫ MAX.
        let bbox = Bbox {
            min: Vec2::new(0.0, 0.0),
            max: Vec2::new(0.001, 0.001),
        };
        let v = Camera::view_fitting(bbox, Vec2::new(10_000.0, 10_000.0), 0.0001);
        assert!(v.zoom <= MAX_ZOOM + 1e-4);
    }

    #[test]
    fn tween_converges_without_overshoot() {
        let mut cam = Camera::new();
        cam.jump_to(View::new(Vec2::new(0.0, 0.0), 1.0));
        cam.set_target(View::new(Vec2::new(100.0, -50.0), 4.0));
        let target = cam.target();
        let start = cam.view();

        let sign_x = (target.center.x - start.center.x).signum();
        let sign_y = (target.center.y - start.center.y).signum();
        let sign_z = (target.zoom - start.zoom).signum();

        let dt = 1.0 / 60.0;
        let half_life = 0.1;

        let mut prev_dist = distance_to_target(&cam);
        for _ in 0..120 {
            cam.step(dt, half_life);
            let v = cam.view();

            // Remaining distance per axis has the same sign as the initial
            // delta (or zero once snapped). If the sign flipped, that's an
            // overshoot.
            let rx = target.center.x - v.center.x;
            let ry = target.center.y - v.center.y;
            let rz = target.zoom - v.zoom;
            assert!(rx * sign_x >= -1e-4, "x overshoot: r={rx} sign={sign_x}");
            assert!(ry * sign_y >= -1e-4, "y overshoot: r={ry} sign={sign_y}");
            assert!(rz * sign_z >= -1e-4, "z overshoot: r={rz} sign={sign_z}");

            // Monotone convergence: distance to target never grows.
            let now_dist = distance_to_target(&cam);
            assert!(
                now_dist <= prev_dist + 1e-5,
                "distance grew: {prev_dist} -> {now_dist}",
            );
            prev_dist = now_dist;
        }
        assert!(cam.is_settled());
        assert_eq!(cam.view(), cam.target());
    }

    fn distance_to_target(cam: &Camera) -> f32 {
        let v = cam.view();
        let t = cam.target();
        (v.center.x - t.center.x).abs() + (v.center.y - t.center.y).abs() + (v.zoom - t.zoom).abs()
    }

    #[test]
    fn snap_to_target_finalizes_immediately() {
        let mut cam = Camera::new();
        cam.set_target(View::new(Vec2::new(10.0, 20.0), 2.0));
        assert!(!cam.is_settled());
        cam.snap_to_target();
        assert!(cam.is_settled());
        assert_eq!(cam.view(), cam.target());
    }

    #[test]
    fn step_on_settled_camera_is_noop() {
        let mut cam = Camera::new();
        cam.jump_to(View::new(Vec2::new(3.0, 4.0), 2.0));
        cam.step(1.0 / 60.0, 0.1);
        assert_eq!(cam.view(), View::new(Vec2::new(3.0, 4.0), 2.0));
    }

    #[test]
    fn pan_pixels_moves_both_current_and_target() {
        let mut cam = Camera::new();
        cam.jump_to(View::new(Vec2::new(0.0, 0.0), 2.0));
        cam.set_target(View::new(Vec2::new(50.0, 0.0), 2.0));
        // Drag 20 pixels right at zoom=2 -> world moves left by 10.
        cam.pan_pixels(Vec2::new(20.0, 0.0));
        assert!((cam.view().center.x - (-10.0)).abs() < 1e-4);
        // Target was (50, 0) before — pan shifts it by the same world delta.
        assert!((cam.target().center.x - 40.0).abs() < 1e-4);
    }

    #[test]
    fn zoom_at_pixel_offset_preserves_world_point_under_cursor() {
        // The world point under the pointer should not move on screen
        // across a zoom change. This is the core invariant that makes
        // scroll-wheel zoom feel right.
        let mut cam = Camera::new();
        cam.jump_to(View::new(Vec2::new(0.0, 0.0), 1.0));
        let offset = Vec2::new(120.0, -30.0);
        // Compute the world point under the cursor before zooming.
        let world_before = Vec2::new(
            offset.x / cam.zoom() + cam.center().x,
            offset.y / cam.zoom() + cam.center().y,
        );
        cam.zoom_at_pixel_offset(offset, 1.75);
        let world_after = Vec2::new(
            offset.x / cam.zoom() + cam.center().x,
            offset.y / cam.zoom() + cam.center().y,
        );
        assert!((world_before.x - world_after.x).abs() < 1e-3);
        assert!((world_before.y - world_after.y).abs() < 1e-3);
    }

    #[test]
    fn world_to_screen_round_trips_with_identity_view() {
        let cam = Camera::new();
        let s = cam.world_to_screen(Vec2::new(10.0, -4.0), Vec2::new(200.0, 100.0));
        assert_eq!(s, Vec2::new(210.0, 96.0));
    }
}
