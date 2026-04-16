use egui::Color32;

pub const BG: Color32 = Color32::from_rgb(0x0E, 0x10, 0x14);
/// Default node color when no owning package is known (stray files).
pub const NODE: Color32 = Color32::from_rgb(0x6D, 0xA7, 0xE8);
pub const EDGE: Color32 = Color32::from_rgb(0x33, 0x3B, 0x47);
/// Edges that participate in a circular dependency. Bright red so cycles
/// read at a glance on the dark background without being washed out.
pub const CYCLE_EDGE: Color32 = Color32::from_rgb(0xE2, 0x4B, 0x4B);
pub const HINT: Color32 = Color32::from_rgb(0x6E, 0x7A, 0x8B);
pub const SELECTED: Color32 = Color32::from_rgb(0xF6, 0xC1, 0x4C);
pub const SELECTED_RING: Color32 = Color32::from_rgb(0xFF, 0xE1, 0x8A);
/// Neutral color reserved for external `node_modules` nodes introduced in
/// slice #4. Kept muted so external leaves don't compete with workspace files
/// for visual attention.
pub const EXTERNAL_NODE: Color32 = Color32::from_rgb(0x78, 0x80, 0x8C);
/// Edges and nodes on the currently-highlighted dependency path. Bright cyan
/// sits well clear of cycle-red and selection-yellow so a highlighted path
/// that passes through cycles or the selected node remains readable.
pub const PATH_EDGE: Color32 = Color32::from_rgb(0x4C, 0xC9, 0xF0);

/// Deterministic distinct color for a package at position `index` in the
/// discovery order. Uses golden-ratio hue rotation so any number of packages
/// gets well-separated hues without hand-curating a palette.
pub fn package_color(index: usize) -> Color32 {
    // Golden-ratio hue stepping — neighboring indices end up far apart on the
    // color wheel, which is exactly what we want for distinguishing packages.
    const PHI_CONJ: f32 = 0.618_034;
    // Offset the sequence so package #0 doesn't start on a jarring pure red.
    let hue = (0.137 + (index as f32) * PHI_CONJ).fract();
    // Moderate saturation + mid-bright lightness reads well on the dark bg
    // without blowing out. Matches the feel of the default NODE blue.
    let (r, g, b) = hsl_to_rgb(hue, 0.55, 0.62);
    Color32::from_rgb(r, g, b)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let r = hue_to_channel(p, q, h + 1.0 / 3.0);
    let g = hue_to_channel(p, q, h);
    let b = hue_to_channel(p, q, h - 1.0 / 3.0);
    (
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn hue_to_channel(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_colors_are_distinct_for_small_index_range() {
        // Sample the first 12 package colors and confirm each is meaningfully
        // different from every other — guards against a palette collapse if
        // someone changes the hue stepping carelessly.
        let colors: Vec<_> = (0..12).map(package_color).collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                let [ra, ga, ba, _] = colors[i].to_array();
                let [rb, gb, bb, _] = colors[j].to_array();
                let d = (ra as i32 - rb as i32).abs()
                    + (ga as i32 - gb as i32).abs()
                    + (ba as i32 - bb as i32).abs();
                assert!(
                    d > 30,
                    "colors at {i} and {j} too close: {:?} vs {:?}",
                    colors[i],
                    colors[j],
                );
            }
        }
    }
}
