use egui::Color32;

pub const BG: Color32 = Color32::from_rgb(0x0E, 0x10, 0x14);
/// Default node color when no owning package is known (stray files).
pub const NODE: Color32 = Color32::from_rgb(0x6D, 0xA7, 0xE8);
// Lifted from 0x333B47 when arrowheads landed (#28). A 6-8 px arrowhead
// triangle at the former level read as nearly-black against the dark bg —
// this shade keeps the line subdued but gives the triangle enough contrast
// to register as a direction cue.
pub const EDGE: Color32 = Color32::from_rgb(0x55, 0x5F, 0x70);
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

/// HSL lightness used by `package_color`. Extracted so `lib_color`'s one-lib
/// collapse returns a bit-identical result to `package_color` without the two
/// drifting if the palette ever retunes.
const PACKAGE_BASE_LIGHTNESS: f32 = 0.62;
/// HSL saturation shared by `package_color` and `lib_color`.
const PACKAGE_SATURATION: f32 = 0.55;

/// Deterministic distinct color for a package at position `index` in the
/// discovery order. Uses golden-ratio hue rotation so any number of packages
/// gets well-separated hues without hand-curating a palette.
pub fn package_color(index: usize) -> Color32 {
    let hue = package_hue(index);
    // Moderate saturation + mid-bright lightness reads well on the dark bg
    // without blowing out. Matches the feel of the default NODE blue.
    let (r, g, b) = hsl_to_rgb(hue, PACKAGE_SATURATION, PACKAGE_BASE_LIGHTNESS);
    Color32::from_rgb(r, g, b)
}

/// Distinct shade of a package's hue for a lib at position `lib_index` within
/// the package's lib set (size `lib_count`).
///
/// Keeps packages visually cohesive — every lib of the same package shares
/// the golden-ratio hue `package_color` would pick — while varying lightness
/// evenly across the package's libs so individual libs are distinguishable.
///
/// When `lib_count <= 1` the result is bit-identical to
/// `package_color(package_index)`: single-lib packages render unchanged, per
/// PRD #24's "packages with exactly one lib preserve the existing palette".
pub fn lib_color(package_index: usize, lib_index: usize, lib_count: usize) -> Color32 {
    if lib_count <= 1 {
        return package_color(package_index);
    }
    let hue = package_hue(package_index);
    // Evenly-spaced lightness band centered on `PACKAGE_BASE_LIGHTNESS`. The
    // band stays narrow enough that every shade keeps contrast against the
    // dark background (≥ 0.48) without washing out toward pure white (≤ 0.76).
    // `t` lands at 0 for the first lib and 1 for the last, so adjacent libs
    // always step by a perceptible amount regardless of `lib_count`.
    const LIGHTNESS_MIN: f32 = 0.48;
    const LIGHTNESS_MAX: f32 = 0.76;
    let t = (lib_index.min(lib_count - 1)) as f32 / ((lib_count - 1) as f32).max(1.0);
    let lightness = LIGHTNESS_MIN + t * (LIGHTNESS_MAX - LIGHTNESS_MIN);
    let (r, g, b) = hsl_to_rgb(hue, PACKAGE_SATURATION, lightness);
    Color32::from_rgb(r, g, b)
}

/// Shared hue function for `package_color` and `lib_color`, extracted so the
/// two functions can't drift out of sync on the hue axis.
fn package_hue(index: usize) -> f32 {
    // Golden-ratio hue stepping — neighboring indices end up far apart on the
    // color wheel, which is exactly what we want for distinguishing packages.
    const PHI_CONJ: f32 = 0.618_034;
    // Offset the sequence so package #0 doesn't start on a jarring pure red.
    (0.137 + (index as f32) * PHI_CONJ).fract()
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
    fn lib_color_collapses_to_package_color_for_single_lib() {
        // Acceptance criterion from #26: a package with `lib_count == 1` must
        // render bit-identically to `package_color(package_index)` so
        // single-lib packages don't visibly change after lib detection lands.
        for pkg in 0..8 {
            for count in [0usize, 1] {
                assert_eq!(
                    lib_color(pkg, 0, count),
                    package_color(pkg),
                    "lib_color({pkg},0,{count}) must equal package_color({pkg})",
                );
            }
        }
    }

    #[test]
    fn lib_color_shades_are_distinct_within_a_package() {
        // Multi-lib packages must give every lib a meaningfully different
        // shade — otherwise the "shade per lib" rule collapses into noise.
        for pkg in [0, 3, 7] {
            for count in [2usize, 3, 5, 8] {
                let shades: Vec<_> = (0..count).map(|i| lib_color(pkg, i, count)).collect();
                for i in 0..shades.len() {
                    for j in (i + 1)..shades.len() {
                        let [ra, ga, ba, _] = shades[i].to_array();
                        let [rb, gb, bb, _] = shades[j].to_array();
                        let d = (ra as i32 - rb as i32).abs()
                            + (ga as i32 - gb as i32).abs()
                            + (ba as i32 - bb as i32).abs();
                        assert!(
                            d > 10,
                            "shades {i} and {j} of pkg {pkg} / {count} libs too close: \
                             {:?} vs {:?}",
                            shades[i],
                            shades[j],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lib_color_is_stable_across_runs() {
        // Pure function determinism — two calls with the same inputs must
        // produce identical output. Guards against future refactors
        // accidentally introducing hashing or random state.
        for pkg in [0, 4, 11] {
            for count in [1usize, 2, 7] {
                for idx in 0..count {
                    assert_eq!(lib_color(pkg, idx, count), lib_color(pkg, idx, count));
                }
            }
        }
    }

    #[test]
    fn lib_color_lightness_stays_in_contrast_band() {
        // Every shade must stay in a lightness band that reads against the
        // dark background (≥ 0.48) without washing out (≤ 0.76). Sampling
        // every shade for a handful of lib counts is enough to catch a
        // regression that pushes an endpoint out of band.
        fn approx_lightness(c: Color32) -> f32 {
            let [r, g, b, _] = c.to_array();
            let r = r as f32 / 255.0;
            let g = g as f32 / 255.0;
            let b = b as f32 / 255.0;
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            (max + min) * 0.5
        }
        for pkg in 0..4 {
            for count in [2usize, 3, 5, 8] {
                for idx in 0..count {
                    let l = approx_lightness(lib_color(pkg, idx, count));
                    assert!(
                        (0.45..=0.80).contains(&l),
                        "lib_color({pkg},{idx},{count}) lightness {l} out of band",
                    );
                }
            }
        }
    }

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
