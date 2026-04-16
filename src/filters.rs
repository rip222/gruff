//! File-classification helpers — test files and build/tooling configs.
//!
//! Keeping the default exclusion lists in one module satisfies the
//! "centralized so the default set can be audited" acceptance criterion:
//! every pattern a user might wonder about is spelled out right here.
//!
//! The indexer consumes these to drop noise before nodes are ever added to
//! the graph. `.gitignore` filtering lives in the indexer's
//! `ignore::WalkBuilder`; this module covers the two things `.gitignore`
//! can't (tests are usually committed; build configs too).

use std::path::Path;

/// Config-file name prefixes. A file is treated as a config when its name
/// starts with one of these followed by a `.` — so `vite.config.ts`,
/// `jest.config.cjs`, `babel.config.js` all match their respective prefix.
///
/// Kept as a list rather than a regex so it stays grep-friendly when a user
/// asks "does Gruff skip my `xyz.config.ts`?"
pub const CONFIG_PREFIXES: &[&str] = &[
    "vite.config",
    "vitest.config",
    "jest.config",
    "rollup.config",
    "webpack.config",
    "babel.config",
    "tsup.config",
    "postcss.config",
    "tailwind.config",
    "next.config",
    "prettier.config",
    "cypress.config",
    "playwright.config",
    "svelte.config",
    "nuxt.config",
    "astro.config",
    "remix.config",
    "esbuild.config",
];

/// Exact config filenames without a common `.config.` pattern — dotfile RCs
/// and build scripts whose names don't follow the `<tool>.config.*` shape.
pub const CONFIG_FILES: &[&str] = &[
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.mjs",
    ".eslintrc.ts",
    ".prettierrc.js",
    ".prettierrc.cjs",
    ".stylelintrc.js",
    ".stylelintrc.cjs",
    ".babelrc.js",
    ".babelrc.cjs",
    "gulpfile.js",
    "gulpfile.ts",
    "gulpfile.cjs",
    "gruntfile.js",
    "Gruntfile.js",
];

/// Substrings that mark a source file as a test. A file is treated as a test
/// when its name stem (before the final extension) ends with one of these —
/// covers `foo.test.ts`, `Foo.spec.tsx`, `bar.test.js`, etc.
pub const TEST_SUFFIXES: &[&str] = &[".test", ".spec"];

/// True if `path`'s filename matches a test-file convention (`*.test.*`,
/// `*.spec.*` across ts/tsx/js/jsx/mjs/cjs). Extension is not inspected
/// separately; the indexer has already gated on source extensions.
pub fn is_test_file(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    TEST_SUFFIXES.iter().any(|sfx| stem.ends_with(sfx))
}

/// True if `path`'s filename is a known build/tooling config that should be
/// skipped from parsing. Matches [`CONFIG_FILES`] exactly, or [`CONFIG_PREFIXES`]
/// followed by `.<ext>`.
pub fn is_config_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if CONFIG_FILES.contains(&name) {
        return true;
    }
    CONFIG_PREFIXES.iter().any(|p| starts_with_segment(name, p))
}

/// True if `name` starts with `prefix` followed by `.` — so `vite.config` the
/// prefix matches `vite.config.ts` but not `vite.configs.ts`.
fn starts_with_segment(name: &str, prefix: &str) -> bool {
    name.len() > prefix.len()
        && name.starts_with(prefix)
        && name.as_bytes()[prefix.len()] == b'.'
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detects_test_files_across_extensions() {
        assert!(is_test_file(&PathBuf::from("foo.test.ts")));
        assert!(is_test_file(&PathBuf::from("Foo.test.tsx")));
        assert!(is_test_file(&PathBuf::from("bar.spec.ts")));
        assert!(is_test_file(&PathBuf::from("Baz.spec.tsx")));
        assert!(is_test_file(&PathBuf::from("foo.test.js")));
        assert!(is_test_file(&PathBuf::from("foo.test.jsx")));
        assert!(is_test_file(&PathBuf::from("foo.spec.mjs")));
    }

    #[test]
    fn detects_test_files_in_nested_paths() {
        assert!(is_test_file(&PathBuf::from(
            "src/components/Button.test.tsx"
        )));
        assert!(is_test_file(&PathBuf::from("packages/a/b/c.spec.ts")));
    }

    #[test]
    fn regular_source_files_are_not_test_files() {
        assert!(!is_test_file(&PathBuf::from("foo.ts")));
        assert!(!is_test_file(&PathBuf::from("Foo.tsx")));
        assert!(!is_test_file(&PathBuf::from("index.ts")));
        // `.d.ts` declaration files are not tests.
        assert!(!is_test_file(&PathBuf::from("types.d.ts")));
        // Just having "test" as a prefix is not enough; we anchor on `.test`.
        assert!(!is_test_file(&PathBuf::from("tests.ts")));
        assert!(!is_test_file(&PathBuf::from("testUtils.ts")));
    }

    #[test]
    fn detects_config_files_by_prefix() {
        assert!(is_config_file(&PathBuf::from("vite.config.ts")));
        assert!(is_config_file(&PathBuf::from("vite.config.js")));
        assert!(is_config_file(&PathBuf::from("jest.config.cjs")));
        assert!(is_config_file(&PathBuf::from("rollup.config.mjs")));
        assert!(is_config_file(&PathBuf::from("webpack.config.js")));
        assert!(is_config_file(&PathBuf::from("babel.config.js")));
        assert!(is_config_file(&PathBuf::from("tailwind.config.ts")));
    }

    #[test]
    fn detects_exact_config_files() {
        assert!(is_config_file(&PathBuf::from(".eslintrc.js")));
        assert!(is_config_file(&PathBuf::from(".prettierrc.cjs")));
        assert!(is_config_file(&PathBuf::from(".babelrc.js")));
        assert!(is_config_file(&PathBuf::from("gulpfile.ts")));
    }

    #[test]
    fn regular_files_are_not_configs() {
        assert!(!is_config_file(&PathBuf::from("foo.ts")));
        assert!(!is_config_file(&PathBuf::from("vite.ts")));
        // A file that only shares the prefix (no dot segment boundary) must
        // not false-match — `vite.configs.ts` is not a config.
        assert!(!is_config_file(&PathBuf::from("vite.configs.ts")));
        assert!(!is_config_file(&PathBuf::from("jest.configuration.ts")));
    }

    #[test]
    fn config_detection_uses_filename_not_full_path() {
        // A file named `vite.config.ts` is a config regardless of the
        // directory it lives in; a regular file inside a "config" directory
        // is not.
        assert!(is_config_file(&PathBuf::from("apps/web/vite.config.ts")));
        assert!(!is_config_file(&PathBuf::from("src/config/index.ts")));
    }
}
