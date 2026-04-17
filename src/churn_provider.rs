//! Git-churn lookup for the hotspot overlay.
//!
//! The hotspot heatmap (PRD #37) scores each node as
//! `fan_in * (1 + ln(1 + churn_30d))`. Fan-in is a pure graph property — the
//! churn side is "how many times has this file been touched in the last 30
//! days", and the only portable answer is asking the user's SCM. This module
//! exposes that answer behind a trait so the app can inject a fake in tests
//! and so future ecosystems (jj / hg) can slot in without touching the app
//! flow.
//!
//! The default implementation shells to `git`:
//!
//! 1. `git rev-parse --is-inside-work-tree` (with `-C <root>`) — if this
//!    exits non-zero, the workspace isn't a git checkout and the provider
//!    returns an empty map. The scorer then degrades to fan-in-only per the
//!    PRD's "non-git fallback" decision.
//! 2. `git log --since=30.days.ago --name-only --pretty=format:` — emits one
//!    path per line, with blank separators between commits. We split on
//!    whitespace, filter blanks, and tally path appearances across the
//!    window. Each commit that touches a file counts once.
//!
//! Paths in the returned map are absolute (`root.join(relative)`) so the
//! scorer can look them up by `Node.path` without a workspace-relative
//! conversion step. A missing-from-map lookup is scored as zero-churn
//! downstream, which handles new / untracked files gracefully.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Look up per-file churn counts across a recent time window. Injected into
/// [`crate::app::GruffApp`] so tests can swap [`FakeChurnProvider`] in without
/// shelling to `git`, and so future non-git providers (jj, hg) can drop in
/// without touching the app flow.
pub trait ChurnProvider {
    /// Return `{absolute_path: commit_count}` covering the last `days`
    /// days of history for the repo rooted at `root`. A non-repo workspace
    /// must return an empty map, not an error — the scorer degrades to
    /// fan-in-only rather than blocking the overlay.
    fn churn(&self, root: &Path, days: u32) -> HashMap<PathBuf, u32>;
}

/// Default churn provider — shells to `git` via two `Command` calls. See the
/// module-level doc for the exact invocation. Stateless; construct a fresh
/// one per query.
#[derive(Debug, Default, Clone, Copy)]
pub struct GitChurn;

impl GitChurn {
    pub fn new() -> Self {
        Self
    }
}

impl ChurnProvider for GitChurn {
    fn churn(&self, root: &Path, days: u32) -> HashMap<PathBuf, u32> {
        // Gate on `git rev-parse --is-inside-work-tree` first: a non-git
        // workspace shouldn't see a `git log` invocation at all. A
        // non-zero exit or a missing `git` binary both land in the same
        // branch — empty map, scorer degrades to fan-in-only.
        let inside = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--is-inside-work-tree"])
            .output();
        let inside_ok = matches!(&inside, Ok(o) if o.status.success());
        if !inside_ok {
            return HashMap::new();
        }

        // `--name-only --pretty=format:` emits one filename per line with
        // blank lines between commits. Each filename appearance within the
        // window counts as one touch — splitting on whitespace + filtering
        // blanks handles the separator naturally.
        let since = format!("--since={days}.days.ago");
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["log", &since, "--name-only", "--pretty=format:"])
            .output();
        let Ok(output) = output else {
            return HashMap::new();
        };
        if !output.status.success() {
            return HashMap::new();
        }
        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut counts: HashMap<PathBuf, u32> = HashMap::new();
        for line in stdout.split('\n') {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Join with root so the keys line up with `Node.path` (which the
            // indexer canonicalises against the workspace root). Callers
            // don't have to do a workspace-relative conversion step.
            let absolute = root.join(trimmed);
            *counts.entry(absolute).or_insert(0) += 1;
        }
        counts
    }
}

/// Hand-seeded provider for tests. Wraps a `{path: count}` map verbatim so
/// app-level tests can exercise the toggle flow end-to-end without a real
/// `git` binary and without touching the filesystem.
#[derive(Debug, Default, Clone)]
pub struct FakeChurnProvider {
    pub counts: HashMap<PathBuf, u32>,
}

impl FakeChurnProvider {
    pub fn new(counts: HashMap<PathBuf, u32>) -> Self {
        Self { counts }
    }
}

impl ChurnProvider for FakeChurnProvider {
    fn churn(&self, _root: &Path, _days: u32) -> HashMap<PathBuf, u32> {
        self.counts.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    /// Run `git` with `args` under `cwd`, panicking on failure. Keeps the
    /// tempfile smoke tests readable — any setup failure bubbles up as a
    /// clear panic rather than a silently-truncated history.
    fn git(cwd: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .status()
            .expect("failed to spawn git");
        assert!(status.success(), "git {args:?} failed in {cwd:?}");
    }

    #[test]
    fn git_churn_counts_one_file_across_two_commits() {
        // Happy path: `git init` + two commits touching the same file →
        // churn map reports count == 2 for that file. This is the test
        // the PRD's testing-decisions doc spells out verbatim.
        let repo = tempfile::tempdir().expect("create tempdir");
        let root = repo.path();

        git(root, &["init", "-q"]);
        // Set a local identity so `git commit` doesn't fail on machines
        // without a global user.email / user.name.
        git(root, &["config", "user.email", "hotspot@example.com"]);
        git(root, &["config", "user.name", "Hotspot Test"]);
        // Default branch name varies by git version; pin one so the
        // commit-reachability test below doesn't rely on the default.
        git(root, &["checkout", "-q", "-b", "main"]);

        let file = root.join("a.txt");
        fs::write(&file, "v1").expect("write v1");
        git(root, &["add", "a.txt"]);
        git(root, &["commit", "-q", "-m", "first"]);

        fs::write(&file, "v2").expect("write v2");
        git(root, &["add", "a.txt"]);
        git(root, &["commit", "-q", "-m", "second"]);

        let counts = GitChurn.churn(root, 30);
        let got = counts.get(&file).copied().unwrap_or(0);
        assert_eq!(got, 2, "a.txt should appear in both commits; got {counts:?}");
    }

    #[test]
    fn git_churn_non_git_workspace_returns_empty() {
        // A tempdir with no `.git` subtree is a clean "not a git
        // workspace" gate — `git rev-parse --is-inside-work-tree` exits
        // non-zero and the provider returns an empty map. This backs the
        // PRD's "non-git fallback" decision.
        let dir = tempfile::tempdir().expect("create tempdir");
        let counts = GitChurn.churn(dir.path(), 30);
        assert!(
            counts.is_empty(),
            "non-git workspace should produce empty churn; got {counts:?}"
        );
    }

    #[test]
    fn fake_provider_returns_seeded_map() {
        // Injection helper for app-level tests: hand in a map, get the
        // same map back regardless of root / days.
        let mut seed = HashMap::new();
        seed.insert(PathBuf::from("/tmp/a.ts"), 5u32);
        seed.insert(PathBuf::from("/tmp/b.ts"), 1u32);
        let provider = FakeChurnProvider::new(seed.clone());

        let got = provider.churn(Path::new("/tmp"), 30);
        assert_eq!(got, seed);

        // Different root / days arguments don't change the output — the
        // fake is literally a map dump.
        let got2 = provider.churn(Path::new("/other"), 7);
        assert_eq!(got2, seed);
    }
}
