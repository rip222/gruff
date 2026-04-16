//! Filesystem watcher on top of `notify` with debounced, gitignore-filtered
//! event delivery.
//!
//! The indexer consumes `Vec<ChangeEvent>` from `drain()` once per UI frame —
//! non-blocking, cheap even when the repo is quiet. Filtering and debouncing
//! happen in a worker thread so the UI thread never pays the cost of a
//! filesystem-event storm (e.g. `webpack` dumping chunks into `dist/`).

use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify::{
    Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher as _,
    event::{ModifyKind, RemoveKind},
};

use crate::error;
use crate::indexer::is_source_file;

/// What happened to a file after debouncing. The watcher coalesces multiple
/// raw `notify` events per path inside the debounce window into a single
/// outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeEvent {
    /// File was created or modified — the indexer should re-parse it.
    Touched(PathBuf),
    /// File was deleted — the indexer should drop its node and incident edges.
    Removed(PathBuf),
}

impl ChangeEvent {
    pub fn path(&self) -> &Path {
        match self {
            ChangeEvent::Touched(p) | ChangeEvent::Removed(p) => p,
        }
    }
}

/// RAII watcher. Drop to stop the underlying `notify` watcher and the worker
/// thread that debounces events.
pub struct Watcher {
    rx: Receiver<ChangeEvent>,
    _watcher: RecommendedWatcher,
    /// Signals the worker to exit on drop. The worker also exits naturally
    /// if the raw-event channel closes.
    stop_tx: Sender<()>,
    /// Join handle for the debouncer thread. Kept so `Drop` can block briefly
    /// and make shutdown deterministic in tests.
    worker: Option<thread::JoinHandle<()>>,
}

impl Watcher {
    /// Start watching `root` recursively. Events are filtered against
    /// `.gitignore` rules found by walking up from each changed path and
    /// coalesced inside a `debounce` window. Non-source files are dropped.
    pub fn new(root: PathBuf, debounce: Duration) -> Result<Self, notify::Error> {
        error::install_panic_hook();

        let (raw_tx, raw_rx) = mpsc::channel::<Event>();
        let (out_tx, out_rx) = mpsc::channel::<ChangeEvent>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        // fsevents on macOS delivers canonical (`/private/var/…`) paths, while
        // the caller's `root` is whatever they passed in (`/var/…`). Normalize
        // up front so downstream filters can strip-prefix reliably.
        let root = root.canonicalize().unwrap_or(root);
        let root_for_watcher = root.clone();
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                // Ignore send errors — means the worker has exited and
                // this watcher is about to be dropped anyway.
                let _ = raw_tx.send(event);
            }
        })?;
        watcher.watch(&root_for_watcher, RecursiveMode::Recursive)?;

        let worker = thread::Builder::new()
            .name("gruff-watcher".to_string())
            .spawn(move || {
                let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                    run_worker(raw_rx, out_tx, stop_rx, root, debounce);
                }));
            })
            .map_err(|err| notify::Error::generic(&err.to_string()))?;

        Ok(Self {
            rx: out_rx,
            _watcher: watcher,
            stop_tx,
            worker: Some(worker),
        })
    }

    /// Non-blocking drain of every event that has finished its debounce
    /// window. Called once per UI frame; returns an empty vec when the repo
    /// is quiet.
    pub fn drain(&self) -> Vec<ChangeEvent> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(ev) => out.push(ev),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Debouncer/worker loop. Consumes raw notify events, coalesces per-path,
/// drops ignored paths, and forwards a single [`ChangeEvent`] per settled
/// path when the debounce window expires.
fn run_worker(
    raw_rx: Receiver<Event>,
    out_tx: Sender<ChangeEvent>,
    stop_rx: Receiver<()>,
    root: PathBuf,
    debounce: Duration,
) {
    // `pending[path]` holds the latest decision (Touched/Removed) plus the
    // timestamp of the most recent raw event for that path. We flush an
    // entry once its timestamp is `debounce` old — so a burst of saves from
    // a single editor write (e.g. write-to-temp + rename) collapses to one
    // emitted event per file.
    let mut pending: HashMap<PathBuf, (PendingKind, Instant)> = HashMap::new();

    loop {
        if stop_rx.try_recv().is_ok() {
            return;
        }

        // If we have anything pending, don't block longer than it would take
        // for the oldest entry to age out; otherwise a quiet watcher would
        // sit forever on a still-active path.
        let timeout = if pending.is_empty() {
            // No pending work — block up to 200ms so we stay responsive to
            // shutdown without burning CPU.
            Duration::from_millis(200)
        } else {
            let now = Instant::now();
            let oldest_age = pending
                .values()
                .map(|(_, t)| now.saturating_duration_since(*t))
                .min()
                .unwrap_or(Duration::ZERO);
            // Sleep just long enough for the oldest event to expire.
            debounce
                .saturating_sub(oldest_age)
                .max(Duration::from_millis(5))
        };

        match raw_rx.recv_timeout(timeout) {
            Ok(event) => apply_raw_event(&event, &root, &mut pending),
            Err(RecvTimeoutError::Disconnected) => {
                flush_ready(&mut pending, debounce, &out_tx, /*force=*/ true);
                return;
            }
            Err(RecvTimeoutError::Timeout) => {}
        }

        flush_ready(&mut pending, debounce, &out_tx, /*force=*/ false);
    }
}

/// Intermediate state per path: `notify` reports create/modify/remove as a
/// stream; we reduce it to one of these before flushing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingKind {
    /// File was created or modified — the most recent observation.
    Touched,
    /// File was deleted — the most recent observation. If a later event
    /// re-creates the file, the entry flips back to `Touched`.
    Removed,
}

fn apply_raw_event(
    event: &Event,
    root: &Path,
    pending: &mut HashMap<PathBuf, (PendingKind, Instant)>,
) {
    let now = Instant::now();
    let kind = match event.kind {
        EventKind::Create(_) => PendingKind::Touched,
        EventKind::Modify(ModifyKind::Data(_))
        | EventKind::Modify(ModifyKind::Any)
        | EventKind::Modify(ModifyKind::Metadata(_)) => PendingKind::Touched,
        // Rename events show up as Modify::Name on the old path and Create on
        // the new path. Treat the old-path side as a removal; the new side
        // arrives as its own Create event.
        EventKind::Modify(ModifyKind::Name(_)) => PendingKind::Removed,
        EventKind::Remove(RemoveKind::File) | EventKind::Remove(RemoveKind::Any) => {
            PendingKind::Removed
        }
        // Access, Other, folder removes, etc. — not interesting.
        _ => return,
    };

    for path in &event.paths {
        if !should_forward(path, root) {
            continue;
        }
        pending.insert(path.clone(), (kind, now));
    }
}

/// Drain entries whose last observation is at least `debounce` old and send
/// them. With `force`, drain everything (used at shutdown to avoid losing
/// the last burst of events).
fn flush_ready(
    pending: &mut HashMap<PathBuf, (PendingKind, Instant)>,
    debounce: Duration,
    out_tx: &Sender<ChangeEvent>,
    force: bool,
) {
    let now = Instant::now();
    let ready: Vec<PathBuf> = pending
        .iter()
        .filter(|(_, (_, t))| force || now.saturating_duration_since(*t) >= debounce)
        .map(|(p, _)| p.clone())
        .collect();

    for path in ready {
        if let Some((kind, _)) = pending.remove(&path) {
            let event = match kind {
                PendingKind::Touched => ChangeEvent::Touched(path),
                PendingKind::Removed => ChangeEvent::Removed(path),
            };
            if out_tx.send(event).is_err() {
                return;
            }
        }
    }
}

/// Gate at event-forward time: source extension, not inside `node_modules`
/// or a hidden dir beneath the repo root, and not gitignored.
fn should_forward(path: &Path, root: &Path) -> bool {
    // Only source extensions are relevant to the indexer. Whatever else is
    // happening in the tree is noise.
    if !is_source_file(path) {
        return false;
    }
    // Inspect only the components *beneath* the repo root — otherwise a repo
    // living inside a hidden parent path (macOS tempdirs, `$HOME/.dotfiles/…`)
    // would be rejected wholesale before we even evaluate gitignore rules.
    let rel = path.strip_prefix(root).unwrap_or(path);
    for comp in rel.components() {
        if let Some(s) = comp.as_os_str().to_str() {
            if s == "node_modules" {
                return false;
            }
            if s.starts_with('.') && s != "." && s != ".." {
                return false;
            }
        }
    }
    !is_gitignored(path, root)
}

/// Check `path` against the chain of `.gitignore` files between `root` and
/// the file. Mirrors the behavior of `ignore::WalkBuilder` that the indexer
/// uses, so what the initial scan skips, the watcher also skips.
fn is_gitignored(path: &Path, root: &Path) -> bool {
    // Collect .gitignore files from the file's directory up to the root.
    // Ignored if ANY matcher in the chain says so (TypeScript monorepos often
    // ignore build artifacts via nested gitignores, not the root one).
    let root_canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    let mut current = match path.parent() {
        Some(p) => p.to_path_buf(),
        None => return false,
    };

    loop {
        let gi_path = current.join(".gitignore");
        if gi_path.is_file() {
            let gi_dir = gi_path.parent().unwrap_or(&current).to_path_buf();
            let mut builder = GitignoreBuilder::new(&gi_dir);
            if builder.add(&gi_path).is_none() {
                if let Ok(gitignore) = builder.build() {
                    if matches_gitignore(&gitignore, path) {
                        return true;
                    }
                }
            }
        }

        let current_canon = current.canonicalize().unwrap_or_else(|_| current.clone());
        if current_canon == root_canon {
            break;
        }
        match current.parent() {
            Some(p) => current = p.to_path_buf(),
            None => break,
        }
    }

    false
}

fn matches_gitignore(gi: &Gitignore, path: &Path) -> bool {
    // `matched_path_or_any_parents` walks the path's ancestors too, so a
    // rule like `dist/` matches `dist/bundle.js`. Plain `matched` would miss
    // that because the file itself doesn't match the directory rule.
    matches!(
        gi.matched_path_or_any_parents(path, path.is_dir()),
        ignore::Match::Ignore(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread::sleep;
    use tempfile::tempdir;

    /// Poll `rx` for up to `timeout`, returning every event observed. Handles
    /// the fact that `notify`'s fsevent backend can take a moment to deliver
    /// the first event in a test (macOS coalesces events to ~100ms).
    fn collect_events(w: &Watcher, timeout: Duration) -> Vec<ChangeEvent> {
        let deadline = Instant::now() + timeout;
        let mut out = Vec::new();
        while Instant::now() < deadline {
            out.extend(w.drain());
            sleep(Duration::from_millis(50));
        }
        out
    }

    #[test]
    fn emits_touched_event_for_modified_source_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();

        let w =
            Watcher::new(dir.path().to_path_buf(), Duration::from_millis(100)).expect("watcher");

        // Give fsevents a moment to register the watcher before touching.
        sleep(Duration::from_millis(200));
        fs::write(dir.path().join("a.ts"), "export const x = 1;").unwrap();

        let events = collect_events(&w, Duration::from_millis(1500));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChangeEvent::Touched(p) if p.ends_with("a.ts"))),
            "expected Touched(a.ts) in {events:?}",
        );
    }

    #[test]
    fn ignores_non_source_extensions() {
        let dir = tempdir().unwrap();
        let w =
            Watcher::new(dir.path().to_path_buf(), Duration::from_millis(100)).expect("watcher");

        sleep(Duration::from_millis(200));
        // .md isn't a source file — the watcher should drop the event.
        fs::write(dir.path().join("README.md"), "hi").unwrap();

        let events = collect_events(&w, Duration::from_millis(800));
        assert!(
            events.is_empty(),
            "non-source file should produce no events, got {events:?}",
        );
    }

    #[test]
    fn ignores_node_modules() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        let w =
            Watcher::new(dir.path().to_path_buf(), Duration::from_millis(100)).expect("watcher");

        sleep(Duration::from_millis(200));
        fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();

        let events = collect_events(&w, Duration::from_millis(800));
        assert!(
            !events
                .iter()
                .any(|e| e.path().to_string_lossy().contains("node_modules")),
            "node_modules paths should be filtered out, got {events:?}",
        );
    }

    #[test]
    fn honors_root_gitignore() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join(".gitignore"), "dist/\n").unwrap();
        fs::create_dir_all(dir.path().join("dist")).unwrap();

        let w =
            Watcher::new(dir.path().to_path_buf(), Duration::from_millis(100)).expect("watcher");

        sleep(Duration::from_millis(200));
        fs::write(dir.path().join("dist/bundle.js"), "").unwrap();

        let events = collect_events(&w, Duration::from_millis(800));
        assert!(
            !events
                .iter()
                .any(|e| e.path().to_string_lossy().contains("dist")),
            "gitignored paths should not fire events, got {events:?}",
        );
    }

    #[test]
    fn emits_removed_event_for_deleted_source_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();

        let w =
            Watcher::new(dir.path().to_path_buf(), Duration::from_millis(100)).expect("watcher");

        sleep(Duration::from_millis(200));
        fs::remove_file(dir.path().join("a.ts")).unwrap();

        let events = collect_events(&w, Duration::from_millis(1500));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChangeEvent::Removed(p) if p.ends_with("a.ts"))),
            "expected Removed(a.ts) in {events:?}",
        );
    }

    #[test]
    fn watcher_feeds_indexer_end_to_end() {
        // Integration check: a live watcher → the stateful indexer exactly
        // matches the "saving a file updates the graph without user action"
        // acceptance criterion from the issue.
        use crate::indexer::Indexer;

        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::write(dir.path().join("b.ts"), "").unwrap();

        let mut indexer = Indexer::build(dir.path());
        assert_eq!(indexer.graph.edges.len(), 0);

        let w = Watcher::new(indexer.ws.root.clone(), Duration::from_millis(100)).expect("watcher");
        sleep(Duration::from_millis(200));

        // Add an import from a.ts → b.ts on disk.
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();

        // Drain events for up to 1.5s and apply them to the indexer. A real
        // UI thread does the same drain-and-apply once per frame.
        let deadline = Instant::now() + Duration::from_millis(1500);
        while Instant::now() < deadline && indexer.graph.edges.is_empty() {
            for ev in w.drain() {
                match ev {
                    ChangeEvent::Touched(p) => {
                        let _ = indexer.update_file(&p);
                    }
                    ChangeEvent::Removed(p) => {
                        let _ = indexer.remove_file(&p);
                    }
                }
            }
            sleep(Duration::from_millis(50));
        }

        assert_eq!(
            indexer.graph.edges.len(),
            1,
            "watcher should have produced the edge via incremental update",
        );
    }

    #[test]
    fn debounces_burst_of_writes_to_single_event() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();

        let w =
            Watcher::new(dir.path().to_path_buf(), Duration::from_millis(200)).expect("watcher");

        sleep(Duration::from_millis(200));
        // Five quick saves in a row within the debounce window.
        for i in 0..5 {
            fs::write(dir.path().join("a.ts"), format!("export const x = {i};")).unwrap();
            sleep(Duration::from_millis(20));
        }

        let events = collect_events(&w, Duration::from_millis(1200));
        let touched: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ChangeEvent::Touched(p) if p.ends_with("a.ts")))
            .collect();
        // Without debouncing we'd see 5+ events; with it we expect ≤ 2
        // (one for the last write; occasionally macOS splits metadata from
        // data events across the debounce boundary).
        assert!(
            touched.len() <= 2,
            "expected ≤ 2 debounced events, got {} in {events:?}",
            touched.len(),
        );
        assert!(!touched.is_empty(), "expected ≥ 1 debounced event");
    }
}
