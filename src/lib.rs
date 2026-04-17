pub mod aggregation;
pub mod app;
pub mod camera;
pub mod cli;
pub mod colors;
pub mod config;
pub mod editor;
pub mod entry_points;
pub mod error;
pub mod export;
pub mod filter_state;
pub mod filters;
pub mod graph;
pub mod indexer;
pub mod layout;
pub mod lib_detect;
pub mod node_label;
pub mod orphan_detector;
pub mod package_tree;
pub mod parser;
pub mod reachability;
pub mod recents;
pub mod resolver;
pub mod search;
pub mod shortcuts;
pub mod watcher;
pub mod workspace;
pub mod workspace_state;

#[cfg(test)]
pub(crate) mod test_support {
    //! Crate-local test helpers shared across modules.
    //!
    //! `HOME_GUARD` serialises tests that mutate the process-global `HOME`
    //! env var. Individual tests holding their own module-local mutex
    //! would still race against each other because the env var is one
    //! global; collecting them behind a single crate-wide mutex keeps
    //! `cargo test` stable regardless of thread count.

    use std::sync::Mutex;

    pub(crate) static HOME_GUARD: Mutex<()> = Mutex::new(());
}
