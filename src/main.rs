use eframe::egui;

use gruff::app::GruffApp;
use gruff::cli::{self, CliOutcome};
use gruff::error;

fn main() -> eframe::Result {
    error::install_panic_hook();

    // Resolve the CLI outcome up front so the eframe callback just picks the
    // right constructor. `cli::parse_args` is pure — all env access happens
    // here. A failed `current_dir` falls back to "/" so relative paths still
    // parse (they'll simply canonicalize against root and likely invalidate).
    let argv: Vec<String> = std::env::args().collect();
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
    let outcome = cli::parse_args(&argv, &cwd);

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Gruff")
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([480.0, 320.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Gruff",
        native_options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            let app = match outcome {
                CliOutcome::Autoload => GruffApp::with_autoload(),
                CliOutcome::OpenPath(path) => GruffApp::with_path(path),
                CliOutcome::InvalidPath { raw, reason } => {
                    GruffApp::with_error_state(format!("Couldn't open {raw}: {reason}"))
                }
            };
            Ok(Box::new(app))
        }),
    )
}
