use eframe::egui;

use gruff::app::GruffApp;
use gruff::error;

fn main() -> eframe::Result {
    error::install_panic_hook();

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
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(GruffApp::with_autoload()))
        }),
    )
}
