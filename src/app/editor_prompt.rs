use std::path::PathBuf;

use eframe::egui;

use crate::colors;
use crate::config;
use crate::editor::{self, OpenError};

use super::GruffApp;

/// Quick-pick editor commands surfaced as buttons in the picker. Ordered by
/// rough popularity on macOS — VS Code first, then the classic terminal
/// editors, then JetBrains.
const QUICK_PICK_EDITORS: &[&str] = &["code", "cursor", "subl", "idea", "nvim", "vim"];

/// Action queued while the editor picker modal is open, resumed after the
/// user picks a valid editor.
#[derive(Debug, Clone)]
enum PendingEditorAction {
    OpenFile(PathBuf),
    RevealConfig,
}

pub(super) struct EditorPromptState {
    /// Text the user is typing / has picked — pre-seeded with the current
    /// config value (often empty on first launch).
    input: String,
    /// What to do once an editor is chosen. Kept on the prompt so the same
    /// picker handles "open file" and "reveal config" symmetrically.
    pending: PendingEditorAction,
    /// Error from the previous attempt (if any) — e.g. "editor not found".
    error: Option<String>,
}

impl GruffApp {
    /// Launch the user's editor on `path`. Opens the editor picker modal when
    /// no editor is configured or the configured one isn't on PATH.
    pub(super) fn try_open_in_editor(&mut self, path: PathBuf) {
        match editor::open_in_editor(&self.config.editor.name, &path) {
            Ok(()) => {
                self.status = format!("Opened {} in {}", path.display(), self.config.editor.name);
            }
            Err(OpenError::NotConfigured) => {
                self.open_editor_prompt(PendingEditorAction::OpenFile(path), None);
            }
            Err(OpenError::NotFound(name)) => {
                self.open_editor_prompt(
                    PendingEditorAction::OpenFile(path),
                    Some(format!("Couldn't find `{name}` on your PATH. Pick another editor:")),
                );
            }
            Err(OpenError::Io(e)) => {
                self.status = format!("Failed to launch editor: {e}");
            }
        }
    }

    /// Open `~/.gruff/config.toml` in the user's editor, creating it with
    /// sensible defaults first if it doesn't exist yet. Backs the "Reveal
    /// config file" menu item.
    pub(super) fn reveal_config_file(&mut self) {
        let path = match config::ensure_exists() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("Failed to prepare config file: {e}");
                return;
            }
        };
        match editor::open_in_editor(&self.config.editor.name, &path) {
            Ok(()) => self.status = format!("Opened {}", path.display()),
            Err(OpenError::NotConfigured) => {
                self.open_editor_prompt(PendingEditorAction::RevealConfig, None);
            }
            Err(OpenError::NotFound(name)) => {
                self.open_editor_prompt(
                    PendingEditorAction::RevealConfig,
                    Some(format!("Couldn't find `{name}` on your PATH. Pick another editor:")),
                );
            }
            Err(OpenError::Io(e)) => {
                self.status = format!("Failed to launch editor: {e}");
            }
        }
    }

    fn open_editor_prompt(&mut self, pending: PendingEditorAction, error: Option<String>) {
        // Seed the input with the current config value so users who already
        // have (say) a broken `code` config can tweak it rather than retype.
        self.editor_prompt = Some(EditorPromptState {
            input: self.config.editor.name.clone(),
            pending,
            error,
        });
    }

    /// Persist the new editor name to config and resume the pending action.
    /// Leaves the modal open with an error message if the editor still isn't
    /// found so the user can pick another without reopening the prompt.
    fn apply_picked_editor(&mut self, name: String) {
        let trimmed = name.trim().to_string();
        if trimmed.is_empty() {
            if let Some(prompt) = &mut self.editor_prompt {
                prompt.error = Some("Please enter an editor command (e.g. `code` or `vim`).".to_string());
            }
            return;
        }
        self.config.editor.name = trimmed.clone();
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Saved editor choice in-memory only — couldn't write config: {e}");
        }

        let Some(prompt) = self.editor_prompt.take() else { return; };
        let pending = prompt.pending;
        match pending {
            PendingEditorAction::OpenFile(path) => {
                match editor::open_in_editor(&self.config.editor.name, &path) {
                    Ok(()) => {
                        self.status = format!("Opened {} in {}", path.display(), trimmed);
                    }
                    Err(OpenError::NotFound(n)) => {
                        // Reopen the prompt with the new error so the user can
                        // try yet another editor without an extra click.
                        self.editor_prompt = Some(EditorPromptState {
                            input: trimmed,
                            pending: PendingEditorAction::OpenFile(path),
                            error: Some(format!("Couldn't find `{n}` on your PATH. Pick another editor:")),
                        });
                    }
                    Err(OpenError::Io(e)) => {
                        self.status = format!("Failed to launch editor: {e}");
                    }
                    Err(OpenError::NotConfigured) => {
                        // Shouldn't happen — we just saved a non-empty name.
                    }
                }
            }
            PendingEditorAction::RevealConfig => {
                match config::ensure_exists() {
                    Ok(path) => match editor::open_in_editor(&self.config.editor.name, &path) {
                        Ok(()) => self.status = format!("Opened {}", path.display()),
                        Err(OpenError::NotFound(n)) => {
                            self.editor_prompt = Some(EditorPromptState {
                                input: trimmed,
                                pending: PendingEditorAction::RevealConfig,
                                error: Some(format!("Couldn't find `{n}` on your PATH. Pick another editor:")),
                            });
                        }
                        Err(OpenError::Io(e)) => {
                            self.status = format!("Failed to launch editor: {e}");
                        }
                        Err(OpenError::NotConfigured) => {}
                    },
                    Err(e) => self.status = format!("Failed to prepare config file: {e}"),
                }
            }
        }
    }

    pub(super) fn draw_editor_prompt(&mut self, ctx: &egui::Context) {
        let Some(prompt) = &self.editor_prompt else { return; };
        // Lift fields out — the modal body wants `&mut self` access to apply
        // the user's choice, so we operate on a snapshot and write back at the
        // end via `self.editor_prompt`.
        let mut input = prompt.input.clone();
        let error = prompt.error.clone();
        let pending_label = match &prompt.pending {
            PendingEditorAction::OpenFile(p) => {
                format!("to open {}", p.display())
            }
            PendingEditorAction::RevealConfig => "to open your config file".to_string(),
        };

        let mut cancel = false;
        let mut submit: Option<String> = None;

        egui::Modal::new(egui::Id::new("gruff-editor-prompt")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.heading("Pick your editor");
            ui.add_space(6.0);
            ui.label(format!("Gruff needs an editor command {pending_label}."));
            ui.add_space(8.0);

            if let Some(msg) = &error {
                ui.colored_label(colors::CYCLE_EDGE, msg);
                ui.add_space(6.0);
            }

            ui.label(
                egui::RichText::new("Editor command")
                    .color(colors::HINT)
                    .small(),
            );
            let edit = ui.add(
                egui::TextEdit::singleline(&mut input)
                    .desired_width(f32::INFINITY)
                    .hint_text("e.g. code, vim, subl"),
            );
            // Submit on Enter from the text field — standard keyboard flow.
            let enter_pressed =
                edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Quick picks")
                    .color(colors::HINT)
                    .small(),
            );
            ui.horizontal_wrapped(|ui| {
                for name in QUICK_PICK_EDITORS {
                    if ui.button(*name).clicked() {
                        input = (*name).to_string();
                    }
                }
            });

            ui.add_space(10.0);
            ui.label(
                egui::RichText::new(
                    "You can change this later in ~/.gruff/config.toml.",
                )
                .italics()
                .color(colors::HINT)
                .small(),
            );

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
                let open_clicked = ui.button("Use this editor").clicked();
                if open_clicked || enter_pressed {
                    submit = Some(input.clone());
                }
            });
        });

        // Write the edited input back so keystrokes persist across frames.
        if let Some(prompt) = &mut self.editor_prompt {
            prompt.input = input;
        }
        if cancel {
            self.editor_prompt = None;
            return;
        }
        if let Some(name) = submit {
            self.apply_picked_editor(name);
        }
    }
}
