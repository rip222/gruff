use std::collections::VecDeque;
use std::fmt;
use std::panic::{self, PanicHookInfo};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const MAX_RUNTIME_ERRORS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GruffError {
    ReadFile { path: PathBuf, message: String },
    ParseFile { path: PathBuf, message: String },
    WatcherStartup { message: String },
    Panic { thread: String, message: String },
}

impl GruffError {
    pub fn short_message(&self, root: Option<&Path>) -> String {
        match self {
            Self::ReadFile { path, message } => {
                format!("Couldn't read {}: {message}", display_path(path, root))
            }
            Self::ParseFile { path, message } => {
                format!("Skipped {}: {message}", display_path(path, root))
            }
            Self::WatcherStartup { message } => {
                format!("File watcher unavailable: {message}")
            }
            Self::Panic { thread, message } => format!("{thread} panicked: {message}"),
        }
    }

    pub fn sort_key(&self) -> String {
        match self {
            Self::ReadFile { path, .. } | Self::ParseFile { path, .. } => {
                path.display().to_string()
            }
            Self::WatcherStartup { message } => format!("watcher:{message}"),
            Self::Panic { thread, message } => format!("panic:{thread}:{message}"),
        }
    }
}

impl fmt::Display for GruffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_message(None))
    }
}

impl std::error::Error for GruffError {}

static RUNTIME_ERRORS: OnceLock<Mutex<VecDeque<GruffError>>> = OnceLock::new();
static PANIC_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

pub fn install_panic_hook() {
    PANIC_HOOK_INSTALLED.get_or_init(|| {
        let default_hook = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current()
                .name()
                .unwrap_or("unnamed thread")
                .to_string();
            report_runtime_error(GruffError::Panic {
                thread: thread.clone(),
                message: panic_message(info),
            });

            if !thread.starts_with("gruff-") {
                default_hook(info);
            }
        }));
    });
}

pub fn report_runtime_error(error: GruffError) {
    let queue = RUNTIME_ERRORS.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut queue = queue
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if queue.len() == MAX_RUNTIME_ERRORS {
        queue.pop_front();
    }
    queue.push_back(error);
}

pub fn drain_runtime_errors() -> Vec<GruffError> {
    let Some(queue) = RUNTIME_ERRORS.get() else {
        return Vec::new();
    };
    let mut queue = queue
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    queue.drain(..).collect()
}

fn panic_message(info: &PanicHookInfo<'_>) -> String {
    let payload = if let Some(message) = info.payload().downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = info.payload().downcast_ref::<String>() {
        message.clone()
    } else {
        "panic without message".to_string()
    };

    match info.location() {
        Some(location) => format!("{payload} ({location})"),
        None => payload,
    }
}

fn display_path(path: &Path, root: Option<&Path>) -> String {
    if let Some(root) = root {
        if let Ok(relative) = path.strip_prefix(root) {
            return relative.display().to_string();
        }
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::thread;

    #[test]
    fn worker_panic_is_reported_to_runtime_queue() {
        install_panic_hook();
        let _ = drain_runtime_errors();

        let worker = thread::Builder::new()
            .name("gruff-test-worker".to_string())
            .spawn(|| {
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    panic!("synthetic worker panic");
                }));
            })
            .unwrap();
        worker.join().unwrap();

        let errors = drain_runtime_errors();
        assert!(errors.iter().any(|error| {
            matches!(
                error,
                GruffError::Panic { thread, message }
                if thread == "gruff-test-worker" && message.contains("synthetic worker panic")
            )
        }));
    }
}
