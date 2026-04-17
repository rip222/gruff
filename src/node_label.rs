//! Pure node → display label mapping.
//!
//! The canvas renderer and any other reader that wants a short, human-readable
//! caption for a node goes through [`display_label`]. Rules are picked to
//! preserve identity without reproducing the whole path:
//!
//! - Workspace-package synthetic nodes (`NodeKind::WorkspacePackage`) and
//!   external leaves (`NodeKind::External`) show the package name verbatim —
//!   `lodash`, `@org/utils`, `@org/app`. The indexer already stores that in
//!   `node.label`.
//! - Regular files (`NodeKind::File`) show the filename stem
//!   (`Button.tsx` → `Button`, `useAuth.ts` → `useAuth`). When the stem is
//!   `index`, the parent directory name is used instead
//!   (`components/index.ts` → `components`, `nested/deep/index.tsx` → `deep`).
//! - A file we can't extract any useful stem from falls back to the full
//!   filename, then to the stored `label` as a last resort — the function is
//!   infallible so the renderer never has to branch on "no label."
//!
//! Kept egui-free and purely data-driven so the rules are unit-testable the
//! same way as `src/filters.rs` and `src/camera.rs`.

use crate::graph::{Node, NodeKind};

/// Display label for a node. See the module docs for the rule list.
pub fn display_label(node: &Node) -> String {
    match node.kind {
        // Synthetic package nodes (workspace aggregators and external leaves)
        // already carry the package name in `label` — use it as-is so scoped
        // names like `@org/utils` render correctly without extra parsing.
        NodeKind::External | NodeKind::WorkspacePackage => node.label.clone(),
        NodeKind::File => file_label(node),
    }
}

/// Label for a file node: filename stem, or the parent directory when the
/// stem is `index` (matching the `index.ts` / `index.tsx` / `index.js` idiom
/// where the directory name is the meaningful identifier).
fn file_label(node: &Node) -> String {
    let stem = node
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string());

    if let Some(stem) = stem.as_deref()
        && stem.eq_ignore_ascii_case("index")
        && let Some(parent_name) = node
            .path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
        && !parent_name.is_empty()
    {
        return parent_name.to_string();
    }

    if let Some(stem) = stem
        && !stem.is_empty()
    {
        return stem;
    }

    // Fallback chain for exotic paths: try the full filename, then the
    // indexer-supplied label. This keeps `display_label` total so the
    // renderer never has to handle a missing caption.
    if let Some(name) = node.path.file_name().and_then(|s| s.to_str())
        && !name.is_empty()
    {
        return name.to_string();
    }
    node.label.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn file_node(path: &str) -> Node {
        Node {
            id: path.to_string(),
            path: PathBuf::from(path),
            label: PathBuf::from(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string()),
            package: None,
            kind: NodeKind::File,
        }
    }

    fn external_node(name: &str) -> Node {
        Node {
            id: format!("external:{name}"),
            path: PathBuf::from(name),
            label: name.to_string(),
            package: None,
            kind: NodeKind::External,
        }
    }

    fn workspace_package_node(name: &str) -> Node {
        Node {
            id: format!("package:{name}"),
            path: PathBuf::from(name),
            label: name.to_string(),
            package: Some(name.to_string()),
            kind: NodeKind::WorkspacePackage,
        }
    }

    #[test]
    fn button_tsx_becomes_button() {
        assert_eq!(display_label(&file_node("src/Button.tsx")), "Button");
    }

    #[test]
    fn use_auth_ts_becomes_use_auth() {
        assert_eq!(display_label(&file_node("hooks/useAuth.ts")), "useAuth");
    }

    #[test]
    fn components_index_ts_becomes_components() {
        assert_eq!(
            display_label(&file_node("components/index.ts")),
            "components"
        );
    }

    #[test]
    fn nested_deep_index_tsx_becomes_deep() {
        assert_eq!(display_label(&file_node("nested/deep/index.tsx")), "deep");
    }

    #[test]
    fn external_lodash_becomes_lodash() {
        assert_eq!(display_label(&external_node("lodash")), "lodash");
    }

    #[test]
    fn scoped_external_preserves_full_name() {
        assert_eq!(display_label(&external_node("@org/utils")), "@org/utils");
    }

    #[test]
    fn workspace_package_uses_package_name() {
        assert_eq!(
            display_label(&workspace_package_node("@org/app")),
            "@org/app"
        );
    }

    #[test]
    fn file_without_stem_falls_back_to_filename() {
        // A filename with no stem (pure dotfile) falls back to the filename
        // itself rather than producing an empty label. `.eslintrc`'s stem is
        // `.eslintrc`, but `.` alone has no file_stem — exercise that edge.
        let mut node = file_node(".");
        node.label = "(root)".to_string();
        // On an input the stem machinery rejects, we must still return
        // *something* — the indexer-supplied label is the last-resort fallback.
        assert!(!display_label(&node).is_empty());
    }

    #[test]
    fn bare_index_file_at_root_falls_back_to_filename() {
        // `index.ts` with no parent directory name — the "use parent" rule
        // doesn't apply, so we fall through to the stem (`index`) rather
        // than producing an empty string.
        assert_eq!(display_label(&file_node("index.ts")), "index");
    }

    #[test]
    fn index_js_in_named_dir_uses_dir_name() {
        // Variant of the components case for `.js` — the rule is extension-
        // agnostic, matching the PRD's "`index.*`" wording.
        assert_eq!(display_label(&file_node("utils/index.js")), "utils");
    }

    #[test]
    fn file_stem_with_multiple_dots_keeps_first_stem() {
        // `Component.stories.tsx` — `file_stem` strips only the final
        // extension, leaving `Component.stories`. That's the right answer:
        // "stories" is a meaningful suffix the user shouldn't lose.
        assert_eq!(
            display_label(&file_node("Component.stories.tsx")),
            "Component.stories"
        );
    }
}
