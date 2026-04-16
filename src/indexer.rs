use std::collections::HashMap;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::graph::{Graph, Node};
use crate::parser::parse_file_imports;
use crate::resolver::resolve_relative;
use crate::workspace::Workspace;

const SOURCE_EXTS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];

/// Scan `root` into a `Workspace`, then index every JS/TS file under it.
/// This is the entry point the app uses when the user drops a folder.
pub fn index_folder(root: &Path) -> Graph {
    let ws = Workspace::discover(root);
    index_workspace(&ws)
}

/// Walk the workspace, parse every JS/TS file, resolve relative imports, and
/// build a file-level dependency graph with each node tagged by its owning
/// package. Respects the workspace's `.gitignore`-aware matcher.
pub fn index_workspace(ws: &Workspace) -> Graph {
    let mut graph = Graph::new();

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(&ws.root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| e.depth() == 0 || !ws.is_ignored(e.path()))
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !SOURCE_EXTS.contains(&ext) {
            continue;
        }
        files.push(path.to_path_buf());
    }

    let mut id_by_path: HashMap<PathBuf, String> = HashMap::new();
    for f in &files {
        let canonical = f.canonicalize().unwrap_or_else(|_| f.clone());
        let rel = canonical
            .strip_prefix(&ws.root)
            .unwrap_or(&canonical)
            .to_path_buf();
        let id = rel.to_string_lossy().replace('\\', "/");
        let label = canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| id.clone());
        let package = ws.owning_package(&canonical).map(|p| p.name.clone());
        graph.add_node(Node {
            id: id.clone(),
            path: canonical.clone(),
            label,
            package,
        });
        id_by_path.insert(canonical, id);
    }

    for f in &files {
        let from_canonical = f.canonicalize().unwrap_or_else(|_| f.clone());
        let Some(from_id) = id_by_path.get(&from_canonical).cloned() else {
            continue;
        };
        for imp in parse_file_imports(&from_canonical) {
            let Some(resolved) = resolve_relative(&from_canonical, &imp.source) else {
                continue;
            };
            let resolved_canonical = resolved.canonicalize().unwrap_or(resolved);
            if let Some(to_id) = id_by_path.get(&resolved_canonical) {
                graph.add_edge(&from_id, to_id);
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn indexes_two_files_with_one_import() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), r#"import { b } from "./b";"#).unwrap();
        fs::write(dir.path().join("b.ts"), "export const b = 1;").unwrap();

        let g = index_folder(dir.path());
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn skips_node_modules() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.ts"), "").unwrap();
        fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        fs::write(dir.path().join("node_modules/pkg/index.js"), "").unwrap();

        let g = index_folder(dir.path());
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn tags_files_with_owning_package() {
        // Two workspace packages — each file must carry its owning package name.
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("yarn.lock"), "").unwrap();
        fs::write(
            dir.path().join("package.json"),
            r#"{"name":"root","workspaces":["packages/*"]}"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("packages/a")).unwrap();
        fs::create_dir_all(dir.path().join("packages/b")).unwrap();
        fs::write(
            dir.path().join("packages/a/package.json"),
            r#"{"name":"@org/a"}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("packages/b/package.json"),
            r#"{"name":"@org/b"}"#,
        )
        .unwrap();
        fs::write(dir.path().join("packages/a/index.ts"), "").unwrap();
        fs::write(dir.path().join("packages/b/index.ts"), "").unwrap();

        let g = index_folder(dir.path());
        let packages: Vec<_> = g
            .nodes
            .values()
            .filter_map(|n| n.package.clone())
            .collect();
        assert!(packages.contains(&"@org/a".to_string()));
        assert!(packages.contains(&"@org/b".to_string()));
    }
}
