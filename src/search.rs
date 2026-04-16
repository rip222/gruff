use std::collections::{HashMap, HashSet};

use crate::graph::{Node, NodeId};

/// Case-insensitive subsequence match: every character of `query` appears in
/// `target` in order. Empty queries never match — callers treat an empty
/// query as "no search active" rather than "match everything".
pub fn fuzzy_match(query: &str, target: &str) -> bool {
    if query.is_empty() {
        return false;
    }
    let mut q = query.chars().flat_map(char::to_lowercase).peekable();
    for tc in target.chars().flat_map(char::to_lowercase) {
        if let Some(&qc) = q.peek() {
            if tc == qc {
                q.next();
            }
        } else {
            return true;
        }
    }
    q.peek().is_none()
}

/// Compute the set of nodes that match `query` under the package-aware rule:
///
/// 1. Any node whose basename, full id, or owning-package name fuzzy-matches.
/// 2. Every node whose owning-package name fuzzy-matches — even if the node's
///    own path doesn't. This is what lets typing `shared` light up every file
///    in the `shared` package, not just files whose path contains `shared`.
///
/// Returns an empty set for an empty/whitespace-only query so the caller can
/// treat "no matches" and "no search active" identically.
pub fn compute_matches(query: &str, nodes: &HashMap<NodeId, Node>) -> HashSet<NodeId> {
    let q = query.trim();
    let mut out = HashSet::new();
    if q.is_empty() {
        return out;
    }

    // First pass: which packages match? Expanding by package is what makes
    // "shared" light up every file in shared/, not just files whose path
    // happens to contain the substring.
    let mut matched_packages: HashSet<&str> = HashSet::new();
    for node in nodes.values() {
        if let Some(pkg) = node.package.as_deref() {
            if fuzzy_match(q, pkg) {
                matched_packages.insert(pkg);
            }
        }
    }

    // Second pass: a node matches if its package is in the matched set OR
    // its own path/basename matches directly.
    for (id, node) in nodes {
        let pkg_hit = node
            .package
            .as_deref()
            .is_some_and(|p| matched_packages.contains(p));
        if pkg_hit {
            out.insert(id.clone());
            continue;
        }
        // Basename is a separate check from the full id so typing `app`
        // matches `apps/web/app.ts` both by basename (`app.ts`) and full
        // path, but a single-char match against something deep in the path
        // still has a shot via the full-id check.
        let basename = node.path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if fuzzy_match(q, basename) || fuzzy_match(q, id) {
            out.insert(id.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeKind};
    use std::path::PathBuf;

    fn node(id: &str, pkg: Option<&str>) -> (NodeId, Node) {
        (
            id.to_string(),
            Node {
                id: id.to_string(),
                path: PathBuf::from(id),
                label: id.to_string(),
                package: pkg.map(str::to_string),
                kind: NodeKind::File,
            },
        )
    }

    fn graph(pairs: &[(&str, Option<&str>)]) -> HashMap<NodeId, Node> {
        pairs.iter().map(|(id, pkg)| node(id, *pkg)).collect()
    }

    #[test]
    fn fuzzy_match_basic_subsequence() {
        assert!(fuzzy_match("abc", "aXbYcZ"));
        assert!(fuzzy_match("abc", "abc"));
        assert!(!fuzzy_match("abc", "acb"));
        assert!(!fuzzy_match("abc", "ab"));
    }

    #[test]
    fn fuzzy_match_is_case_insensitive() {
        assert!(fuzzy_match("App", "src/components/application.ts"));
        assert!(fuzzy_match("SHARED", "shared"));
    }

    #[test]
    fn fuzzy_match_empty_query_is_no_match() {
        // Callers treat empty query as "search inactive". An empty query
        // that matches everything would force the caller to special-case
        // "don't dim anything" — easier if the matcher says no.
        assert!(!fuzzy_match("", "anything"));
    }

    #[test]
    fn empty_query_returns_empty_matches() {
        let g = graph(&[("a.ts", Some("shared")), ("b.ts", Some("web"))]);
        assert!(compute_matches("", &g).is_empty());
        assert!(compute_matches("   ", &g).is_empty());
    }

    #[test]
    fn matches_by_basename_only() {
        // Query matches the basename of one file; the rest stay unmatched.
        let g = graph(&[
            ("apps/web/main.ts", Some("web")),
            ("apps/web/util.ts", Some("web")),
            ("apps/server/index.ts", Some("server")),
        ]);
        let m = compute_matches("main", &g);
        assert_eq!(m.len(), 1);
        assert!(m.contains("apps/web/main.ts"));
    }

    #[test]
    fn matches_by_full_path() {
        // "serveri" isn't a basename substring but fuzzy-matches the full id
        // `apps/server/index.ts`. Confirms we search both.
        let g = graph(&[
            ("apps/web/main.ts", Some("web")),
            ("apps/server/index.ts", Some("server")),
        ]);
        let m = compute_matches("serveri", &g);
        assert!(m.contains("apps/server/index.ts"));
        assert!(!m.contains("apps/web/main.ts"));
    }

    #[test]
    fn package_name_match_lights_entire_package() {
        // The hero behavior from issue #10: typing `shared` must light up
        // every file whose owning package fuzzy-matches, even files whose
        // path doesn't literally contain `shared`.
        let g = graph(&[
            ("libs/stuff/core.ts", Some("shared")),
            ("libs/stuff/util.ts", Some("shared")),
            ("apps/web/main.ts", Some("web")),
        ]);
        let m = compute_matches("shared", &g);
        assert!(m.contains("libs/stuff/core.ts"));
        assert!(m.contains("libs/stuff/util.ts"));
        assert!(!m.contains("apps/web/main.ts"));
    }

    #[test]
    fn package_match_is_additive_with_path_match() {
        // A query can match by package for some nodes and by path for others,
        // both contribute to the final match set.
        let g = graph(&[
            ("libs/core/a.ts", Some("core")),         // pkg match
            ("apps/web/core-config.ts", Some("web")), // basename match
            ("apps/web/other.ts", Some("web")),       // no match
        ]);
        let m = compute_matches("core", &g);
        assert!(m.contains("libs/core/a.ts"));
        assert!(m.contains("apps/web/core-config.ts"));
        assert!(!m.contains("apps/web/other.ts"));
    }

    #[test]
    fn query_matching_no_nodes_returns_empty() {
        let g = graph(&[("a.ts", Some("web"))]);
        assert!(compute_matches("xyzzy", &g).is_empty());
    }
}
