use std::path::Path;

use swc_common::{sync::Lrc, FileName, SourceMap};
use swc_ecma_ast::{
    CallExpr, Callee, Expr, Lit, ModuleDecl, ModuleItem, Tpl,
};
use swc_ecma_parser::{lexer::Lexer, EsSyntax, Parser, StringInput, Syntax, TsSyntax};
use swc_ecma_visit::{Visit, VisitWith};

/// Discriminator for what kind of import produced an [`ImportStatement`].
///
/// The resolver and indexer treat each kind differently: `Dynamic` imports can
/// be template-prefixed (resolving to a directory glob) and are the only kind
/// that contributes to the "unresolved dynamic imports" status-bar count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportKind {
    /// `import x from '...'` / `import '...'`.
    Static,
    /// `require('...')` (CommonJS).
    Require,
    /// `import('...')` call expression.
    Dynamic,
    /// `export * from '...'` / `export { x } from '...'`.
    ReExport,
}

/// A single import-like reference extracted from a source file.
///
/// `source` is the literal specifier when known. For template-string dynamic
/// imports it holds the literal *prefix* (everything up to the first
/// interpolation), and `is_template` is true so the resolver expands it as a
/// directory glob. For fully variable dynamic imports (`import(modName)`),
/// `is_unresolvable` is true and `source` is empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportStatement {
    pub source: String,
    pub kind: ImportKind,
    /// True only for dynamic imports whose source is a template string with
    /// at least one interpolation. The resolver expands the literal prefix
    /// to a directory glob in this case.
    pub is_template: bool,
    /// True for dynamic imports whose source can't be statically determined
    /// at all (e.g. `import(modName)` where `modName` is a runtime variable).
    /// These are counted in the "unresolved dynamic imports" status badge.
    pub is_unresolvable: bool,
}

impl ImportStatement {
    /// Constructor for a plain literal-source import. Most call sites and
    /// tests want this — keeps them from spelling out every flag.
    pub fn literal(source: impl Into<String>, kind: ImportKind) -> Self {
        Self {
            source: source.into(),
            kind,
            is_template: false,
            is_unresolvable: false,
        }
    }
}

/// Parse `src` and return every import-like reference: static `import`,
/// `require`, dynamic `import()` (literal or template), and re-exports.
///
/// `ext` picks the syntax preset (`ts`/`tsx`/`jsx`/`js`/`mjs`/`cjs`).
/// On parse failure, returns the imports already accumulated up to that
/// point — the AST walk is best-effort, never panics.
pub fn parse_imports(src: &str, ext: &str) -> Vec<ImportStatement> {
    let (is_ts, is_jsx) = match ext {
        "ts" => (true, false),
        "tsx" => (true, true),
        "jsx" => (false, true),
        // mjs/cjs/js and unknowns → parse as ES.
        _ => (false, false),
    };

    let syntax = if is_ts {
        Syntax::Typescript(TsSyntax {
            tsx: is_jsx,
            ..Default::default()
        })
    } else {
        Syntax::Es(EsSyntax {
            jsx: is_jsx,
            ..Default::default()
        })
    };

    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(
        Lrc::new(FileName::Anon),
        src.to_string(),
    );

    let lexer = Lexer::new(syntax, Default::default(), StringInput::from(&*fm), None);
    let mut parser = Parser::new_from(lexer);

    let module = match parser.parse_module() {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };

    let mut imports = Vec::new();
    for item in &module.body {
        if let ModuleItem::ModuleDecl(decl) = item {
            match decl {
                ModuleDecl::Import(d) => {
                    imports.push(ImportStatement::literal(
                        d.src.value.to_string_lossy().into_owned(),
                        ImportKind::Static,
                    ));
                }
                ModuleDecl::ExportNamed(d) => {
                    if let Some(src) = &d.src {
                        imports.push(ImportStatement::literal(
                            src.value.to_string_lossy().into_owned(),
                            ImportKind::ReExport,
                        ));
                    }
                }
                ModuleDecl::ExportAll(d) => {
                    imports.push(ImportStatement::literal(
                        d.src.value.to_string_lossy().into_owned(),
                        ImportKind::ReExport,
                    ));
                }
                _ => {}
            }
        }
    }

    let mut visitor = CallVisitor { imports: Vec::new() };
    module.visit_with(&mut visitor);
    imports.extend(visitor.imports);

    imports
}

/// Convenience: read a file from disk and extract its imports.
/// Returns an empty vec on read or parse error (best-effort behavior).
pub fn parse_file_imports(path: &Path) -> Vec<ImportStatement> {
    let Ok(src) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("js");
    parse_imports(&src, ext)
}

/// Recursive call-expression visitor that picks up `require(...)` and
/// `import(...)` references anywhere in the AST — top level, nested in an
/// expression, conditional branches, etc.
struct CallVisitor {
    imports: Vec<ImportStatement>,
}

impl Visit for CallVisitor {
    fn visit_call_expr(&mut self, n: &CallExpr) {
        match &n.callee {
            // `import(...)` — the dedicated `Callee::Import` variant.
            Callee::Import(_) => {
                if let Some(arg) = n.args.first() {
                    self.add_dynamic_arg(&arg.expr);
                } else {
                    // `import()` with no args — treat as fully unresolvable so
                    // the status-bar count surfaces it rather than silently
                    // dropping a real call site.
                    self.imports.push(ImportStatement {
                        source: String::new(),
                        kind: ImportKind::Dynamic,
                        is_template: false,
                        is_unresolvable: true,
                    });
                }
            }
            // `require(...)` — bare identifier callee. We deliberately don't
            // match `module.require` or other member-call shapes; those are
            // exotic enough to be out of scope.
            Callee::Expr(callee_expr) => {
                if let Expr::Ident(id) = &**callee_expr {
                    if &*id.sym == "require" {
                        self.add_require_arg(n);
                    }
                }
            }
            _ => {}
        }
        // Keep walking so we find `import()` nested inside other calls.
        n.visit_children_with(self);
    }
}

impl CallVisitor {
    /// Push a `Require` import for a literal-string `require("...")` call.
    /// Non-literal `require(name)` calls are silently dropped — they're the
    /// CJS analogue of truly dynamic imports, which we don't count here.
    fn add_require_arg(&mut self, n: &CallExpr) {
        let Some(arg) = n.args.first() else { return };
        if let Expr::Lit(Lit::Str(s)) = &*arg.expr {
            self.imports.push(ImportStatement::literal(
                s.value.to_string_lossy().into_owned(),
                ImportKind::Require,
            ));
        }
    }

    /// Inspect the argument to a dynamic `import(...)` call and emit the
    /// matching [`ImportStatement`].
    fn add_dynamic_arg(&mut self, expr: &Expr) {
        match expr {
            Expr::Lit(Lit::Str(s)) => {
                self.imports.push(ImportStatement::literal(
                    s.value.to_string_lossy().into_owned(),
                    ImportKind::Dynamic,
                ));
            }
            Expr::Tpl(tpl) => self.add_template_arg(tpl),
            _ => {
                // Variable, binary concat, member expression, etc. — fully
                // unresolvable. Counted as a dynamic-import miss.
                self.imports.push(ImportStatement {
                    source: String::new(),
                    kind: ImportKind::Dynamic,
                    is_template: false,
                    is_unresolvable: true,
                });
            }
        }
    }

    /// Handle `` import(`./foo${x}`) `` — capture the literal prefix and mark
    /// as a template so the resolver expands it to a directory glob. Pure
    /// literal templates (no interpolations) collapse to a regular dynamic
    /// import; templates with no usable prefix become unresolvable.
    fn add_template_arg(&mut self, tpl: &Tpl) {
        let prefix = tpl
            .quasis
            .first()
            .map(|q| {
                q.cooked
                    .as_ref()
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_else(|| q.raw.to_string())
            })
            .unwrap_or_default();

        if tpl.exprs.is_empty() {
            // No interpolation — equivalent to a string literal.
            self.imports.push(ImportStatement::literal(
                prefix,
                ImportKind::Dynamic,
            ));
            return;
        }

        if prefix.is_empty() {
            self.imports.push(ImportStatement {
                source: String::new(),
                kind: ImportKind::Dynamic,
                is_template: true,
                is_unresolvable: true,
            });
            return;
        }

        self.imports.push(ImportStatement {
            source: prefix,
            kind: ImportKind::Dynamic,
            is_template: true,
            is_unresolvable: false,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources(imports: &[ImportStatement]) -> Vec<&str> {
        imports.iter().map(|i| i.source.as_str()).collect()
    }

    fn kinds(imports: &[ImportStatement]) -> Vec<ImportKind> {
        imports.iter().map(|i| i.kind).collect()
    }

    #[test]
    fn extracts_single_named_import() {
        let src = "import { foo } from './foo';";
        let imps = parse_imports(src, "ts");
        assert_eq!(sources(&imps), vec!["./foo"]);
        assert_eq!(kinds(&imps), vec![ImportKind::Static]);
    }

    #[test]
    fn extracts_default_import() {
        let src = "import Foo from './Foo';";
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./Foo"]);
    }

    #[test]
    fn extracts_namespace_import() {
        let src = "import * as utils from '../utils';";
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["../utils"]);
    }

    #[test]
    fn extracts_side_effect_import() {
        let src = "import './register';";
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./register"]);
    }

    #[test]
    fn extracts_multiple_imports_in_order() {
        let src = r#"
            import a from "./a";
            import b from "./b";
            import c from "./c";
        "#;
        assert_eq!(
            sources(&parse_imports(src, "ts")),
            vec!["./a", "./b", "./c"]
        );
    }

    #[test]
    fn handles_typescript_type_annotations() {
        let src = r#"
            import { Foo } from "./foo";
            type Bar = { x: number };
            const y: Bar = { x: 1 };
        "#;
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./foo"]);
    }

    #[test]
    fn handles_type_only_import() {
        let src = "import type { Foo } from './foo';";
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./foo"]);
    }

    #[test]
    fn handles_tsx_jsx() {
        let src = r#"
            import React from "react";
            import { Button } from "./Button";
            export const App = () => <Button />;
        "#;
        assert_eq!(
            sources(&parse_imports(src, "tsx")),
            vec!["react", "./Button"]
        );
    }

    #[test]
    fn captures_require_calls_as_require_kind() {
        let src = r#"
            const foo = require("./foo");
            const bar = require("./bar");
        "#;
        let imps = parse_imports(src, "js");
        assert_eq!(sources(&imps), vec!["./foo", "./bar"]);
        assert!(imps.iter().all(|i| i.kind == ImportKind::Require));
    }

    #[test]
    fn require_alongside_static_import() {
        let src = r#"
            const foo = require("./foo");
            import bar from "./bar";
        "#;
        let imps = parse_imports(src, "js");
        // Static imports come first (collected from module body); require is
        // collected by the AST visitor that runs second.
        assert_eq!(sources(&imps), vec!["./bar", "./foo"]);
        assert_eq!(
            kinds(&imps),
            vec![ImportKind::Static, ImportKind::Require]
        );
    }

    #[test]
    fn ignores_non_literal_require() {
        // Non-literal `require(name)` is the CJS counterpart of a truly
        // dynamic import — we drop it silently rather than guessing.
        let src = r#"
            const name = "./foo";
            const x = require(name);
        "#;
        let imps = parse_imports(src, "js");
        assert!(imps.is_empty(), "got {imps:?}");
    }

    #[test]
    fn captures_dynamic_string_import() {
        let src = r#"
            const mod = import("./dynamic");
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(sources(&imps), vec!["./dynamic"]);
        assert_eq!(imps[0].kind, ImportKind::Dynamic);
        assert!(!imps[0].is_template);
        assert!(!imps[0].is_unresolvable);
    }

    #[test]
    fn captures_template_prefix_dynamic_import() {
        let src = r#"
            const mod = import(`./locales/${locale}`);
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(imps.len(), 1);
        assert_eq!(imps[0].source, "./locales/");
        assert_eq!(imps[0].kind, ImportKind::Dynamic);
        assert!(imps[0].is_template);
        assert!(!imps[0].is_unresolvable);
    }

    #[test]
    fn template_with_empty_prefix_is_unresolvable() {
        // No literal head before the first interpolation — there's nothing
        // for the resolver to anchor on, so it's marked truly dynamic.
        let src = r#"
            const mod = import(`${dir}/foo`);
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(imps.len(), 1);
        assert!(imps[0].is_unresolvable);
        assert!(imps[0].is_template);
    }

    #[test]
    fn fully_variable_dynamic_import_is_unresolvable() {
        let src = r#"
            const mod = import(name);
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(imps.len(), 1);
        assert_eq!(imps[0].kind, ImportKind::Dynamic);
        assert!(imps[0].is_unresolvable);
        assert_eq!(imps[0].source, "");
    }

    #[test]
    fn template_without_interpolation_is_static_dynamic() {
        // `` import(`./foo`) `` — a template literal with no `${...}` is
        // semantically a string. We treat it as a normal Dynamic import.
        let src = r#"
            const mod = import(`./foo`);
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(imps.len(), 1);
        assert_eq!(imps[0].source, "./foo");
        assert_eq!(imps[0].kind, ImportKind::Dynamic);
        assert!(!imps[0].is_template);
    }

    #[test]
    fn nested_dynamic_imports_are_visited() {
        // Dynamic imports inside conditional/expression positions must still
        // be captured — proves the visitor descends into children.
        let src = r#"
            const mod = condition ? import("./a") : import("./b");
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(sources(&imps), vec!["./a", "./b"]);
        assert!(imps.iter().all(|i| i.kind == ImportKind::Dynamic));
    }

    #[test]
    fn captures_re_export_all() {
        let src = r#"
            export * from "./foo";
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(sources(&imps), vec!["./foo"]);
        assert_eq!(imps[0].kind, ImportKind::ReExport);
    }

    #[test]
    fn captures_re_export_named() {
        let src = r#"
            export { foo, bar } from "./foo";
        "#;
        let imps = parse_imports(src, "ts");
        assert_eq!(sources(&imps), vec!["./foo"]);
        assert_eq!(imps[0].kind, ImportKind::ReExport);
    }

    #[test]
    fn export_named_without_source_is_not_an_import() {
        // `export { foo }` without a `from` clause is a local export — not
        // an import edge to anything.
        let src = r#"
            const foo = 1;
            export { foo };
        "#;
        let imps = parse_imports(src, "ts");
        assert!(imps.is_empty());
    }

    #[test]
    fn captures_all_kinds_in_one_file() {
        let src = r#"
            import a from "./a";
            export * from "./b";
            const c = require("./c");
            const d = import("./d");
        "#;
        let imps = parse_imports(src, "ts");
        let pairs: Vec<(&str, ImportKind)> = imps
            .iter()
            .map(|i| (i.source.as_str(), i.kind))
            .collect();
        assert!(pairs.contains(&("./a", ImportKind::Static)));
        assert!(pairs.contains(&("./b", ImportKind::ReExport)));
        assert!(pairs.contains(&("./c", ImportKind::Require)));
        assert!(pairs.contains(&("./d", ImportKind::Dynamic)));
    }

    #[test]
    fn empty_on_parse_failure() {
        // Deliberate syntax error → empty vec, no panic.
        let src = "import { from './broken'";
        assert_eq!(parse_imports(src, "ts"), Vec::<ImportStatement>::new());
    }

    #[test]
    fn preserves_bare_specifiers_for_resolver_to_reject() {
        let src = r#"
            import React from "react";
            import { merge } from "lodash-es";
        "#;
        assert_eq!(
            sources(&parse_imports(src, "ts")),
            vec!["react", "lodash-es"]
        );
    }
}
