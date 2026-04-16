use std::path::Path;

use swc_common::{sync::Lrc, FileName, SourceMap};
use swc_ecma_ast::{ModuleDecl, ModuleItem};
use swc_ecma_parser::{lexer::Lexer, EsSyntax, Parser, StringInput, Syntax, TsSyntax};

/// A single static `import ... from '<source>'` statement extracted from a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportStatement {
    pub source: String,
}

/// Walking-skeleton parser: static ES `import ... from '...'` only.
///
/// Non-goals for this slice: CommonJS `require`, dynamic `import()`,
/// re-exports (`export * from '...'`). `ext` picks a syntax preset.
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
    for item in module.body {
        if let ModuleItem::ModuleDecl(ModuleDecl::Import(decl)) = item {
            imports.push(ImportStatement {
                source: decl.src.value.to_string_lossy().into_owned(),
            });
        }
    }
    imports
}

/// Convenience: read a file from disk and extract its imports.
/// Returns an empty vec on read or parse error (walking-skeleton behavior).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sources(imports: &[ImportStatement]) -> Vec<&str> {
        imports.iter().map(|i| i.source.as_str()).collect()
    }

    #[test]
    fn extracts_single_named_import() {
        let src = "import { foo } from './foo';";
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./foo"]);
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
    fn ignores_require_calls() {
        let src = r#"
            const foo = require("./foo");
            import bar from "./bar";
        "#;
        // `require` is CommonJS; only the static `import` should be captured.
        assert_eq!(sources(&parse_imports(src, "js")), vec!["./bar"]);
    }

    #[test]
    fn ignores_dynamic_imports() {
        let src = r#"
            import "./side-effect";
            const mod = import("./dynamic");
        "#;
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./side-effect"]);
    }

    #[test]
    fn ignores_re_exports() {
        let src = r#"
            import a from "./a";
            export * from "./b";
            export { foo } from "./c";
        "#;
        assert_eq!(sources(&parse_imports(src, "ts")), vec!["./a"]);
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
