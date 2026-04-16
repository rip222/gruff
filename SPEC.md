# Gruff - Module Dependency Graph Visualizer

## Overview

A native Rust desktop application for visualizing JavaScript/TypeScript module dependencies in monorepos. Built with egui + wgpu for true native GPU rendering with minimal RAM usage.

## Core Features

1. **Drag & Drop** - Drop a folder or select via file picker
2. **Force-directed Graph** - Physics-based layout showing module relationships
3. **Interactive** - Pan, zoom, click to select, drag nodes
4. **Filter/Search** - By name, package type
5. **Watch Mode** - Real-time updates as code changes
6. **Open in Editor** - Click to open file in VS Code/Vim/etc
7. **Circular Dependency Detection** - Highlight and warn

## UI/UX Specification

### Layout
```
+------------------------------------------+
| [Sidebar 280px]    |  [Graph Canvas]     |
| - Search/Filter   |                    |
| - Selected Node   |   (pan, zoom)       |
| - Details         |                    |
+------------------------------------------+
| Status Bar (parsing progress, errors)   |
+------------------------------------------+
```

### Visual Design
- **Dark mode only** (v1), design for future light mode
- **Nodes**: Rounded rectangles with text label inside
  - Workspace packages: Blue
  - External packages: Gray
- **Edges**: Straight lines, single color
- **Node size**: Variable based on dependents count

### Window
- Native macOS window frame with traffic lights
- Single window, switch repos via Open action

### Interactions
- `Cmd+O` - Open folder
- `Cmd+F` - Search/filter
- `Cmd+R` - Refresh/re-scan
- `Escape` - Deselect
- Scroll - Zoom
- Click node - Select + show sidebar details
- Click edge - Highlight dependency path
- Drag node - Reposition (optional v1.1)

## Technical Architecture

### Tech Stack
- **GUI**: egui + wgpu (native GPU rendering)
- **Parser**: swc (TypeScript/JS parsing)
- **File Watching**: notify crate
- **Config**: toml file at `~/.gruff/config.toml`

### Data Model

```rust
// Core types - future proof with traits
trait Module {
    fn path(&self) -> &Path;
    fn name(&self) -> &str;
    fn imports(&self) -> &[Import];
    fn importers(&self) -> &[&str]; // computed
}

struct Import {
    pub source: String,      // "lodash", "./utils", "@org/pkg"
    pub is_external: bool,
    pub is_workspace: bool,
}

struct DependencyGraph {
    nodes: HashMap<String, Node>,
    edges: Vec<Edge>,
}

struct Node {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub node_type: NodeType,  // Workspace | External
    pub dependents: Vec<String>,
}

enum NodeType {
    Workspace,
    External,
}

struct Edge {
    pub from: String,  // node id
    pub to: String,   // node id
}
```

### Config File (`~/.gruff/config.toml`)

```toml
[editor]
# Store name, resolve to full path at startup
name = "code"

[watch]
debounce_ms = 500
```

### Performance
- Full graph in memory
- Warn after 20,000 nodes
- Use `.gitignore` for ignore patterns

### Error Handling
- Catch panics, show user-friendly message
- Log parse errors to status bar
- Don't crash for one bad file

### Package Manager Detection
Auto-detect in order:
1. `pnpm-lock.yaml` → pnpm
2. `yarn.lock` → yarn
3. `package-lock.json` → npm

Parse `package.json` workspaces field for workspace packages.

### Parse Coverage
- ES modules (`import`)
- CommonJS (`require`)
- TypeScript path mappings (`tsconfig.json`)
- Dynamic `import()`

## Export

- JSON export for scripting

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd+O | Open folder |
| Cmd+F | Search/filter |
| Cmd+R | Refresh |
| Escape | Deselect |
| Scroll | Zoom |

## MVP Scope (v1)

1. Drag/drop folder → parse
2. Force-directed graph display
3. Pan/zoom
4. Click to select + sidebar details
5. Search/filter
6. Open in editor
7. Watch mode (real-time updates)
8. Circular dependency detection

## Future (Post-v1)

- Light mode
- Drag nodes
- Image export
- Multiple tabs
- Other language support (Go, Rust)
