use gruff::graph::{Graph, Node};
use gruff::layout::Layout;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    for &n in &[100usize, 500, 1000, 2000, 5000, 10_000, 20_000] {
        let mut g = Graph::new();
        for i in 0..n {
            g.add_node(Node {
                id: i.to_string(),
                path: PathBuf::new(),
                label: String::new(),
            });
        }
        // sprinkle edges — roughly 2 per node
        for i in 0..n {
            g.add_edge(&i.to_string(), &((i + 7) % n).to_string());
            g.add_edge(&i.to_string(), &((i + 13) % n).to_string());
        }
        let mut layout = Layout::new();
        layout.sync(&g);
        // warmup
        layout.step(1.0 / 60.0);
        let start = Instant::now();
        let iters = if n <= 500 {
            30
        } else if n <= 5000 {
            5
        } else {
            3
        };
        for _ in 0..iters {
            layout.step(1.0 / 60.0);
        }
        let ms_per_step = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("n={n:5}  step: {ms_per_step:7.2} ms  => {:.1} fps budget",
                 1000.0 / ms_per_step);
    }
}
