# Gamma

A graph library for Rust.

Gamma provides primitives and traversals for working with [graphs](https://en.wikipedia.org/wiki/Graph_theory). It is based on ideas presented in *[A Minimal Graph API](https://depth-first.com/articles/2020/01/06/a-minimal-graph-api/)*.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
gamma = 0.8
```

## Examples

`ArrayGraph` is a reference `Graph` implementation. Node, neighbor, and
edge iteration order are stable and determined by the `from_adjacency` function.

```rust
use std::convert::TryFrom;
use gamma::graph::{ Graph, DefaultGraph, Error };

fn main() -> Result<(), Error> {
    let p3 = DefaultGraph::try_from(vec![
        vec![ 1 ],
        vec![ 0, 2 ],
        vec![ 1 ]
    ])?;

    assert_eq!(p3.is_empty(), false);
    assert_eq!(p3.order(), 3);
    assert_eq!(p3.size(), 2);
    assert_eq!(p3.nodes().to_vec(), vec![ 0, 1, 2 ]);
    assert_eq!(p3.neighbors(1)?.to_vec(), vec![ 0, 2 ]);
    assert_eq!(p3.has_node(4), false);
    assert_eq!(p3.degree(0)?, 1);
    assert_eq!(p3.edges().to_vec(), vec![
        (0, 1),
        (1, 2)
    ]);
    assert_eq!(p3.has_edge(1, 2)?, true);

    let result = DefaultGraph::try_from(vec![
        vec![ 1 ]
    ]);

    assert_eq!(result, Err(Error::MissingNode(1)));

    Ok(())
}
```

Features include:

- depth-first and breadth-first traversal
- connected components
- maximum matching using Edmonds' Blossom algorithm

## Versions

Gamma is not yet stable, but care is taken to limit breaking changes whenever possible. Patch versions never introduce breaking changes.

## License

Tinygraph is distributed under the terms of the MIT License. See
[LICENSE-MIT](LICENSE-MIT) and [COPYRIGHT](COPYRIGHT) for details.