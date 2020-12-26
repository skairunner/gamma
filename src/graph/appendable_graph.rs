use crate::graph::Error;

/// A graph that can be "grown" using append methods.
pub trait AppendableGraph {
    /// Add a new node, returning the id of the newly created node.
    fn add_node(&mut self) -> Result<usize, Error>;

    /// Add a new node with id usize
    fn add_node_with(&mut self, id: usize) -> Result<(), Error>;

    /// Add a new edge from node sid to node tid.
    fn add_edge(&mut self, sid: usize, tid: usize) -> Result<(), Error>;
}
