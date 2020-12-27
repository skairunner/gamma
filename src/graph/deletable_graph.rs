/// A graph that can be "shrunk" using remove methods.
pub trait RemovableGraph {
    /// Delete a node with the given id.
    /// Returns 0 if the id doesn't exist.
    /// Should also delete all edges involving the deleted node.
    fn remove_node(&mut self, id: usize) -> usize;

    /// Delete an edge with the given sid & tid.
    /// Returns 0 if the id doesn't exist.
    fn remove_edge(&mut self, sid: usize, tid: usize) -> usize;

    /// Delete all edges that involve the specified node. Should no-op if there are no edges.
    /// Returns the number of edges deleted.
    fn remove_edges_with(&mut self, id: usize) -> usize;
}