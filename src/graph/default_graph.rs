use std::cmp::PartialEq;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::TryFrom;

use super::{Error, Graph};
use crate::graph::appendable_graph::AppendableGraph;
use crate::traversal::DepthFirst;
use crate::graph::deletable_graph::RemovableGraph;

/// A Graph backed by an adjacency matrix. Nodes and neighbors are iterated in
/// the order in which they're added.
///
/// ```rust
/// use std::convert::TryFrom;
/// use gamma::graph::{ AppendableGraph, Graph, Error, DefaultGraph };
///
/// fn main() -> Result<(), Error> {
///     let mut c3 = DefaultGraph::try_from(vec![
///         vec![ 1 ],
///         vec![ 0, 2 ],
///         vec![ 1 ]
///     ])?;
///
///     assert_eq!(c3.nodes().to_vec(), vec![ 0, 1, 2 ]);
///
///     assert_eq!(c3.add_edge(0, 1), Err(Error::DuplicateEdge(0, 1)));
///
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct DefaultGraph {
    /// Map of node id to index
    indices: HashMap<usize, usize>,
    /// Map of index to vec of adjacent node ids
    adjacency: Vec<Vec<usize>>,
    /// Map of index to node id
    nodes: Vec<usize>,
    edges: Vec<(usize, usize)>,
    /// Used when attempting to add a node without providing an id
    next_id: usize,
}

impl DefaultGraph {
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
            adjacency: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            next_id: 0,
        }
    }

    fn index_for(&self, id: usize) -> Result<usize, Error> {
        match self.indices.get(&id) {
            Some(index) => Ok(*index),
            None => Err(Error::MissingNode(id)),
        }
    }

    /// Remove the node "id" from the adjancency list "list_id".
    fn remove_adjacency(&mut self, list_id: usize, id: usize) {
        let pos = self.adjacency[list_id].iter().position(|e| *e == id);
        if let Some(pos) = pos {
            self.adjacency[list_id].swap_remove(pos);
        }
    }
}

impl Graph for DefaultGraph {
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn order(&self) -> usize {
        self.nodes.len()
    }

    fn size(&self) -> usize {
        self.edges.len()
    }

    fn nodes(&self) -> &[usize] {
        &self.nodes[..]
    }

    fn neighbors(&self, id: usize) -> Result<&[usize], Error> {
        let index = self.index_for(id)?;

        Ok(&self.adjacency[index])
    }

    fn has_node(&self, id: usize) -> bool {
        self.indices.contains_key(&id)
    }

    fn degree(&self, id: usize) -> Result<usize, Error> {
        let index = self.index_for(id)?;

        Ok(self.adjacency[index].len())
    }

    fn edges(&self) -> &[(usize, usize)] {
        &self.edges[..]
    }

    fn has_edge(&self, sid: usize, tid: usize) -> Result<bool, Error> {
        let index = self.index_for(sid)?;

        if self.indices.contains_key(&tid) {
            Ok(self.adjacency[index].contains(&tid))
        } else {
            return Err(Error::MissingNode(tid));
        }
    }
}

impl AppendableGraph for DefaultGraph {
    fn add_node(&mut self) -> Result<usize, Error> {
        // Find the next unoccupied id.
        while let Entry::Occupied(_) = self.indices.entry(self.next_id) {
            self.next_id += 1;
        }

        self.add_node_with(self.next_id)?;
        self.next_id += 1;

        Ok(self.next_id - 1)
    }

    fn add_node_with(&mut self, id: usize) -> Result<(), Error> {
        match self.indices.entry(id) {
            Entry::Occupied(_) => return Err(Error::DuplicateNode(id)),
            Entry::Vacant(entry) => {
                entry.insert(self.nodes.len());
            }
        }

        self.nodes.push(id);
        self.adjacency.push(vec![]);

        Ok(())
    }

    fn add_edge(&mut self, sid: usize, tid: usize) -> Result<(), Error> {
        let &source_index = match self.indices.get(&sid) {
            Some(index) => index,
            None => return Err(Error::MissingNode(sid)),
        };
        let &target_index = match self.indices.get(&tid) {
            Some(index) => index,
            None => return Err(Error::MissingNode(tid)),
        };

        if self.adjacency[source_index].contains(&tid) {
            return Err(Error::DuplicateEdge(sid, tid));
        }

        self.adjacency[source_index].push(tid);
        self.adjacency[target_index].push(sid);
        self.edges.push((sid, tid));

        Ok(())
    }
}

impl RemovableGraph for DefaultGraph {
    fn remove_node(&mut self, id: usize) -> usize {
        match self.indices.get(&id).map(|i| *i) {
            None => 0,
            Some(index) => {
                self.remove_edges_with(id);
                // Need to delete node by swapping it with the final element, then renumbering the
                // previously final element.
                if self.nodes.len() > 1 {
                    let final_id = *self.nodes.last().unwrap();
                    let final_index = *self.indices.get(&final_id).unwrap();
                    self.indices.insert(final_id, index);
                    self.adjacency.swap(final_index, index);
                    self.nodes.swap(final_index, index);
                }
                // Remove the old id -> index mapping
                self.indices.remove(&id);
                self.adjacency.pop();

                // Remove the actual node.
                self.nodes.pop();

                1
            }
        }
    }

    fn remove_edge(&mut self, sid: usize, tid: usize) -> usize {
        let index_sid = match self.indices.get(&sid) {
            Some(index) => *index,
            None => return 0,
        };
        let index_tid = match self.indices.get(&tid) {
            Some(index) => *index,
            None => return 0,
        };
        match self.adjacency[index_sid].contains(&tid) {
            false => 0,
            true => {
                // First, remove edge from adjacency lists.
                self.remove_adjacency(index_sid, tid);
                self.remove_adjacency(index_tid, sid);
                // Remove edge(s) from the edge list
                self.edges = self.edges.iter()
                    .filter(|edge| **edge != (sid, tid) && **edge != (tid, sid))
                    .map(|edge| *edge)
                    .collect();

                1
            }
        }
    }

    fn remove_edges_with(&mut self, id: usize) -> usize {
        let edges: Vec<_> = self.edges.iter()
            .filter(|edge| (**edge).0 == id || (**edge).1 == id)
            .map(|edge| *edge)
            .collect();
        edges.into_iter()
            .map(|(sid, tid)| self.remove_edge(sid, tid))
            .sum()
    }
}

impl TryFrom<Vec<Vec<usize>>> for DefaultGraph {
    type Error = Error;

    fn try_from(adjacency: Vec<Vec<usize>>) -> Result<Self, Self::Error> {
        let mut result = Self::new();

        for (sid, neighbors) in adjacency.iter().enumerate() {
            for (index, &tid) in neighbors.iter().enumerate() {
                if tid >= adjacency.len() {
                    return Err(Error::MissingNode(tid));
                } else if neighbors[index + 1..].contains(&tid) {
                    return Err(Error::DuplicateEdge(sid, tid));
                } else if !adjacency[tid].contains(&sid) {
                    return Err(Error::MissingEdge(tid, sid));
                }

                if sid < tid {
                    result.edges.push((sid, tid));
                }
            }

            result.nodes.push(sid);
            result.indices.insert(sid, sid);
        }

        result.adjacency = adjacency;

        Ok(result)
    }
}

impl<'a, G: Graph> TryFrom<DepthFirst<'a, G>> for DefaultGraph {
    type Error = Error;

    fn try_from(traversal: DepthFirst<'a, G>) -> Result<Self, Self::Error> {
        let mut result = DefaultGraph::new();

        for step in traversal {
            if result.is_empty() {
                result.add_node_with(step.sid)?;
            }

            if !step.cut {
                result.add_node_with(step.tid)?;
            }

            result.add_edge(step.sid, step.tid)?;
        }

        Ok(result)
    }
}

impl TryFrom<Vec<(usize, usize)>> for DefaultGraph {
    type Error = Error;

    fn try_from(edges: Vec<(usize, usize)>) -> Result<Self, Self::Error> {
        let mut result = DefaultGraph::new();

        for (sid, tid) in edges {
            if !result.has_node(sid) {
                result.add_node_with(sid)?;
            }

            if !result.has_node(tid) {
                result.add_node_with(tid)?;
            }

            result.add_edge(sid, tid)?;
        }

        Ok(result)
    }
}

impl PartialEq for DefaultGraph {
    fn eq(&self, other: &Self) -> bool {
        if self.size() != other.size() {
            return false;
        } else if self.order() != other.order() {
            return false;
        }

        for &id in self.nodes() {
            if !other.has_node(id) {
                return false;
            }
        }

        for (sid, tid) in self.edges() {
            match other.has_edge(*sid, *tid) {
                Ok(result) => {
                    if !result {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }

        true
    }
}

#[cfg(test)]
mod try_from_adjacency {
    use super::*;

    #[test]
    fn missing_node() {
        let graph = DefaultGraph::try_from(vec![vec![1]]);

        assert_eq!(graph, Err(Error::MissingNode(1)))
    }

    #[test]
    fn duplicate_edge() {
        let graph = DefaultGraph::try_from(vec![vec![1, 1], vec![0]]);

        assert_eq!(graph, Err(Error::DuplicateEdge(0, 1)))
    }

    #[test]
    fn missing_edge() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![]]);

        assert_eq!(graph, Err(Error::MissingEdge(1, 0)))
    }
}

#[cfg(test)]
mod try_from_edges {
    use super::*;

    #[test]
    fn duplicate_edge() {
        let graph = DefaultGraph::try_from(vec![(0, 1), (0, 1)]);

        assert_eq!(graph, Err(Error::DuplicateEdge(0, 1)))
    }

    #[test]
    fn duplicate_edge_reverse() {
        let graph = DefaultGraph::try_from(vec![(0, 1), (1, 0)]);

        assert_eq!(graph, Err(Error::DuplicateEdge(1, 0)))
    }

    #[test]
    fn valid() {
        let graph = DefaultGraph::try_from(vec![(0, 1), (1, 2), (3, 4)]).unwrap();
        let mut expected = DefaultGraph::new();

        assert_eq!(expected.add_node_with(0), Ok(()));
        assert_eq!(expected.add_node_with(1), Ok(()));
        assert_eq!(expected.add_node_with(2), Ok(()));
        assert_eq!(expected.add_node_with(3), Ok(()));
        assert_eq!(expected.add_node_with(4), Ok(()));
        assert_eq!(expected.add_edge(0, 1), Ok(()));
        assert_eq!(expected.add_edge(1, 2), Ok(()));
        assert_eq!(expected.add_edge(3, 4), Ok(()));

        assert_eq!(graph, expected);
    }
}

#[cfg(test)]
mod try_from_depth_first {
    use super::*;

    #[test]
    fn p3_internal() {
        let g1 = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();
        let traversal = DepthFirst::new(&g1, 1).unwrap();
        let g2 = DefaultGraph::try_from(traversal).unwrap();

        assert_eq!(g2.edges(), [(1, 0), (1, 2)])
    }

    #[test]
    fn c3() {
        let g1 = DefaultGraph::try_from(vec![vec![1, 2], vec![0, 2], vec![1, 0]]).unwrap();
        let traversal = DepthFirst::new(&g1, 0).unwrap();
        let g2 = DefaultGraph::try_from(traversal).unwrap();

        assert_eq!(g2.edges(), [(0, 1), (1, 2), (2, 0)])
    }
}

#[cfg(test)]
mod add_default_graph {
    use super::*;

    #[test]
    fn add_duplicate_node() {
        let mut graph = DefaultGraph::try_from(vec![vec![]]).unwrap();

        assert_eq!(graph.add_node_with(0), Err(Error::DuplicateNode(0)))
    }

    #[test]
    fn add_duplicate_edge() {
        let mut graph = DefaultGraph::try_from(vec![vec![1], vec![0]]).unwrap();

        assert_eq!(graph.add_edge(0, 1), Err(Error::DuplicateEdge(0, 1)))
    }

    #[test]
    fn add_duplicate_edge_reverse() {
        let mut graph = DefaultGraph::try_from(vec![vec![1], vec![0]]).unwrap();

        assert_eq!(graph.add_edge(1, 0), Err(Error::DuplicateEdge(1, 0)))
    }

    #[test]
    fn add_edge_missing_sid() {
        let mut graph = DefaultGraph::try_from(vec![vec![]]).unwrap();

        assert_eq!(graph.add_edge(1, 0), Err(Error::MissingNode(1)))
    }

    #[test]
    fn add_edge_missing_tid() {
        let mut graph = DefaultGraph::try_from(vec![vec![]]).unwrap();

        assert_eq!(graph.add_edge(0, 1), Err(Error::MissingNode(1)))
    }
}

#[cfg(test)]
mod remove_default_graph {
    use super::*;

    /// Return two nodes connected to each other.
    fn get_small_graph() -> DefaultGraph {
        let mut graph = DefaultGraph::new();
        graph.add_node_with(0).unwrap();
        graph.add_node_with(1).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph
    }

    /// Return two nodes, 0 connected to 1 connected to 2.
    fn get_medium_graph() -> DefaultGraph {
        let mut graph = DefaultGraph::new();
        graph.add_node_with(0).unwrap();
        graph.add_node_with(1).unwrap();
        graph.add_node_with(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        graph
    }

    #[test]
    /// Removing an edge.
    fn remove_edge() {
        let mut graph = get_small_graph();
        graph.remove_edge(0, 1);
        assert_eq!(graph.edges().len(), 0);
    }

    #[test]
    /// Removing the reverse of an edge, should still be valid.
    fn remove_edge_reverse() {
        let mut graph = get_small_graph();
        graph.remove_edge(1, 0);
        assert_eq!(graph.edges().len(), 0);
    }

    #[test]
    fn remove_edges() {
        let mut graph = get_medium_graph();
        let n = graph.remove_edges_with(1);
        assert_eq!(n, 2);
        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn remove_node() {
        let mut graph = get_medium_graph();
        let n = graph.remove_node(1);
        assert_eq!(n, 1);
        assert_eq!(graph.edges().len(), 0);
        assert_eq!(graph.nodes().len(), 2);
        assert!(graph.nodes().contains(&0));
        assert!(!graph.nodes().contains(&1));
        assert!(graph.nodes().contains(&2));
    }
}

#[cfg(test)]
mod is_empty {
    use super::*;

    #[test]
    fn p0() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.is_empty(), true)
    }

    #[test]
    fn p1() {
        let graph = DefaultGraph::try_from(vec![vec![]]).unwrap();

        assert_eq!(graph.is_empty(), false)
    }
}

#[cfg(test)]
mod order {
    use super::*;

    #[test]
    fn p0() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.order(), 0)
    }

    #[test]
    fn p3() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(graph.order(), 3)
    }
}

#[cfg(test)]
mod size {
    use super::*;

    #[test]
    fn p0() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.size(), 0)
    }

    #[test]
    fn p3() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(graph.size(), 2)
    }
}

#[cfg(test)]
mod nodes {
    use super::*;

    #[test]
    fn p0() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.nodes(), [])
    }

    #[test]
    fn p3() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(graph.nodes(), [0, 1, 2])
    }
}

#[cfg(test)]
mod neighbors {
    use super::*;

    #[test]
    fn given_outside() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.neighbors(1), Err(Error::MissingNode(1)))
    }

    #[test]
    fn given_inside_p3() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(graph.neighbors(1).unwrap(), [0, 2])
    }
}

#[cfg(test)]
mod has_node {
    use super::*;

    #[test]
    fn given_outside() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.has_node(0), false)
    }

    #[test]
    fn given_inside_p1() {
        let graph = DefaultGraph::try_from(vec![vec![]]).unwrap();

        assert_eq!(graph.has_node(0), true)
    }
}

#[cfg(test)]
mod degree {
    use super::*;

    #[test]
    fn given_outside() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.degree(0), Err(Error::MissingNode(0)))
    }

    #[test]
    fn given_inside_p3() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(graph.degree(1), Ok(2))
    }
}

#[cfg(test)]
mod edges {
    use super::*;

    #[test]
    fn p0() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.edges().to_vec(), vec![])
    }

    #[test]
    fn p3() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(graph.edges(), [(0, 1), (1, 2)])
    }
}

#[cfg(test)]
mod has_edge {
    use super::*;

    #[test]
    fn unk_unk() {
        let graph = DefaultGraph::new();

        assert_eq!(graph.has_edge(0, 1), Err(Error::MissingNode(0)))
    }

    #[test]
    fn sid_unk() {
        let graph = DefaultGraph::try_from(vec![vec![]]).unwrap();

        assert_eq!(graph.has_edge(0, 1), Err(Error::MissingNode(1)))
    }

    #[test]
    fn sid_tid() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0]]).unwrap();

        assert_eq!(graph.has_edge(0, 1), Ok(true))
    }

    #[test]
    fn tid_sid() {
        let graph = DefaultGraph::try_from(vec![vec![1], vec![0]]).unwrap();

        assert_eq!(graph.has_edge(1, 0), Ok(true))
    }
}

#[cfg(test)]
mod eq {
    use super::*;

    #[test]
    fn c3_and_p3() {
        let c3 = DefaultGraph::try_from(vec![vec![1, 2], vec![0, 2], vec![1, 0]]).unwrap();
        let p3 = DefaultGraph::try_from(vec![vec![1], vec![0, 2], vec![1]]).unwrap();

        assert_eq!(c3 == p3, false)
    }

    #[test]
    fn p2_and_p2_p1() {
        let p2 = DefaultGraph::try_from(vec![vec![1], vec![0]]).unwrap();
        let p2_p1 = DefaultGraph::try_from(vec![vec![1], vec![0], vec![]]).unwrap();

        assert_eq!(p2 == p2_p1, false)
    }

    #[test]
    fn p2_and_p2_reverse() {
        let g1 = DefaultGraph::try_from(vec![(0, 1)]).unwrap();
        let g2 = DefaultGraph::try_from(vec![(1, 0)]).unwrap();

        assert_eq!(g1 == g2, true)
    }

    #[test]
    fn p2_and_p2_different_ids() {
        let g1 = DefaultGraph::try_from(vec![(0, 1)]).unwrap();
        let g2 = DefaultGraph::try_from(vec![(0, 2)]).unwrap();

        assert_eq!(g1 == g2, false)
    }
}
