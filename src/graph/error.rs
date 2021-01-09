use thiserror::Error;

#[derive(Debug, PartialEq, Eq, Error)]
pub enum Error {
    #[error("Missing node {0}")]
    MissingNode(usize),
    #[error("Duplicate node {0}")]
    DuplicateNode(usize),
    #[error("Missing edge ({0}, {1})")]
    MissingEdge(usize, usize),
    #[error("Duplicate edge ({0}, {1})")]
    DuplicateEdge(usize, usize),
}
