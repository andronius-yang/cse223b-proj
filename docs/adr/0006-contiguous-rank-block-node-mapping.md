# Contiguous Rank Block Node Mapping

Failure scenarios use explicit rank events, while contiguous rank blocks map ranks to derived node metadata. The ranks-per-node value is supplied as scenario input. The total rank count must be divisible by ranks per node, and `traffic-gen` and `topsim` must be run with the same ranks-per-node value for the matrices to describe the intended topology.

**Consequences**

Full-node rank-block failures, same-node replica routing, and migration destination choices all depend on this mapping. The MVP does not infer or validate `topsim` command-line choices beyond emitting scenario metadata that records the ranks-per-node assumption.

Node ids are zero-based. Node `0` owns ranks `0..ranks_per_node-1`, node `1` owns the next contiguous block, and so on.

The MVP default is 16 ranks arranged as 4 nodes with 4 ranks per node.
