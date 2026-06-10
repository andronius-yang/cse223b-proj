# Rank-Scoped Request Stream Identity

Failure-aware scenarios identify request streams by `(source_rank, local_request_index)`. The source rank is the rank originally assigned by the selected workload, and the local request index identifies the request within that rank.

**Consequences**

Rank failure pauses all request streams whose source rank is failed. A full-node failure is represented by failing every rank in that node's contiguous rank block. Requests are not reshuffled to other ranks in the MVP.
