# Rank-Scoped Request Stream Identity

Failure-aware scenarios identify request streams by `(source_rank, local_request_index)`. The source rank is the rank originally assigned by the selected workload, and the local request index identifies the request within that rank.

**Consequences**

Node failure pauses all request streams whose source ranks belong to that node. Requests are not reshuffled to other ranks in the MVP.
