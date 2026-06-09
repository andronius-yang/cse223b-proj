# Locality-First Replica Routing

The MVP routes each token to a live layer-expert replica using a locality-first deterministic policy: prefer a replica on the source rank, then a replica on the same node, then a deterministic primary live replica outside the node. It does not choose the globally least-loaded replica for the timestep, because that would assume cross-rank visibility into other requests' routing choices.

**Considered Options**

- Globally least-loaded live replica: may produce better-balanced matrices, but it assumes an oracle or centralized router with per-step knowledge of all requests.
- Random live replica: simple, but makes scenario generation less reproducible and harder to compare.
- Locality-first deterministic routing: conservative, reproducible, and closer to a runtime that can cheaply know local-node placement while relying on a static or directory-provided fallback for remote replicas.

**Consequences**

Generated traffic may show hot remote replicas when no local-node copy exists, because the MVP does not hide skew with oracle balancing. Future work can add a realistic distributed routing protocol or history-based load-aware routing.
