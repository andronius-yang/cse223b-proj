# Event-Only Manifest Rows

The MVP records simulation steps with node events in the scenario manifest even when they emit no traffic matrix. Empty all-zero matrices are not written; event-only rows preserve the failure/join timeline without adding noise to `topsim` inputs.

**Consequences**

Consumers should not assume every manifest row has a matrix path. Rows with traffic kinds such as expert migration or AllToAllV are simulator inputs; event-only rows are timeline metadata.
