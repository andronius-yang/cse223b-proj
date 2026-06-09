# Cursor Histograms in Scenario Manifests

AllToAllV rows in the scenario timeline manifest include compact cursor histogram metadata rather than full per-request cursor lists. The histogram records how many request streams contributed from each token/layer cursor in that simulation step.

**Consequences**

The manifest remains readable while still exposing lockstep drift caused by pause and resume. Full per-request cursor traces are outside the MVP and can be added later as a debug artifact if needed.

Cursor histogram keys use raw dataset layer ids, not ordinal MoE-layer positions, so they remain traceable to the source expert-selection data and baseline layer filenames.
