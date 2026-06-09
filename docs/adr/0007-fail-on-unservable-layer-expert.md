# Fail on Unservable Layer Experts

If inference requires a layer expert with no live replica, the MVP fails the scenario explicitly rather than dropping the token, skipping the expert, or inventing a replacement copy. Expert state on failed nodes is lost, and migration requires a live source replica.

**Consequences**

Failure schedules can make scenarios invalid. This is intentional: it exposes when the replica placement does not survive the modeled failures instead of hiding the loss behind reduced traffic.
