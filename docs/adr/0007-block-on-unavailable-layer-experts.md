# Block on Unavailable Layer Experts

If inference requires a layer expert with no live replica, the MVP blocks that request stream for the tick rather than dropping the token, skipping the expert, or inventing a replacement copy. No partial traffic is emitted for that stream, and its cursor does not advance.

**Consequences**

Failure schedules can create idle periods while streams wait for a future join or repair. The scenario only becomes terminal if incomplete live streams cannot advance and no future event can restore progress.
