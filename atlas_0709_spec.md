# ATLAS 0709 System Spec

This project intentionally keeps one execution strategy:

```text
Drafter tree decode: ordinary batch-k decode
Drafter forest decode: ordinary batch-k^2 decode
Target verify: ordinary batch verify over stage-1 paths
```

No Cascade, no masked-tree attention microbenchmarks, no shared-KV performance
claims are included in this clean controller.

## Function Definitions

`Drafter prefill` receives the prompt, builds the drafter prefix state, and
creates `k` initial active routes from the top-k tokens of the last prompt
logits. These routes share the committed prompt logically and each route owns an
independent pending token and logical path state.

`Target prefill` receives the same prompt and builds the target prefix state.

`Drafter build tree step` decodes exactly one pending token for each of the `k`
active routes using one ordinary batch-k model call. Each route returns one row
of logits. Taking top-k per row yields `k^2` candidates; global top-k by
cumulative draft probability become the next active routes.

`Drafter build tree` repeats the step `d` times. The resulting `k` completed
routes are the stage-1 candidate paths. The final logits from those `k` routes
are used to initialize `k^2` stage-2 routes.

`Drafter build forest step` decodes one pending token for all `k^2` active
routes with one ordinary batch-k^2 model call. Candidates are grouped by their
stage-1 root. Each group keeps top-k candidates, so the next frontier remains
`k^2`.

`Drafter build forest` repeats the forest step up to `d` times while target
verify runs concurrently.

`Target verify tree` verifies all stage-1 candidate paths in one ordinary batch
verify call and selects one stage-1 route. This clean project does not implement
masked tree verify; the target backend may internally score paths however it
wants, but the controller treats it as a batch verify operation.

## Standard Flow

```text
send prompt to Drafter prefill and Target prefill concurrently
Drafter build tree
do:
    send stage-1 paths to Target verify
    Drafter build forest while Target verify is running
    if Target verify returns before forest finishes:
        stop forest
        commit selected stage-1 path
        keep selected stage-1 branch's k stage-2 active routes
        continue build tree until their stage depth reaches d
    else:
        wait for Target verify
        commit selected stage-1 path
        keep selected stage-1 branch's k completed stage-2 routes
        use them directly as the next stage-1 candidates
while not EOS and not max_new_tokens
```

