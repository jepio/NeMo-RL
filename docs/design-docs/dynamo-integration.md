# Managed Dynamo generation design

The managed Dynamo backend owns a fixed vLLM fleet inside the Ray allocation.
It is deliberately narrower than Dynamo itself: there is no external-runtime,
Kubernetes, DGD, data-parallel engine, or non-vLLM mode.

## Ownership and placement

Constructing `ManagedDynamoRuntime` is inert. Its explicit `start()` method
allocates ports, launches etcd and NATS JetStream, creates one Ray-managed
`dynamo.vllm` process per model-parallel group, and starts the frontend. Its
world size is derived from vLLM tensor parallelism times pipeline parallelism;
expert parallelism must be either one or equal to tensor parallelism. A group
that does not fit on one node must use whole nodes, for example TP8 on two
four-GPU nodes. The first node runs the leader, which serves the Dynamo
endpoints and the refit routes. Every other node runs a headless `dynamo.vllm`
that hosts vLLM workers only, and joins the leader through vLLM's multiprocessing
backend (`--nnodes`, `--node-rank`, `--master-addr`, `--master-port`). All nodes
of one engine use the seed of the leader. The refit addresses the leader only, and
vLLM's own collective RPC reaches the ranks on the other nodes.

Aggregated mode creates one homogeneous backend fleet from all inference GPU
groups. Disaggregated mode creates fixed decode and prefill pools. The pools
have independent replica counts but share TP, PP, EP, and all other vLLM
settings. Decode engines are ordered before prefill engines. This order is the
fixed engine order for placement, membership checks, metrics, and refit ranks.

Every P/D engine uses vLLM's `NixlConnector` with `kv_role: kv_both`, which
Dynamo requires for prefill workers. NeMo RL owns each engine's NIXL host and
side-channel port; vLLM serves every TP rank's handshake of an engine from the
one listener, so one port per engine suffices. Prefill engines also receive
managed KV-event configuration for the router. Dynamo disables these events when
vLLM prefix caching is disabled because there are no cached prefixes to reuse.
Same-node transfer can use CUDA transport. Cross-node transfer requires a working
NIXL UCX/RDMA path and is not qualified when it falls back to TCP.

The connector also sets two vLLM options that matter across weight updates.
`kv_load_failure_policy: recompute` turns a failed KV pull into a local prefill.
`kv_lease_duration` (`dynamo_cfg.disaggregation.kv_lease_duration_s`) is how long
a prefill engine keeps a finished request's KV for the decode engine. The decode
scheduler renews it with heartbeats, and only while it steps. A refit freezes the
decode engines, so the lease runs on during the pause. A prefill engine frees the
blocks when the lease ends, whether or not decode has read them, and a decode
engine does not check the lease before it reads. A lease that ends first can
therefore make decode use freed memory and wrong KV without an error. The lease
must outlast the pause. A heartbeat renews a lease to two thirds of its length and
decode engines send one every sixth, so a pause can start with half of the lease
left. Setup therefore requires at least twice `control_timeout_s`, and the
SingleController at least twice (`refit_timeout_s` + 60 s). The legacy GRPO loop
does not bound the time between pause and resume, so size the lease for the
longest refit there. A long lease only keeps the KV of a lost handoff allocated for
longer.

Startup completes only after the frontend sees the same fixed membership at
the generation and RL endpoints for both the `backend` and `prefill`
components and advertises the configured model. Worker
handles are recorded before readiness checks so partial startup failures can
be torn down. Shutdown is idempotent and guards the frontend, worker pool,
NATS, etcd, and temporary state independently.

## Generation state

Both GRPO trainers use `DynamoGeneration.generate_async()` against the managed
frontend. NeMo-Gym traffic passes through a process-local token wrapper. It
uses the policy tokenizer, preserves caller `nvext.extra_fields`, adds Dynamo
engine metadata, and translates rendered multi-turn prefixes back to the exact
caller token IDs.

Serialized rollout copies contain only frontend URLs, immutable worker admin
endpoints, and the prepared refit metadata. They cannot own or stop services.
That is enough for the SingleController actor to issue weight updates and for
AREAL-style post-refit cache invalidation; Magistral keeps its existing
driver-side invalidation lifecycle.

## Weight refit

Every engine loads `DynamoWorkerExtension`
(`nemo_rl/models/generation/vllm/dynamo_worker_extension.py`), a subclass of the
vLLM backend's `VllmInternalWorkerExtension`. `DynamoRefitChannel` drives it
through Dynamo's per-worker `/engine/*` RL routes, which forward an
`engine_rpc` name to vLLM `collective_rpc`; Dynamo needs no NeMo RL code of its
own. Setup calls go to `init_weights_update_group`, whose server deadline is
`dynamo_cfg.control_timeout_s`, and refits go to
`update_weights_from_distributed` with `allow_unpaused=True`.

Each engine is one DP shard of `E = TP x PP` refit ranks
(`get_refit_fleet_shape`), and engine `i` gets rank prefix `i * E`; the
extension adds its local rank. Both transports work this way:

- **Collective** (`refit_transport: null`): `CollectiveWeightSynchronizer`
  broadcasts packed full tensors over the NeMo RL `model_update_group`.
- **NCCL reshard** (`refit_transport: nccl_reshard`):
  `NcclReshardWeightSynchronizer` sends each training shard straight into the
  engines' shards, plus a packed broadcast for the remaining parameters.

The fleet is fixed, so any refit membership other than every engine is
rejected. Three properties of the HTTP hop shape the extension's additions:

- Dynamo discards the RPC's return value, so the `nrl_*` entry points raise
  rather than return False, and an abort comes back as `RefitAborted`.
- The admin server caps request bodies at 2 MiB, so refit metadata is staged in
  chunks (`nemo_rl/utils/staged_payload.py`) and reassembled in the engine. It is
  kept there, so a pickled client copy needs only the endpoints.
- Engine RPC tasks never retry: re-posting a collective call to a dead worker
  could never complete.

Every refit has a deadline: GRPO uses `dynamo_cfg.control_timeout_s`, the
SingleController `async_rl.generation_fleet_health.refit_timeout_s`. When it
passes, the extension and the trainers each abort their own communicators and
raise, so neither side hangs in NCCL.
KV-cache invalidation stays outside the generic synchronizer because GRPO's
cache mode determines where it runs.

## Pausing around a refit

vLLM's layerwise reload materializes parameters while the update runs, so no
request may execute beside it. The refit channel therefore pauses every engine,
prefill and decode alike, through Dynamo's `pause_generation` route, which hands
`mode` and `clear_cache` to vLLM's `pause_generation`. Both trainers use the same
pause:

- **In-flight refit** (`DynamoGeneration.pause_generation_for_refit` for the async
  GRPO loop, `drain_for_weight_update` for the SingleController): `keep` mode. The
  scheduler stops stepping and keeps its requests. They continue on the new
  weights after `resume_generation`. With
  `recompute_kv_cache_after_weight_updates` the same pause also clears the KV and
  prefix caches, on an aggregated and on a prefill/decode fleet. The
  SingleController hook `drain_for_weight_update(clear_cache=...)` returns True
  when the backend cleared the caches, and the SingleController then skips its own
  `invalidate_kv_cache` call.
- **Idle fleet** (`flush_cache`, after the async collector has drained every
  trajectory): `wait` mode with a cache clear on an aggregated fleet, and `keep`
  mode with a cache clear on a prefill/decode fleet.

All engines pause at the same time and resume decode first. The decode engines
send the heartbeats that extend the KV leases, and a resumed prefill engine
expires the leases that ended, so prefill resumes last. A resume call returns
before the decode heartbeat goes out, so the order narrows the gap and the lease
length is the guard.

One rule protects a prefill/decode fleet, and the channel enforces it with a
`ValueError` before it sends any request: no `wait` pause. A decode engine in
`PAUSED_NEW` runs no waiting-request loop, so it admits no request that waits for
remote KV. The pause of a prefill engine then waits for decode to admit its whole
backlog, a time that grows with the backlog and the generation length and can
exceed `control_timeout_s`.

If a pause fails on any engine, the channel resumes every engine before it
raises, including the engine that failed, because a pause call that timed out can
still finish later. Dynamo runs pause and resume under one lock, so the resume
lands after that late pause. The rollback waits for twice the control timeout for
the same reason. A failed refit therefore never leaves part of the fleet frozen.

`finish_generation()`, which the async trainers call after validation while
rollouts can still be in flight, posts Dynamo's `flush_cache` route to every
engine. This resets the prefix cache without a pause. The reset is all or nothing:
while a running request or a held lease still uses a block, vLLM skips the whole
reset and logs a warning, and the route still reports success. The reset therefore
only keeps the prefix cache free of one phase of generation, and it does not
protect against old-weight KV. The refit pause does that. With
`in_flight_weight_updates` off, the async collector drains its own trajectories
instead and refits an idle fleet.

A request stays open while the engines are frozen, so `request_timeout_s` must
exceed the refit time. The SingleController checks this against its refit
deadline.

### vLLM source patches

Three patches in `nemo_rl/models/generation/vllm/patches.py` make a pause and a cache
clear finish on a prefill/decode fleet. `DynamoVllmWorker` calls
`require_held_kv_patches()` when its role is not `aggregated`, and a missing
anchor stops the worker at startup instead of stalling a refit later.

**Pause with held KV** (`_patch_vllm_pause_with_held_kv`). A pause in `keep` mode
completes only when the engine core is idle. vLLM 0.29
counts a prefill engine's finished requests as work while it holds their KV for a
decode engine (`Scheduler.has_finished_requests()`). A paused decode engine pulls
nothing, so the pause of a prefill engine waits for the KV lease. NeMo RL patches
the installed vLLM source (`_patch_vllm_pause_with_held_kv` in
`nemo_rl/models/generation/vllm/patches.py`) so that
`has_finished_requests()` returns False in the `PAUSED_ALL` state for requests
that the connector holds. A request that has just finished still counts, so it
gets its flush step. The pause then completes after the last batch. The held
blocks stay allocated. The paused prefill engine is idle, so it handles no
notification and expires no lease until it resumes. The patch does not change
`PAUSED_NEW`.

**Cache clear with held KV** (`_patch_vllm_reset_cache_with_held_kv`). A pause with
`clear_cache` ends in `Scheduler.reset_prefix_cache(reset_running_requests=True)`.
That call preempts the running requests and then needs every block free, but a
prefill engine holds the KV of finished requests and a decode engine holds the
blocks of requests that wait for remote KV. Stock vLLM raises "Failed to reset KV
cache" then. With a KV connector the patch instead drops every cached prefix hash
and leaves the blocks in use alone. The holders still read valid blocks, and the
blocks stop being cacheable. Without a connector vLLM raises as before.

**Stale remote KV** (`_patch_vllm_stale_remote_kv_after_reset`). A decode request
that waits for remote KV, or that has not started its load, holds KV that a
prefill engine computed on the old weights. When its load ends, vLLM registers the
blocks in the prefix cache again (`Scheduler._update_waiting_for_remote_kv`), and
the other samples of a GRPO prompt, and the requests that the reset preempted,
would then hit them. A decode engine takes a request only after prefill finished
it, and prefill stays paused until the refit ends. So every request that reaches a
decode engine before the resume carries old-weight KV. A forced reset in a pause
arms the patch, and the resume records its time. A remote load that ends for a
request that arrived before that time has its blocks removed from the prefix cache
again.

A request that is between prefill and decode at the refit still reads its own old
KV once, which is the in-flight behavior without recompute. It does not share it.
A request that was prefilled before the pause and reaches decode only after the
resume, because the router held it for longer than the refit took, is not
recognized. This is a residual and has not been measured.

TensorRT-LLM needs no such patch: each of its engines refits itself at a
scheduler step boundary and never waits for KV transfers. The patches apply to
vLLM 0.29.0 and have to be checked on each upgrade; vLLM main has the same code
today. A refit that needs no pause at all is a possible alternative. The refit is
one collective RPC that the engine core runs between steps. It has not been tested
on GPUs.

### Open items for a GPU run

These conclusions come from reading the vLLM 0.29.0 and Dynamo 1.5.1 sources and
have not run on hardware:

- The pause of a prefill engine completes in about one step while it holds KV.
- No "Potentially invalid KV blocks" line and no expired-KV count appears across a
  refit.
- A resumed decode engine heartbeats before the prefill engine expires a lease.
- A refit contends with NIXL traffic for network bandwidth.
- A pause with `clear_cache` completes on a prefill/decode fleet, the prefix cache
  then reports no hit for old-weight prompts, and decode output stays coherent.
- After a refit with a cache clear, a repeated prompt on a decode engine reports no
  prefix hit from before the refit, including the other samples of a prompt that was
  between prefill and decode at the refit.
- NeMo-Gym under the SingleController sends no request before the first refit.

## Dependencies

`DynamoVllmWorker` runs in the `dynamo` extra's actor venv: the `vllm` extra
(vLLM 0.29.0, nccl4py, `nemo_rl`) plus the base `ai-dynamo==1.5.1` package. The
actor launches `dynamo.vllm` with its own interpreter and the driver resolves
the same venv for `dynamo.frontend`, so the extension always finds the vLLM it
was written for. `BUILD_DYNAMO=1` prebuilds that venv and installs etcd and
NATS into the image.

See [Managed Dynamo generation on Slurm](../guides/dynamo-generation.md) for
build, configuration, and launch instructions.
