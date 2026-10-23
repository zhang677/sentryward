# How SentryWard verifies a kernel

SentryWard runs one concrete schedule of the kernel. It then proves that every
other legal interleaving would behave the same way, using a happens-before (HB)
graph. It runs as a chain of checks, each gated on the one before. The
top-level chain is in `cpp/src/checker/verifier.cpp`: the synchronization check
runs first, and the memory check runs only if it passes. Equivalence runs after
both, orchestrated from `cpp/src/frontend/ptx/run.cpp`.

## 1. Frontend: specialize the kernel

The PTX is parsed and run symbolically for the fixed launch configuration and
problem sizes in `input.json`. Everything that depends on the configuration
(grid, thread IDs, loop bounds) becomes concrete. What's left is a per-thread
stream of events: barrier and mbarrier ops, TMA, `tcgen05`, loads and stores.

Specialization runs one CTA at a time when it can (the grid loop in `run.cpp`).
If a CTA's control flow depends on shared state that another CTA writes, such
as a `shared::cluster` load or a cluster mbarrier, the standalone run fails
with a `shared_control` error. The frontend then re-runs the whole cluster
together (`SpecializeCluster` in `cpp/src/frontend/ptx/execution.cpp`), stepping
its CTAs against one shared runtime. Only specialization is split up this way.
Every CTA's events go into one program with their CTA IDs (`AddCta`), and all
later checks run on that program for the whole grid. A remote arrive from one
CTA on another CTA's mbarrier is therefore an ordinary synchronization edge.

Synchronization across clusters through global memory is not modeled. A spin
loop on a global flag fails as `branch_not_static` or
`polling_wait_not_modeled`, so stream-K-style semaphores and grid barriers built
by hand are reported as `unsupported`.

Anything it can't handle is reported as `unsupported`, never silently
approximated. Examples are branches that depend on tensor data, election, CLC
(cluster launch cancellation) and atomics. A few of these can be handled by
assumptions you opt into, described in [docs/assumptions.md](docs/assumptions.md).

## 2. Well-synchronization

Implemented in `WellSyncChecker::RunExecutionOrder`
(`cpp/src/checker/wellsync_checker.cpp`).

1. **Reference simulation.** It runs one schedule (`ReferenceSimulator`) and
   records each barrier phase: its arrivals and the TMA transaction bytes that
   complete it. If even this schedule deadlocks or misbehaves, the result is
   `not_wellsynced` with that trace as the witness.
2. **HB graph.** It builds a graph containing only the orderings that hold in
   *every* schedule: program order plus the ordering that synchronization
   guarantees.
3. **Phase boundaries.** For each phase, in trace order
   (`cpp/src/analysis/phase_boundary.cpp`), it checks that no schedule can
   complete the phase early, with a different set of effects.
   - Each subset of events that a valid schedule could have executed is a
     downward-closed set (an "ideal") of the HB graph.
   - For each effect left out, it computes the minimum pending
     arrival/transaction count over all ideals that exclude it. This is done
     with `MinimumWeightIdealSolver`, a max-flow/min-cut solver
     (`cpp/include/sentryward/algorithm/solver/dinic.h`).
   - If that minimum can reach zero, the phase could flip early. The check
     fails, and the minimizing ideal is turned into a counterexample schedule.
   - If it passes, the phase-completion edges are added to the HB graph
     (`AddCompletedPhase`). That makes the next phase's check stronger.
4. **Waits** (`cpp/src/analysis/wait_checker.cpp`). Each `try_wait`/parity wait
   must observe the same phase in every schedule. "Lower freshness" means the
   target phase's completion is HB-before the wait. "Upper freshness" means the
   next phase can't already have completed.
5. **Lifecycle** (`cpp/src/analysis/lifecycle_checker.cpp`). Every arrival on
   phase *n+1* must be HB-after a successful wait on phase *n*, so a barrier is
   never re-armed before its previous phase was observed.
6. **Freeze.** The final HB graph must be acyclic. The graph, plus all the
   phase, wait and lifecycle certificates, becomes a `PassCertificate`
   (`cpp/include/sentryward/algorithm/checker/certificate.h`).

## 3. Race and memory check

Implemented in `MemoryRaceChecker` (`cpp/src/checker/memory_race_checker.cpp`).

Using the certified HB graph, it looks for conflicting accesses (at least one
write, overlapping bytes) to global, shared or tensor memory that the HB graph
leaves unordered.

- **Overlap.** Footprints use exact layouts, including swizzle and TMA boxes.
  Symbolic addresses go through two layers (see
  [docs/assumptions.md](docs/assumptions.md)): first allocation separation, then
  a Z3 query under the `address_invariants` you supply.
- **Outcomes:**
  - Overlapping and unordered gives `potential_race`, with both accesses as the
    witness.
  - A pair Z3 can't decide stays unresolved, reported with a candidate
    valuation when one exists. It is never counted as a pass.
- **Search order.** Normally it generates candidate pairs from address overlap,
  then queries HB. If the memory model can't enumerate candidates, it falls back
  to listing HB-unordered pairs and checking overlap for each.
- Tensor memory and barrier use-after-free/invalidate are also checked here.

Passing this stage gives `race_free`.

## 4. Equivalence (optional, needs `--reference`)

This stage runs only after the synchronization and memory checks pass.

1. **Memory elimination**
   (`cpp/include/sentryward/value_analysis/memory_elimination.h`). Because the
   kernel is race-free, any one valid execution order gives the same result. So
   every load can be replaced by the value of the store that reaches it, turning
   memory traffic into pure dataflow.
2. **Scalar expressions** (`cpp/src/scalar_fold/`). The result is folded into
   symbolic expressions for each output tensor family.
3. **Expression check.** The PyTorch reference is captured as an expression
   graph by `sentryward/_capture_reference.py`, which runs it in a subprocess.
   It is mapped onto the kernel through `reference_binding.json`. The two sides
   are then compared in an e-graph (`cpp/src/equivalence/egraph.cpp`), with
   extra certificates for reductions, polynomials and moments. Floating point is
   treated as ideal real arithmetic; integer, bit and address operations stay
   exact.
4. **Outcomes.** A proof gives `verified`. No proof gives
   `equivalence_not_proven`, which is explicitly not a counterexample.

## Reading the outcome

- Precedence follows the stage order: a synchronization, lifecycle or race
  failure always wins over equivalence results.
- Running out of time or resources is reported as `timeout`,
  `memory_budget_exceeded` or `analysis_budget_exceeded`. Checks that had
  already finished keep their status in `checks`.
- The CLI exit code is 0 even with findings, so read `result` and `checks`
  instead. The 14 categories are listed in
  [docs/output-json.md](docs/output-json.md).

## What a pass does not cover

- It holds only for the fixed shapes and launch configuration you gave it.
- Fences and the weak PTX memory model are not modeled; writes are assumed to be
  visible immediately. See [The memory-model gap](#the-memory-model-gap).
- Any assumption you enabled narrows the claim to the paths it selected. The
  assumptions used are recorded in the output.

## The memory-model gap

A pass is a proof about SentryWard's model of the GPU, not about the hardware.
The model uses interleaving semantics with immediate visibility: every event
takes effect atomically and is visible to every thread at once, essentially
sequential consistency. Real GPUs are weaker. A store can become visible to
different threads at different times, and accesses through different memory
proxies (generic loads and stores versus TMA and the tensor core) need explicit
fences to be ordered.

### How fences are handled

Fence instructions (`fence.*`, `membar`) are accepted but carry no ordering: the
lowering emits an empty `Await{}` (`cpp/src/frontend/ptx/lowering.cpp`). So:

- **A missing fence can't cause a failure.** The model already treats every
  write as visible everywhere immediately.
- **A present fence proves nothing.** The model doesn't check what it protects.

The synchronization-removal study in `examples/results.json` acknowledges this:
"fences and `griddepcontrol.wait` are never candidates because the verifier does
not check what they protect."

### Bugs that could pass verification

These follow from immediate visibility and fences being no-ops. None has been
reproduced as a test case.

1. **Missing proxy fence.** Threads write a tile to shared memory with ordinary
   `st.shared`, synchronize, then issue a TMA store that reads the tile. The PTX
   rules require `fence.proxy.async.shared::cta` in between; without it, the TMA
   can read stale data. SentryWard sees the write ordered before the read and
   passes.
2. **Missing `tcgen05` fences.** Tensor core operations need
   `tcgen05.fence::before_thread_sync` / `after_thread_sync` around the thread
   synchronization that orders them. Omitting them passes the model.
3. **Flag-based communication through global memory.** One CTA writes data, then
   sets a flag with plain stores; another CTA polls the flag. Hardware needs
   release/acquire semantics or fences so the data is visible before the flag.
   This pattern is common in stream-K and split-K fixups and cross-CTA
   semaphores, which often also use atomics (analyzed only under the
   order-independent-atomics assumption).
4. **Wrong scope (unconfirmed).** A `.cta`-scoped release/acquire can be too weak
   for communication across CTAs in a cluster. Whether the model checks the scope
   of barrier operations has not been confirmed.

### Where the gap is small

Most warp-specialized Blackwell kernels synchronize mainly through mechanisms
that carry ordering in hardware:

- `mbarrier` arrive and `try_wait` have release/acquire semantics by default.
- `bar.sync` orders memory among participating threads.
- A TMA load's `complete_tx` makes the copied data visible to threads waiting on
  that barrier.

For a kernel synchronized only this way, the model matches hardware guarantees
fairly closely, subject to SentryWard's reading of the PTX documentation. The
risk is concentrated in mixing generic and async-proxy accesses, `tcgen05`
thread-sync ordering, and communication through global memory.

### Consequences

- **"Redundant synchronization" means redundant in the model.** The removal study
  only removes barriers, mbarrier operations and waits, never fences. On
  hardware, a removed operation may still have provided visibility or proxy
  ordering. Matching outputs on B200 runs are evidence, not proof.
- **A pass covers ordering, not fence correctness.** It shows that synchronization
  orders every conflicting access, not that the fences needed for that order to
  hold on hardware are present.

### NVIDIA's tools don't close the gap

According to the
[Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html):

- **racecheck** finds races in shared memory only ("Currently, this tool only
  supports detecting accesses to on-chip shared memory"). It understands
  `cp.async` and `cuda::barrier`; TMA, PTX mbarriers, `tcgen05` and tensor memory
  are not mentioned.
- **synccheck** catches misuse of `__syncthreads()`/`__syncwarp()` and
  `cuda::barrier`, plus some `wgmma` checks on sm_90a. It doesn't check memory
  ordering.
- **memcheck** checks address validity, including `cp.async.bulk` and tensor-core
  MMA guardrails on Blackwell, not ordering.

The documentation describes no check for missing fences, `fence.proxy.async`,
visibility or release/acquire scope.

| Bug | Caught by Compute Sanitizer? |
| --- | --- |
| Missing `fence.proxy.async` between `st.shared` and a TMA store | No proxy-fence check is documented |
| Missing `tcgen05.fence::*` around thread sync | `tcgen05` isn't covered by racecheck or synccheck |
| Flag-based communication through global memory | racecheck doesn't cover global memory |
| Too-weak `.cta` scope for cluster communication | No scope check is documented |

This is based on the main documentation page only; the release notes, which
list known limitations, were not reviewed, and no tool was run. Dynamic tools
also report only problems that occur in the executions they observe, and
visibility bugs depend on hardware timing.

These bugs therefore fall between both tools. Today they are caught by review
against the PTX memory-model and proxy rules, or by testing until wrong output
appears, which is unreliable for timing-dependent bugs. A natural extension for
SentryWard would be checking required fences: its HB graph already knows which
generic-proxy write is ordered before which TMA read, so it could check that a
proxy fence lies on that path.
