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

### Terms

- **Event.** One node of the specialized program: an instruction executed by
  one thread, or one step of an asynchronous operation. A TMA copy, for example,
  has a separate *issue* event (in the issuing thread's program order) and a
  *completion* event that happens later.
- **HB graph.** A directed graph over events. An edge `u → v` means u happens
  before v in *every* legal schedule. The graph only ever contains orderings
  that have been proven; it starts small and grows as the check proceeds.
- **Effect.** An event that changes a barrier's counters. The graph does not
  distinguish effects; they are a side table (`PhaseDatabase`) recording, for
  each such event, its weights (`PhaseEffectWeights`) and the phase it fell
  into in the reference run. An effect's *weights* are the amounts by which
  it changes the barrier's two pending counters, pending arrivals and pending
  TMA bytes. Adding up the weights of a phase's effects gives the change in
  each counter over that phase (see **Phase** below). The weights of each event kind (`GetPhaseEffectWeights` in
  `cpp/src/analysis/phase_database.cpp`):

  | Event | Pending arrivals | Pending TMA bytes |
  |---|---|---|
  | `mbarrier.arrive` (optional count operand, default 1) | −count | 0 |
  | `mbarrier.arrive.expect_tx` | −1 | +bytes |
  | `mbarrier.expect_tx` | 0 | +bytes |
  | `mbarrier.complete_tx`, completion of a TMA copy | 0 | −bytes |
  | `cp.async.mbarrier.arrive` without `.noinc` (the up-front increment) | +1 | 0 |
  | named barrier arrive (`bar.arrive`, `bar.sync`) | −1 | 0 |

  Arrivals are counted per thread: a warp-wide `mbarrier.arrive` with no
  election is 32 arrivals.
- **Phase.** One round of a barrier: it starts with the expected arrival count
  pending and completes when both the pending arrivals and the pending TMA bytes
  reach zero. *Phase n's effects* are the effects the reference run assigned to
  it (`PhaseInfo::effects`). A phase is a set of nodes, not a subgraph: its
  effects usually sit on different threads with no edges between them.
- **Phase-completion node** (`doneₙ`). A node the checker adds to the HB graph
  once phase n is certified (`AddPhaseCompletion`). It corresponds to no
  instruction and stands for the moment phase n completes.
- **Wait issue / wait done.** A `try_wait` loop is two events: the *issue*
  (the thread starts waiting) and the *done* (the wait returns having observed
  the phase). Only the done event is ordered after the phase completes.
- **Downward-closed set (ideal).** A set S of events such that for every edge
  `u → v`, if v is in S then u is in S. Equivalently, no edge enters S from
  outside. For the graph `A → B → C` plus an unrelated `D`:

  | Set | Ideal? |
  |---|---|
  | `{}`, `{A}`, `{A,B}`, `{A,D}`, `{A,B,C,D}` | yes |
  | `{B}` | no: B is in, its predecessor A is not |
  | `{A,C}` | no: C is in, its predecessor B is not |

  The events that have run at any moment of any schedule form an ideal, since
  nothing runs before what it depends on. Conversely, every ideal of the HB
  graph is treated as reachable. Because the graph only holds proven orderings,
  this over-approximates the real schedules, which keeps the check
  conservative.

### Definition

A kernel is **well-synchronized** if every legal schedule agrees with the
reference run on every synchronization outcome:

1. **Phase membership.** Each phase that completes is completed by exactly the
   effects the reference run assigned to it: it never completes without one of
   them, and no effect lands in a different phase.
2. **Open phases.** A phase still incomplete at the end of the reference run
   never completes in any schedule.
3. **Waits.** Each `try_wait`/parity wait observes the same phase as in the
   reference run.
4. **Lifecycle.** A barrier is never re-armed for phase n+1 before its phase n
   was observed.

When these hold, every ordering derived from synchronization (such as "the
producer's arrive happens before the consumer's wait returns") holds in every
schedule. Since the reference run terminated, every schedule terminates, so the
kernel cannot deadlock.

### Foundations

This section states what the check assumes, what it proves and why each stage
is needed. The stages themselves are listed in
[How it is checked](#how-it-is-checked).

**Axioms.** These are SentryWard's model, assumed and not proved:

1. **Fixed events.** Specialization (section 1) produces a fixed, finite set of
   events. Every schedule runs the same events; only their order differs.
   Control flow that depends on data is reported `unsupported`.
2. **Legal schedules.** A schedule is a total order of all events that respects
   the structural edges (program order, issue before completion, …) and the
   barrier rules: a wait done runs only after its target phase has completed,
   and no arrival underflow, transaction overflow or similar violation occurs.
3. **Barrier counters.** A barrier is always in some current phase. Each effect
   changes the pending arrivals and pending TMA bytes by its weights. The phase
   completes at the first moment both reach zero, even if effects assigned to
   it have not yet run. The next phase then starts with the expected arrival
   count and zero bytes, and any later effects count toward it.
4. **Wait targets.** A wait's target phase is fixed when it is issued
   (`issuedTarget` in `cpp/src/semantics/simulator.cpp`). A parity-p wait
   issued while the barrier is in phase k targets k if p = k mod 2, and blocks
   until k completes. Otherwise it targets k−1, which has already completed,
   and returns immediately.
5. **Immediate visibility.** Every event is atomic and visible to all threads at
   once. Only the race check depends on this (see
   [The memory-model gap](#the-memory-model-gap)).

**What is proved.** The reference run produces a *classification* C: the phase
each effect lands in and the phase each wait observes. The kernel is
well-synchronized if every legal schedule produces C. The reference run shows
only that one legal schedule exists, and a badly synchronized kernel usually
has one. If, in the example below, the producer does not wait on `E`, the
schedule `a0, a1, wF0, …` is legal. When `wF0` is issued, `F` is already in
phase 2, which has parity 0, so `wF0` targets phase 2 instead of phase 0. The
reference run need not produce that schedule. Its role is to propose C; the
remaining stages prove C for all schedules or find a counterexample.

**The proof tool.** The check never enumerates schedules. It relies on one
invariant:

> **I.** Every edge `u → v` in the HB graph means u runs before v in every
> legal schedule.

Two facts follow:

- A path `u ⇝ v` also means u runs before v in every legal schedule. Every
  "HB-before" below is a reachability query on the current graph
  (`ReachabilityOracle`, `cpp/include/sentryward/algorithm/analysis/reachability.h`).
- At any moment of any legal schedule, the events that have run form an ideal.
  A property that fails for every ideal therefore fails at every moment of
  every schedule, and the min-cut solver can search over ideals.

The ideals over-approximate the reachable states. A graph with few edges admits
impossible states, so a failed proof may be a false alarm, but a passed proof
is never wrong.

**Facts proved for each phase.** Let Eₙ be the effects C assigns to phase n of
a barrier. Balance alone does not suffice: by axiom 3, a phase completes the
first time both counters reach zero, which may happen before all of Eₙ has run.

| Fact | Stage | Statement | Why it is needed |
|---|---|---|---|
| S. Separation | 2a | every effect of Eₙ is HB-after every effect of Eₙ₋₁ (for phase 0: after the `mbarrier.init`) | chained over all phases, all of E₀ runs before any of E₁, all of E₁ before any of E₂, and so on |
| B. Balance | 2b | the arrivals of Eₙ sum to the expected count, and its bytes net to zero | once all of Eₙ has run, the phase has completed |
| N. No early completion | 2b | no ideal that omits an effect of Eₙ has both counters at zero | the phase cannot complete before its last effect runs |

By S, while phase n is current only effects of Eₙ change the counters: earlier
phases' effects have all run and later ones have not started. The effects of
Eₙ that have run form an ideal, so by N the counters are not both zero until
all of Eₙ has run, and by B they are once it has. Phase n therefore completes
exactly when its last effect runs, and every effect lands in its assigned
phase (item 1 of the definition).

**Why balance is not enough.** In this example, barrier `F` has count 1. P0
loads buffer A and P1 loads buffer B; C reads both once `F` phase 0 completes:

```
P0:  e1 = expect_tx F(16)  →  a = arrive F  →  TMA load A … c1 = complete_tx F(16)
P1:  e2 = expect_tx F(16)  →  TMA load B … c2 = complete_tx F(16)
C:   w = wait F(0)  →  read A, B
```

C assigns all five effects to phase 0, and B holds: one arrival, and
+32 − 32 bytes. Nothing orders P1 with P0, so this schedule is legal:

| Event | Pending arrivals | Pending bytes | Phase |
|---|---|---|---|
| `mbarrier.init F, 1` | 1 | 0 | 0 |
| `e1` | 1 | 16 | 0 |
| `a` | 0 | 16 | 0 |
| `c1` | 0 | 0 | 0 completes |
| `w` returns; C reads B | | | B not yet loaded |
| `e2` | 1 | 16 | 1 |
| `c2` | 1 | 0 | 1 |

The barrier completes phase 0 before P1 has announced its bytes. C then reads B
before it is written, and `e2` and `c2` land in phase 1 instead of phase 0. A
schedule that runs `e2` before `c1` completes phase 0 only at `c2`, so the two
schedules produce different classifications and the kernel is not
well-synchronized. B sees the same totals in both schedules and cannot tell
them apart. N rejects the ideal `{e1, a, c1}` with `kPrematurePhaseCompletion`.
To fix the kernel, announce all the bytes before the arrival that can complete
the phase: P0 does `expect_tx F(32)`, or `F` has count 2 and each producer uses
`arrive.expect_tx F(16)`.

The solver queries of 2b establish N. Arrivals sum to exactly the expected
count, so an ideal missing one still has arrivals pending; only ideals with
every arrival need to be searched. For each non-arrival effect X, a minimum
pending byte count above zero with X excluded (`force_out`) shows that the phase cannot
complete without X. The nonnegativity query is what makes a minimum of zero or
less mean exactly zero, a real early completion. When it fails, the result is
`unsupported` rather than a rejection, unless `reject_negative_tx` is set
(`cpp/src/checker/wellsync_checker.cpp`).

The other stages prove the remaining items of the definition:

- **W. Waits** (stage 4, `cpp/src/analysis/wait_checker.cpp`). For a wait that
  C says observes phase n: every effect of Eₙ₋₁ is HB-before the wait issue,
  so phase n−1 has completed when the wait is issued. Some effect of Eₙ₊₁ is
  HB-after the wait issue, so phase n+1 has not completed. The wait's parity
  is n mod 2. By axiom 4, a wait issued in phase n or n+1 with that parity
  targets phase n.
- **O. Open phases** (stage 3). A phase incomplete at the end of the reference
  run has a counter above zero in every ideal, so it never completes.
- **L. Lifecycle** (stage 5, `cpp/src/analysis/lifecycle_checker.cpp`). Each
  arrival of phase n > 0 is HB-after the done of some wait that observed phase
  n−1, so the barrier is not re-armed before its previous phase was observed.

**Why the graph must grow.** The initial graph holds only structural edges,
which satisfy I by axiom 2. Most of them stay inside one thread or one
operation, but the facts above need orderings between threads, and those come
only from barriers. Each certified phase therefore adds the edges listed in
[How the HB graph grows](#how-the-hb-graph-grows); each one holds in every
schedule because of the facts just proved. Without them, the proofs for later
phases fail. Consider two barriers `F` and `E`, each with count 1:

```
P:  a0 = arrive F  →  wE = wait E(0)  →  a1 = arrive F
C:  wF0 = wait F(0)  →  read  →  c0 = arrive E
```

C assigns `a0` to `F` phase 0, `c0` to `E` phase 0 and `a1` to `F` phase 1.
Certifying `F` phase 0 adds `a0 → done_F0 → wF0 done`. Certifying `E` phase 0
adds `c0 → done_E0 → wE done`. Fact W for `wF0` then needs some effect of `F`
phase 1 to be HB-after `wF0`'s issue. The only path is

```
wF0 issue → wF0 done → read → c0 → done_E0 → wE done → a1
```

and it exists only because `E` phase 0 was certified first.

**Why trace order.** In the reference run, every effect of phase n ran before
phase n completed. Any wait that effect depends on observed a phase that
completed even earlier, so that phase comes earlier in the trace order, and
its edges are in the graph when phase n is checked.

**Soundness.** An edge added in stage 2c relies on facts checked later:
`doneₙ → wait done` relies on W (stage 4), and `effect → doneₙ` relies on S for
phase n+1. Invariant I is therefore established only once every stage has
passed, not as each edge is added. The intended argument runs by contradiction:

1. Suppose every stage passes, yet some legal schedule σ disagrees with C.
2. Take the earliest event of σ at which it disagrees: an effect landing in
   the wrong phase, a phase completing early, or a wait targeting the wrong
   phase.
3. Before that event σ agrees with C, so every edge whose justification
   concerns that prefix holds on it.
4. The fact covering the disagreeing event then rules it out.

No written proof of this exists in the repository. Step 3 is the delicate part:
phases of different barriers can complete in a different order in σ than in
the reference run, so "the prefix" must be defined with care.

### How it is checked

The check is an induction over phases in the order the reference run completed
them. Each step proves one fact for all schedules using only the current HB
graph, then adds the edges that fact implies. Later steps depend on edges added
by earlier ones, so the order of the stages matters.

0. **Reference run.** It runs one schedule (`ReferenceSimulator`,
   `cpp/src/semantics/simulator.cpp`) and records which phase each effect fell
   into and which phase each wait observed. This is the claim the remaining
   stages prove for all schedules. The schedule is built one event at a time,
   and an event may run only when:
   - its predecessors in the event graph (program order, issue before
     completion, …) have run;
   - for a wait done, the phase it targets has completed in the simulated
     barrier state;
   - running it breaks no barrier rule. Arrival underflow, transaction
     overflow, a wait on a stale phase and similar violations
     (`SimulationErrorCode` in
     `cpp/include/sentryward/algorithm/semantics/simulator.h`) stop the run.

   Among the events that may run, it picks the one with the lowest ID. Before
   each such pick it first runs every collective completion that may run (the
   derived event the lowering adds once all of a collective's participants
   have arrived), so a collective completes as soon as its last participant
   arrives. The choice of order is arbitrary: if the kernel is well-synchronized, every legal order
   gives the same phases and waits. If even this schedule deadlocks or breaks
   a barrier rule, the result is `not_wellsynced` with that trace as the
   witness.

   These rules are SentryWard's model of PTX barrier semantics. They are
   assumed, not checked against hardware; features outside the model are
   reported `unsupported` (section 1).
1. **Initial HB graph.** Program order, issue-before-completion inside each
   asynchronous operation, and other orderings that don't depend on any barrier
   outcome. No barrier edges yet.
2. **Each completed phase n, in trace order:**
   - **a. Phase start** (`CheckPhaseStart`). Every effect of phase n must
     already be HB-after every effect of phase n−1 (for phase 0: after the
     `mbarrier.init`). Then no effect can land in an earlier phase. Edges from
     phase n−1's completion node to phase n's effects are added.
   - **b. Phase boundary** (`CheckCompletedPhase`,
     `cpp/src/analysis/phase_boundary.cpp`). The phase must not be able to
     complete without one of its own effects. The HB graph is first projected
     onto the phase's effects, with an edge wherever one effect is HB-before
     another. Each query below is one call to `MinimumWeightIdealSolver`, a
     max-flow/min-cut solver (`cpp/include/sentryward/algorithm/solver/dinic.h`)
     that returns the single ideal with the lowest pending count, never an
     enumeration:
     - The arrivals must sum to the expected count and the TMA bytes must net
       to zero. No solver call is needed for this.
     - With every arrival required (`force_in`), the minimum pending byte count must be
       ≥ 0. Otherwise the result is `kNegativeTransactionPrefix`.
     - For each non-arrival effect E (an `expect_tx` or a byte completion):
       with every arrival required and E excluded, the minimum pending byte
       count must stay > 0. Otherwise the phase can complete without E
       (`kPrematurePhaseCompletion`), and the minimizing ideal, extended to
       all its HB predecessors and listed in topological order, is the
       counterexample schedule.

     Arrivals need no query of their own: they sum to exactly the expected
     count, so an ideal missing one still has arrivals pending. A phase with
     only arrivals needs no solver call at all. Phases with `cp.async`
     increments use a variant (`CheckIncrementPhase`) with 3 queries plus one
     per effect.
   - **c. Completion edges** (`AddCompletedPhase`). The phase-completion node
     `doneₙ` is added, with an edge from each of the phase's effects to it and
     from it to the done event of each wait that observed the phase. Stages a
     and b rely on these edges for later phases (see
     [How the HB graph grows](#how-the-hb-graph-grows)).
3. **Open phases** (`CheckOpenPhase`). For each phase still incomplete at the
   end of the reference run, one query shows the pending count stays > 0 in
   every ideal.
4. **Waits** (`cpp/src/analysis/wait_checker.cpp`). Each wait must observe the
   same phase in every schedule. "Lower freshness" means the target phase's
   completion is HB-before the wait. "Upper freshness" means the next phase
   can't already have completed.
5. **Lifecycle** (`cpp/src/analysis/lifecycle_checker.cpp`). Every arrival on
   phase n+1 must be HB-after a successful wait on phase n.
6. **Freeze.** No derived edge may contradict the reference trace, and the
   final HB graph must be acyclic. The graph, plus all the phase, wait and
   lifecycle certificates, becomes a `PassCertificate`
   (`cpp/include/sentryward/algorithm/checker/certificate.h`). The race check
   in section 3 relies on this graph.

### How the HB graph grows

The initial graph holds only orderings that don't depend on any barrier
outcome. Its edge kinds are program order, issue-before-completion inside an
asynchronous operation, state def-use and a few other structural kinds
(`toHappensBeforeReason` in `cpp/src/analysis/happens_before_graph.cpp`). Most
of them stay inside one thread or one operation.

After that, exactly three edge kinds are added, each only once a certificate
justifies it:

| Edge | Added by | Justified because |
|---|---|---|
| each effect of phase n → `doneₙ` (`kCertifiedPhaseCompletion`) | stage 2c | stages 2a and 2b proved these are exactly the effects that complete phase n, in every schedule |
| `doneₙ` → each wait done that observed phase n (`kWaitRelease`) | stage 2c | a wait done runs only after its target phase has completed, and the wait check (stage 4) proves the target is phase n |
| `doneₙ₋₁` → each effect of phase n (`kCertifiedPreviousPhase`) | stage 2a; the wait check adds the same kind of edge to wait issues | every effect of phase n−1 was already shown to reach that node |

Only the `kWaitRelease` edges create new orderings between real events. The
other two kinds connect a phase-completion node: `doneₙ₋₁ → X` is added only
when every effect of phase n−1 already reaches X, so it never lets one real
event reach another that it couldn't reach before. These edges make `doneₙ` an
exact summary of "phase n has completed", which later queries use.

For example, a producer writes X and then arrives; a consumer waits and then
reads X. Before phase 0 is certified, the write and the read are unordered:

```
P:  write X  →  arrive*
C:  wait issue  →  wait done  →  read X
```

After phase 0 is certified, they are ordered through `done₀`:

```
P:  write X  →  arrive*  ─┐
                          ↓
                        done₀
                          ↓
C:  wait issue  →  wait done  →  read X
```

The path `write X → arrive → done₀ → wait done → read X` is what lets the race
check in section 3 treat the write and the read as ordered.

Each edge has to wait for its proof. `arrive → done₀` is valid only if phase 0
can never complete without that arrive. If it could complete early, as in the
premature-completion example below, the consumer could pass the wait before the
producer's write, and the edge would claim an ordering that doesn't always
hold. So membership is proven first (2a, 2b) and the edges are added after
(2c); those edges then make the next phase's proofs possible.

### Example: a passing phase

A barrier is initialized with count 2. Producers P1 and P2 each arrive once,
and consumer C waits on phase 0.

- **a.** Both arrives are HB-after the init.
- **b.** The arrivals sum to 2 and there are no TMA bytes, so no solver call is
  needed.
- **c.** Edges `P1 → done₀`, `P2 → done₀` and `done₀ → C's wait done` are added.
- **Waits.** C's wait done is HB-after `done₀`, and phase 1 has no effects, so C
  observes phase 0 in every schedule.

### Example: a premature completion

One producer thread loads two tiles but arms the second tile late (`*` marks
the effects of phase 0):

```
arrive.expect_tx(64)*  →  TMA A issue  →  A completes −64*
        ↓
   expect_tx(+64)*     →  TMA B issue  →  B completes −64*
```

Suppose the reference run finished both loads before phase 0 completed, so all
four effects are in phase 0 and the balance holds (+128 − 128 bytes, 1 of 1
arrivals). For E = B's completion, the solver finds the ideal
`{arrive.expect_tx, A completes}`. It contains every arrival, it is
downward-closed among the four effects, and its pending byte count is
64 − 64 = 0. Phase 0 can therefore complete before tile B is armed, and a
consumer waiting on phase 0 may read tile B early. Extended to the full graph,
the ideal also includes `TMA A issue`; listed in topological order, it is the
counterexample schedule.

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
