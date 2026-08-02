# OptiQ Lab — UX/GUI Redesign

*Current-state audit → user model → competitive critique → priority list → gap analysis → mitigation plan*

---

## 0. The one-paragraph thesis

mlx-optiq owns a measurement no competitor has: per-layer KL-divergence sensitivity, and the eval harness that turns it into a Capability Score. That measurement is the reason the product exists. The Lab currently **throws it away at the point of use** — it appears once, as a static number on a Hub card, computed by someone else on someone else's calibration mix, and never again touches the surface where you actually change temperature, context length, KV bits, or adapters. Meanwhile the Lab's information architecture is a one-to-one mirror of the Python package's module list (chat / quantize / finetune / dataset / hub / arena), presented as tabs. The user's real unit of work — *"make this model good at my thing, on this machine, and prove it"* — crosses all six. **Any redesign that keeps the tab bar keeps the wonkiness.** The new app should be organized around three surfaces (Work, Models, Runs) sitting on one persistent Machine strip, with measurement made ambient and provenance attached to every token generated.

---

## 1. Current specification — what actually ships today

### 1.1 Runtime shape

| Element | Current implementation |
|---|---|
| Process model | `optiq lab` → Flask app (`optiq.lab.create_app`) on :7860, plus `mlx_lm.server` booted in a daemon thread on :8080 |
| Long jobs | `multiprocessing.Process` workers, progress streamed to the browser over SSE |
| Auth | First launch → `/setup`, pick a password; thereafter `/login`. HF write token encrypted at rest with a Fernet key derived from that password (lose password = lose token) |
| Binding | 127.0.0.1 by default; `--host 0.0.0.0` opt-in, password still enforced |
| Model residency | One model in the server. Single-model-by-default (request `model` field treated as a label); `--allow-model-switch` / `--models-dir` opts into per-request hot-swap |
| Persistence | Saved chats as files under `~/.optiq/lab/chats/` |

### 1.2 Feature surface, by tab

**Chat** — streaming against whatever the server holds. Three sandboxed local tools (`web_search` via DuckDuckGo with a URL-fetch mode; `python` in a three-tier sandbox with AST blocks on `os.system`/signals/network and inline matplotlib capture; `terminal` for bash one-liners with token-aware rejection of `sudo`/`curl`/`rm`). Tool-call healing across six malformed shapes with a `healed` chip. Server-side tool loop capped at 25 turns, per-call retry budget of 3 with a `retry limit` badge, dedup of consecutive identical successful calls, and a forced tools-disabled final re-prompt so a stuck loop still returns text. JSON mode (any-valid-JSON or paste-a-schema) via `lm-format-enforcer`, mutually exclusive with tools. Chat-with-files: BM25 retrieval over attached documents with `[n]` markers and a sources panel. HTML artifacts render a **Run** card that opens a sandboxed `allow-scripts` iframe side panel. Per-message temperature / max tokens / `enable_thinking`. Image attach on vision-sidecar quants. `"N tool calls"` accordion. Stop button that `SIGKILL`s sandbox subprocesses.

**Deep Research** — a toggle inside the Chat composer. TTD-DR draft-centric loop: plan → noisy draft → denoise rounds (draft gaps become searches, verbatim quotes extracted, folded back with `[n]`) → report → verify-and-repair pass over every cited sentence. Trace card fills in live. Short answer in chat, full cited Markdown report in the artifact panel with a `↓ .md` button. Budget knobs for rounds / queries per round / sources per query / report length. Cancellable at step boundaries. Can point at a cloud OpenAI-compatible endpoint with the key held in the browser.

**Hub** — three lists (published OptiQ quants auto-discovered from `mlx-community` with family/size/bit-profile/downloads; HF search filterable to OptiQ quants; local builds detected by `optiq_metadata.json`, with a vision badge for `optiq_vision` sidecars). One-click *load-to-server* (5–30 s hot swap) and *new-chat*.

**Model Arena** — two models side by side with tokens/sec.

**Quantize** — four-step wizard: HF id → BPW + reference mode (`bf16` / `uniform_4bit` / `auto`) → live progress across sensitivity, knapsack, convert → save locally, optional one-click HF push. Architecture detection with a warning on untested families.

**Fine-tune** — MLX-native LoRA over any OptiQ quant, `by_bits` rank scaling default, live train-loss sparkline from mlx-lm's `TrainingCallback`, save adapter + HF push.

**Build dataset** — twelve templates → JSONL (SFT from QA pairs, DPO from preference pairs, style transfer, code completion, self-instruct, format conversion, prompt reconstruction, multi-turn synthesis, tool-use traces, RAG Q/A from documents, CoT synthesis, verified code generation with sandbox-run asserts). LLM-driven templates call the Lab's own API server; reasoning auto-disabled on generation calls.

**Settings** — Server (model load/hot-swap), Integrations (parameterised copy-paste configs for Claude Code, Codex, OpenCode, OpenClaw, Hermes Agent, Mistral Vibe), Hugging Face (token).

### 1.3 What sits in the CLI but *not* in the Lab

This list is the redesign's richest vein — these are already-built capabilities the GUI never exposes:

- `optiq eval` — smoketest (KL + GSM8K-50) and the full six-metric suite (MMLU, GSM8K, IFEval, BFCL, HumanEval sandboxed, HashHop), `--score` → Capability Score, `--reasoning` mode for always-on thinking models
- `optiq benchmark`, `optiq latency --calibrate`
- `optiq kv-cache` — the mixed-precision KV sensitivity pass (+31–62% decode at 64k)
- MTP speculative decoding (`--mtp`, ~1.4×), Gemma-4 `--drafter` spec decoding
- `--idle-timeout` RAM reclaim, `--max-context auto` OOM guard, `--context-scale`, model `:think` / `:no-think` variants
- Multi-adapter in-process hot-swap (`AdapterActivation`, ~50 MB per extra adapter), adapter stacking (`"sft+dpo"`)
- Distributed inference across Macs over Thunderbolt
- Custom `--calibration-mix`

### 1.4 Structural constraints that *cause* the felt wonkiness

1. **Six Flask blueprints presented as six tabs.** No shared object model; a quant built in Quantize does not become a first-class thing that Hub, Eval, Arena, and Chat all reference.
2. **One scarce global resource, six pretenders to it.** Unified memory and the single loaded model are contended by chat, arena (needs two), dataset generation, deep research, and fine-tuning — and the UI gives no one a view of the contention.
3. **Progress is per-page SSE, not a job bus.** Navigate away from a running job and you lose your window into it.
4. **State lives in files and process memory**, not in a queryable store, so nothing is searchable, comparable, or resumable.

---

## 2. Start from the user: who, what job, what loop

### 2.1 Four personas and their jobs-to-be-done

| | Persona | Job | Success looks like |
|---|---|---|---|
| **P1** | **Private Operator** — has a capable Mac, wants a good assistant that never phones home | "Give me a private ChatGPT that is actually good on my hardware" | First useful token < 5 min from install; never has to think about bits |
| **P2** | **Agent Plumber** — points Claude Code / Codex / OpenClaw at a local endpoint | "Serve me a fast, reliable, always-up model my tools can drive" | Endpoint stays up, TTFT is low, context doesn't silently truncate, one place to see health |
| **P3** | **Optimizer** — OptiQ's home turf | "Extract the most capability per GB from *this* machine" | Can see the capability/latency/memory trade of every knob before committing |
| **P4** | **Domain Specialist** — fine-tunes on proprietary data and must defend the result | "Adapt a model to my corpus, prove it didn't get worse, and keep a record" | Reproducible builds, own eval set, per-answer provenance, nothing leaves the machine |

P1 and P2 are the volume. P3 is the differentiator. **P4 is the underserved market** — nobody in the local-LLM space serves the person who has to answer *"how do you know this model is safe for this use?"*, and that answer is exactly what per-layer sensitivity plus a local eval harness can produce.

### 2.2 The real workflow is a loop, not a tab bar

```
        ┌─────────────────────────────────────────────────────┐
        │                                                     │
   ┌────▼─────┐   ┌──────┐   ┌───────┐   ┌─────┐   ┌─────────┐│
   │ ACQUIRE  │──▶│ FIT  │──▶│ SERVE │──▶│ USE │──▶│ MEASURE ││
   │ find a   │   │ will │   │ load  │   │chat/│   │ against ││
   │ model    │   │ it   │   │ it,   │   │agent│   │ MY eval ││
   │          │   │ run  │   │ tune  │   │/res-│   │ set     ││
   │          │   │ here?│   │ knobs │   │earch│   │         ││
   └──────────┘   └──────┘   └───────┘   └─────┘   └────┬────┘│
                                                        │     │
                              ┌─────────┐   ┌───────────▼───┐ │
                              │ PUBLISH │◀──│    IMPROVE    │ │
                              │ share/  │   │ quantize diff-│ │
                              │ pin a   │   │ erently, LoRA,│─┘
                              │ build   │   │ change KV/MTP │
                              └─────────┘   └───────────────┘
```

Today the Lab implements every box and **none of the arrows**. FIT does not exist at all. MEASURE exists only in the CLI. The arrow from MEASURE back to IMPROVE — the loop that makes the whole thing a *workbench* rather than six utilities — is entirely absent.

### 2.3 The local-LLM failure taxonomy (the thing a local GUI exists to make legible)

Cloud chat UIs can assume the model works. Local ones cannot. These are the failures that present to the user as "the model is dumb":

| Failure | Currently surfaced? |
|---|---|
| Silent context truncation | No |
| Wrong / missing chat template | No |
| OOM, or the pre-OOM compressed-memory cliff (~27–28 GB on a 36 GB Mac, 9–30% throughput loss) | No |
| Apple's MTLResource ceiling (~499 K bound resources — hard failure independent of byte headroom) | No |
| Malformed tool call | **Yes** — `healed` chip |
| Tool call looping | **Yes** — `retry limit` badge, 25-turn cap |
| Empty or irrelevant BM25 retrieval | Partially — sources panel shows what came back, not whether it was any good |
| Thinking-token budget consumed before the answer | No |
| Quantization damage to the specific capability you need | No |

OptiQ is the *only* platform in the field that detects any of these. It then treats the detections as debug decoration. **Elevating this taxonomy into a first-class "Run health" surface is the single most defensible UX differentiator available.**

---

## 3. Critique of the comparable platforms

### LM Studio
**Steal:** best-in-class model discovery and download management; per-model saved presets; explicit engine choice (MLX vs llama.cpp); a headless server mode that feels the same as the GUI; the "will it fit?" indicator on the download list.
**Failures:** closed source. Configuration is a wall of sampler sliders with no statement of consequence — you move `top_k` and nothing tells you what it did. No evaluation whatsoever, so quantization choice is folklore ("Q4_K_M is fine"). Fit prediction is file-size arithmetic that ignores KV cache and activations, so it lies at long context. No fine-tuning, no dataset tooling, no eval. The conversation is the only persistent object — no project, no workspace, no experiment. **Core sin: the model is the unit of work, when the task is the unit of work.**

### Ollama
**Steal:** the install-to-first-token time is the industry benchmark; `Modelfile` as declarative, diffable, shareable model config; automatic idle unload; a drop-in OpenAI endpoint nobody has to think about.
**Failures:** the desktop GUI is an afterthought bolted to a CLI product. Defaults are invisible and occasionally wrong in ways with correctness consequences — the years-long 2048-token default context that silently truncated long prompts is the canonical example of a UX bug masquerading as a config default. Quantization hidden behind opaque tags. No introspection into what is actually resident. When the "it just works" abstraction breaks there is **no ladder down** — you go straight from magic to reading Go source.

### Open WebUI
**Steal:** genuine feature depth (RAG, pipelines, functions, model presets, multi-user); web-native so it works from any device on the LAN.
**Failures:** settings sprawl with no information hierarchy — dozens of toggles across a modal maze, discovered by rumor and Reddit. Docker-first install. The admin/user split is overhead for the solo user who is 90% of the audience. RAG has many knobs and gives no feedback about whether retrieval actually worked.

### Jan / GPT4All / Msty / AnythingLLM
- **Jan:** clean and honest, but shallow — thin model management, unfinished extension story.
- **GPT4All:** friendliest onboarding, lowest ceiling; you outgrow it in a week.
- **Msty:** the best pure-UX work in the category — split chats, branching, knowledge stacks, prompt library. Closed, opinionated, no optimization story at all.
- **AnythingLLM:** the **workspace** abstraction is right — documents, model, and settings scoped to a unit of work. But the abstraction leaks: embedder, vector store, and LLM provider are three independent configs that must agree, and when they don't you get silent empty retrieval.

### text-generation-webui / KoboldCpp
**Steal:** nothing is hidden; every parameter is reachable.
**Failures:** the Gradio-tab paradigm — which OptiQ Lab has inherited wholesale. Tabs are a filing cabinet for the developer's modules, not a workflow for the user. Maximal control, zero curation, zero explanation.

### The seven failures every one of them shares

1. **No feedback loop from output quality back to configuration.** You change a number and hope.
2. **Memory is invisible until it is fatal.** You learn the model doesn't fit by watching your Mac swap.
3. **Configuration is global when it should be per-task.** One temperature, one system prompt, one context length, for everything.
4. **No provenance.** No platform can answer *"what exactly produced this answer?"* — build, adapter, quant profile, KV bits, sampler, tools, retrieved chunks.
5. **Chats are the only durable object.** No projects, no datasets, no runs, no experiments, no comparisons.
6. **Failure is silent** and degrades to "the model is bad."
7. **Long work is modal and blocking**, and navigating away loses it.

OptiQ Lab currently shares all seven. It is uniquely positioned to fix 1, 2, 4, and 6 because it already has the underlying machinery.

---

## 4. Needs and wants — prioritized

**Need** = the product is broken without it. **Want** = it wins the category.

### P0 — Needs (nothing else matters until these ship)

| # | Item | Persona | Why |
|---|---|---|---|
| N1 | **Fit predictor.** Before any load: weights + KV at the chosen context + activation headroom vs. free unified memory, as a budget bar with a verdict. Calibrated per machine by a one-time measured pass. | All | The #1 cause of "this app is broken" is an OOM or a swap-thrash the app could have predicted |
| N2 | **Persistent Machine strip.** Always visible: what's loaded, RAM used/free, context window, tok/s, ports, endpoint health, active adapter, running jobs. | All | The one scarce global resource must have one visible owner |
| N3 | **Non-blocking job system.** Every long operation (quantize, train, eval, deep research, dataset gen) is a queued Run with status, logs, artifacts, cancel/resume, and a notification. Navigation never loses it. | P3, P4 | Current per-page SSE makes multi-hour work feel unsafe |
| N4 | **Unified model lifecycle.** One surface where a model has lineage (source → quant profile → adapters → evals), a fit badge, and every action. Kills the Hub / Settings→Server / Quantize triple mental model. | All | Three places to load a model is three places to be confused |
| N5 | **Run health panel.** The §2.3 failure taxonomy surfaced per turn: context used vs. window, template applied, healed calls, retry hits, retrieval quality, thinking tokens spent, memory peak. | All | Turns "the model is dumb" into a diagnosis |
| N6 | **Per-workspace configuration.** System prompt, model, sampler, tools, files, and eval set scoped to a unit of work, not global. | P1, P4 | Global config is the reason people keep six chat apps open |
| N7 | **Modern conversation table stakes.** Edit-and-resend, regenerate, branch, fork-to-compare, pin/prompt library, search across all chats. | P1 | Absent = disqualifying versus Msty and Open WebUI |

### P1 — High-value

| # | Item | Persona |
|---|---|---|
| N8 | **Eval in the GUI.** `optiq eval` promoted to a first-class surface: run smoketest or full suite against a build, see the six metrics, diff two builds side by side. | P3, P4 |
| N9 | **Bring-your-own eval set.** Import a prompt set (or lift one from your own chat history), score builds against *your* task instead of MMLU. | P4 |
| N10 | **Knob consequence preview.** KV bits, MTP, drafter, context-scale, BPW, thinking on/off — each shows predicted Δcapability / Δtok-s / ΔGB before you commit. | P3 |
| N11 | **Provenance envelope on every message**, with a one-click "explain this answer": build hash, quant profile, adapter stack, KV config, sampler, tools, retrieved chunk IDs, healed flags. Exportable. | P4 |
| N12 | **Arena as a mode, not a tab.** Fork any live thread across two builds; scored on latency, tok/s, and (optionally) a judge or your eval rubric. | P3 |
| N13 | **Integration health, not integration snippets.** Show that Claude Code is connected, what context-scale is in effect, last request, error rate — instead of a static copy-paste page. | P2 |
| N14 | **Dataset provenance and preview.** Every generated row traceable to its template, seed, and generating model; inline row editing; verified/unverified filtering surfaced (already computed for verified-code-gen). | P4 |

### P2 — Wants that win the category

| # | Item |
|---|---|
| W1 | **Recipes** — a declarative, diffable, shareable file capturing the whole stack (model + quant profile + KV config + adapter + sampler + tools + system prompt), Ollama's `Modelfile` idea generalized to the full OptiQ surface. Import/export/version. |
| W2 | **Guided optimization** — "I have 64 GB and I need long-context code review": the app proposes a build, predicts the trade, runs the smoketest, and reports back. |
| W3 | **Responsive / remote-first UI** — drive the Mac Mini from an iPad on the couch over Tailscale, with proper auth. |
| W4 | **Regression gate** — re-quantize or fine-tune and get a pass/fail against your saved eval baseline before you promote the build. |
| W5 | **Local-only Deep Research as the marquee** — the privacy story here is genuinely unmatched; give it its own workspace type with a report library. |
| W6 | **Distributed/cluster visualization** — the Thunderbolt sharding path currently has zero GUI. |
| W7 | **Command palette** (⌘K) over models, chats, runs, settings, and actions. |

---

## 5. Critical gap analysis

Ranked by **severity × frequency**.

| # | Gap | Evidence in the current app | Consequence |
|---|---|---|---|
| **G1** | **No fit prediction** | Nothing between "click load" and the model loading | Users discover the memory cliff and the MTLResource ceiling by crashing. The empirical ceiling map already exists — as a table in a doc page, not as code in the UI |
| **G2** | **Measurement severed from use** | Capability Score is a static Hub-card number; `optiq eval` has no GUI at all | The product's entire reason for existing is invisible at the moment of decision. This is the largest strategic gap |
| **G3** | **Tab IA mirrors the codebase, not the loop** | Six blueprints → six tabs; no handoff between them | Every multi-step task requires the user to hold state in their head and re-paste model IDs across tabs |
| **G4** | **No provenance** | Chats are files; no record of build, adapter, KV config, or sampler per turn | Disqualifies the app for any evaluative, regulated, or scientific use — the exact use where local inference is most valuable |
| **G5** | **Contended global resource with no admission control** | Arena needs two models; dataset gen calls the live model; fine-tune wants all the RAM | Nondeterministic slowness and failures that feel like bugs |
| **G6** | **Blocking, non-resumable jobs** | `multiprocessing` + per-page SSE | Multi-hour quantize/train runs feel unsafe; users won't start them |
| **G7** | **Triple model-loading mental model** | Hub load, Settings→Server, Quantize output — three paths, three affordances | The most common action in the app has no single obvious home |
| **G8** | **Silent failures still mostly silent** | Healed/retry chips exist; template, truncation, retrieval quality, and thinking-budget failures do not | Users blame the model and downgrade to a worse-but-simpler tool |
| **G9** | **Config is global** | Per-message temp/max-tokens/thinking, but no per-workspace system prompt, tool policy, or file scope | Cannot maintain two different working contexts |
| **G10** | **Chat lacks table stakes** | No branch, edit-resend, regenerate-with-different-model, cross-chat search | Loses on direct comparison to Msty/Open WebUI regardless of technical superiority |
| **G11** | **Crown-jewel features invisible** | MTP, spec drafter, KV-quant, adapter stacking, context-scale, idle-timeout are CLI-only | The differentiators don't reach the audience most able to appreciate them |
| **G12** | **Auth/setup friction, single-device assumption** | Password at first launch; Fernet key derived from it, so password loss = token loss; 127.0.0.1 default | Painful for the real deployment pattern (headless Mac + laptop/tablet client) and a data-loss trap |

---

## 6. Mitigation plan — the new app

### 6.1 Six design principles

1. **Measurement is ambient.** Every knob that can degrade quality shows its predicted cost next to it. No number changes without a stated consequence.
2. **The machine is a visible, shared, finite resource.** One strip, always on screen, always truthful.
3. **The workflow loop is the navigation.** Objects and their relationships are the IA; modules are an implementation detail.
4. **Nothing blocks.** Every long operation is a Run you can leave and come back to.
5. **Every failure has a name.** No silent degradation, ever. Diagnosis over vibes.
6. **A ladder from magic to metal.** Sensible defaults that just work (Ollama's lesson), with a visible, continuous path down to every parameter (textgen's virtue) — and never a cliff between them.

### 6.2 Architectural moves

**A1 — One state spine.** SQLite workspace database plus an append-only event log. First-class rows: `Workspace`, `Build` (a model + quant profile + KV config), `Adapter`, `Dataset`, `Run`, `Conversation`, `Message`, `Eval`, `Artifact`. Everything addressable, searchable, diffable, exportable. This single change enables G2, G4, G6, and G9 simultaneously.

**A2 — Job bus.** Replace per-page SSE with one WebSocket carrying a typed job stream. Runs execute in worker processes with checkpointing; the UI subscribes rather than owns. Admission control against the memory budget (G5).

**A3 — React SPA over a JSON+SSE/WS API.** Replace multipage Flask with FastAPI (already an optional dep via `mlx-optiq[serve]`) + a React client with a single store (TanStack Query for server state, Zustand for UI state). Same binary, same `optiq lab` command, same ports.

**A4 — The Fit Engine.** A runtime memory model: `weights(bpw, params) + kv(layers, heads, ctx, kv_bits) + activation_headroom(arch, batch) + framework_overhead` versus free unified memory, with the two Apple-specific cliffs encoded as hard constraints (compressed-memory threshold, MTLResource ceiling). Ship with the published empirical ceiling map as priors; refine with a one-time measured calibration on the user's machine. Renders as a budget bar with a verdict: **fits comfortably / fits with degraded throughput / will not fit / will hard-fail on resource count**.

**A5 — Provenance envelope.** Every assistant message carries an immutable record: build ID and quant profile, adapter stack, KV config, sampler, context used vs. window, tools enabled and called, retrieved chunk IDs, healed/retry flags, tok/s, peak memory. Rendered as an inspector drawer; exportable as JSON alongside the transcript.

**A6 — Eval as a service inside the app.** Wrap `optiq eval` as a Run type. Support both the standard suite and user-imported prompt sets. Store results against `Build`. Enable diffs and the regression gate.

### 6.3 Information architecture — three surfaces, one strip

```
┌──────────────────────────────────────────────────────────────────────┐
│ MACHINE STRIP   ● Qwen3.6-27B-OptiQ-4bit  17.5/64 GB  ctx 32k        │
│                 42 tok/s  :8080 ✓  adapter: clinical-v3  ▸ 1 run     │
├────────────┬─────────────────────────────────────────────────────────┤
│            │                                                          │
│  WORK      │   Conversation surface. Modes: Chat · Research ·         │
│  ▸ ws:...  │   Compare. Branch, fork, regenerate. Per-workspace       │
│  ▸ ws:...  │   config in a right rail. Provenance inspector.          │
│            │   Artifacts panel. Run-health strip per turn.            │
│  MODELS    │                                                          │
│            │   Model lifecycle. Every build a card: lineage,          │
│            │   fit badge, capability scores, adapters, actions        │
│  RUNS      │   (load / eval / fine-tune / re-quantize / publish).     │
│            │   Hub search + local builds + published quants, unified. │
│  ⌘K        │                                                          │
│            │   Runs: jobs, traces, evals, datasets, reports.          │
│            │   History, comparison, resume, artifacts.                │
└────────────┴─────────────────────────────────────────────────────────┘
```

Explicitly: **Chat, Deep Research, and Arena stop being tabs.** They are three modes of one conversation object. Arena becomes "fork this thread across two builds." Deep Research becomes "run this turn in research mode." Quantize, Fine-tune, and Eval stop being tabs and become **actions on a Build**. Dataset becomes an object in Runs that a fine-tune action consumes.

### 6.4 Six screens worth specifying first

1. **Load dialog with the Fit bar.** Model, target context, KV bits → budget bar, verdict, predicted tok/s, and a "what if" row for the three knobs that change the answer.
2. **Machine strip + expanded Machine panel.** Resident model, memory breakdown by component, idle-timeout state, endpoint health, active integrations, job queue.
3. **Conversation with the right rail.** Workspace config (model, system prompt, sampler, tools policy, attached files, eval set) and, per turn, the collapsed Run-health chip that expands into the provenance inspector.
4. **Build card.** Lineage graph (base → quant profile with per-layer bit distribution → adapters), Capability Score with per-metric bars, fit badge for this machine, and the action row.
5. **Eval compare.** Two builds, six metrics, plus your own prompt set; delta column; a "promote" gate.
6. **Runs list.** Everything that has ever run, resumable, with artifacts and logs attached.

### 6.5 Phasing

| Phase | Weeks | Content | Ships value even alone |
|---|---|---|---|
| **0 — Spine** | 2 | SQLite state, event log, job bus, provenance capture. No visible UI change. | Enables everything; makes current features resumable |
| **1 — Machine & Models** | 4 | Machine strip, Fit Engine, unified Models surface, kill the triple-load path | Removes the top cause of "it's broken" |
| **2 — Conversation** | 4 | React conversation surface: modes, branch/fork, per-workspace config, provenance inspector, Run-health panel | Reaches table stakes vs. Msty/Open WebUI and passes them on diagnosis |
| **3 — Measurement** | 4 | Eval in the GUI, BYO eval sets, knob-consequence preview, build diff, regression gate | Delivers the actual product thesis |
| **4 — Reach** | 4 | Recipes (export/import), remote/responsive client with real auth, integration health, cluster view | Category-defining |

### 6.6 Success metrics

| Metric | Today (est.) | Target |
|---|---|---|
| Install → first useful token | ~10 min (pip + setup + model pull + load) | < 5 min |
| Loads that OOM or trigger swap-thrash | unmeasured, non-trivial | < 1%, and zero unpredicted |
| Clicks from "quantize a model" to "chat with the result" | 12+ across 3 tabs | ≤ 4, no tab change |
| Long runs abandoned mid-flight | unmeasured | < 5% |
| Assistant messages with a complete provenance record | 0% | 100% |
| Users who have ever run an eval | ~CLI-only minority | > 40% of Lab users |

### 6.7 Two decisions to make before Phase 0

1. **Is the multi-model server story changing?** Arena-as-a-mode and build-vs-build eval both want two models resident. On 64 GB that's viable for most pairs; on 16 GB it isn't. Either commit to sequential execution with clear queueing, or invest in the memory-aware scheduler. Choose now, because it shapes the job bus.
2. **How far does "workspace" go?** The AnythingLLM lesson is that the workspace abstraction is right and leaks when its components can silently disagree. If workspaces own a model, an eval set, files, and a tool policy, the app must guarantee those stay coherent — and say so loudly when a workspace's model is no longer resident.

---

## 7. The one-line version

Stop shipping six utilities in a tab bar and start shipping a loop: **know if it fits, load it, use it, measure it against your own work, improve it, and be able to prove what produced every answer** — all on one machine, with the memory that only mlx-optiq can actually compute.
