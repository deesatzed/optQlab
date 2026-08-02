# What’s missing — plain language (non‑technical)

**Date:** 2026-08-02  
**For:** product, design, stakeholders  
**Detail plan:** `docs/plans/2026-08-02-gap-mitigation-plan.md`

---

## What we’re trying to be

A private AI workbench on your Mac that:

1. Tells you **if a model fits** before it breaks the machine  
2. Lets you **use** it (chat, research, tools)  
3. **Tests** whether it’s still good at *your* work  
4. **Shows why** it answered the way it did  

---

## What’s missing today (important)

### Trust & “will it work?”

| Missing | What you feel instead |
|---------|------------------------|
| Clear **fit check** before load | Surprise slowdown or crash |
| Always-visible **machine status** | “Why is it so slow?” with no answer |
| **One place** to load a model | Hunting Hub vs Settings vs after quantize |

### Understanding answers

| Missing | What you feel instead |
|---------|------------------------|
| Tags that name problems (context full, no docs found, etc.) | “The model is dumb” |
| **Why this answer?** (which model version, settings, files) | No audit trail |

### Proving quality

| Missing | What you feel instead |
|---------|------------------------|
| Run tests from the app (not only expert tools) | Guessing after shrinking a model |
| Test on **my own questions** | Generic scores that don’t match my job |
| Block “promote this model” if tests got worse | Silent quality loss |

### Everyday comfort

| Missing | What you feel instead |
|---------|------------------------|
| Modern chat (edit, branch, search) | Other apps feel smoother |
| Long jobs that stay visible if you leave the page | Afraid to quantize overnight |
| Separate **workspaces** for different jobs | One global setup for everything |

### Privacy research & power features

| Missing | What you feel instead |
|---------|------------------------|
| Deep research as a polished product | Feature exists but doesn’t feel flagship |
| Advanced speed/memory features in the UI | Hidden unless someone knows the command line |

---

## What we already fixed underneath (you may not see it yet)

- A real database for chats, models, jobs, and history  
- Safer job queue (one heavy job at a time)  
- Plumbing to store “why this answer” data  

**You still need the new screens** (Machine strip, Fit dialog, Work/Models/Runs, Eval) before this feels like a finished product.

---

## How we fix it (simple roadmap)

| Step | What you get |
|------|----------------|
| **Now** | Honest demos (no fake scores looking real) |
| **1 — Machine & Models** | Fit check, status bar, one Models page, one Load button |
| **2 — Work** | Workspaces, health tags, explain answer, modern chat, job history |
| **3 — Measure** | Eval in the app, your own test set, don’t promote worse models |
| **4 — Reach** | Shareable setups, remote use safely, clearer agent connections |

---

## Bottom line

| | |
|--|--|
| **Biggest missing piece** | Making OptiQ’s *measurement* show up where you decide (load, chat, promote) |
| **Biggest everyday missing piece** | Fit + trust (will it run? what went wrong?) |
| **Not missing** | The underlying engine ability — the **product wrapping** is the gap |

Full technical mitigation plan: **`docs/plans/2026-08-02-gap-mitigation-plan.md`**.
