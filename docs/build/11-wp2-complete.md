# WP-2 complete — Conversation, health, Runs

**Date:** 2026-08-02  
**Tests:** full `tests/lab` suite green after WP-2

---

## Delivered

| Item | Detail |
|------|--------|
| **2A Workspaces** | Workspace selector on Chat; create/load; system prompt + sampler; coherence banner vs resident model |
| **2B Run health** | SSE `health` event; chips under assistant (healed, empty retrieval, chunks, retries, tok/s); provenance drawer + Export JSON |
| **2C Table stakes** | Search chats (`GET /api/chats?q=`), Branch, Regenerate, Edit & resend |
| **2D Runs** | `/runs` page, `GET /api/runs`, cancel, log tail; sidebar **Runs** |
| **2E Research mode** | Explicit Chat / Research segmented control (existing deep research stream) |

### Key paths

- `optiq/lab/routes/runs.py`, `templates/runs.html`
- `optiq/lab/chat_store.py` — `search_chat_records`
- `optiq/lab/routes/chat.py` — health SSE + search
- `optiq/lab/templates/chat.html` — WP-2 UI

### Gaps improved

| Gap | Status after WP-2 |
|-----|-------------------|
| G6 Lost jobs in UI | 🟢 Global Runs page |
| G4 Provenance productization | 🟡 UI + save path; full server completeness still partial |
| G8 Silent failures | 🟡 Chips for healed/retrieval/retries; not full taxonomy |
| G9 Workspaces | 🟡 UX + API; not full multi-workspace IA redesign |
| G10 Chat table stakes | 🟡 Search, branch, regen, edit-resend (no full tree UI) |
| G3 Work surface | 🟡 Chat/Research modes; Models/Runs in nav |

---

## Still open (WP-3+)

- **G2** Eval GUI / BYO eval / promote gate  
- Full silent-failure taxonomy (truncation, thinking budget) when measurable  
- SPA three-column IA (still Flask tabs + new primary nav)

## Verify

```bash
cd mlx-optiq-dev && pytest tests/lab -v
```
