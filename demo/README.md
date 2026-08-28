# Product demo film

`promo.html` — a self-playing ~78-second promotional film for the RAG
Question Generator. It is a single standalone HTML file: no build step, no
server, no backend required.

## Watch it

**Locally:** open `demo/promo.html` in any modern browser (double-click it,
or `open demo/promo.html` on macOS). It starts playing automatically and
loops back to a **Replay** button at the end.

- **Space** — pause / resume
- **R** — replay from the start
- Buttons for the same are in the bottom-right of the frame.

It adapts to the viewer's light/dark theme and honours
`prefers-reduced-motion` (snap cuts instead of eased transitions, no typing
animation).

## What it shows

| # | Scene | Point |
|---|-------|-------|
| 1 | A generic printed question list, stamped "Generic" | Static checklists never ask the second question |
| 2 | The premise | Retrieval-grounded, agentic questioning |
| 3 | Company info typed into the app (`POST /sessions`) | You describe the company in plain text |
| 4 | Retrieval diagram → ChromaDB → matched checklist sections | `similarity_search(startup_info, k=10)` grounds the prompt |
| 5 | The ten ranked questions cascade in with category + priority | Phase 1 output |
| 6 | Founder answers → **LangGraph** `Analyze → Gap found → Generate` (with the repair loop) → a generated follow-up | Phase 2: the follow-up decision |
| 7 | Two more turns; an ARR contradiction is flagged | The thread adapts to what was said |
| 8 | Risk report — score dial, category bars, risks / gaps / recommendations | Phase 3, with deterministic scoring |
| 9 | Closing line — Grounded · Adaptive · Auditable | — |

All labels, category names, risk levels, endpoint names and the score range
are taken from the real code (`app/report_engine/schemas.py`,
`app/report_engine/scorer.py`, `app/api.py`). The company shown ("Swish")
mirrors the placeholder already in `frontend/index.html`. The numbers in the
report scene are illustrative.

## Export to MP4 (optional)

There is no headless render step. To get a video file, screen-record the
browser window while it plays once:

- **macOS:** `Cmd + Shift + 5` → record the browser window → the film runs
  ~78s → stop. Trim the ends in QuickTime if needed.
- Any OS: OBS Studio, a window-capture source, record one loop.

For a clean capture, put the browser in full-screen and reload so the film
starts from scene 1.
