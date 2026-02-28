# RBE 549 — Computer Vision

## Assignment Workflow

Each weekly assignment follows this sequence:

1. **Ingest context** — Read the assignment requirements, existing codebase, and any prior plan files. Summarize understanding before proceeding.
2. **Plan** — Use plan mode. Break the assignment into the smallest verifiable increments. Identify deliverables, dependencies, and commit milestones.
3. **Sequence tasks** — Create a task list tracking each increment. Each increment follows the cycle below.

### Per-Increment Cycle

4. **Briefer** — Explain the concept or algorithm being implemented. Cover the theory, the "why," and any subtleties relevant to this specific increment. Do not write code yet.
5. **Wait for author** — Stop and let the author ask questions, push back, or request deeper explanation. Do not proceed until the author signals readiness.
6. **Drafter** — Write the code for this increment. Minimal, verifiable. One function or one logical unit at a time. After writing, run `black` on the modified file before proceeding.
7. **Critic** — Analyze the increment: correctness, edge cases, whether it matches the Briefer's explanation, and anything report-worthy. Flag concerns. Code must already be Black-formatted before this step.
8. **Wait for author** — Stop and let the author review the code, run it, and verify. The student decides when to commit. Do not auto-commit or proceed to the next increment.
9. **Iterate** — If the author requests changes, apply them and return to step 7 (Critic). Once the author commits, move to the next increment (step 4).

### Key Rules

- Never skip the Briefer step. Understanding precedes implementation.
- Never bundle multiple increments. One concept, one function, one commit.
- Never auto-commit or suggest committing. The student commits when ready.
- The Critic notes observations for the lab report in memory (`week{N}-report-notes.md`).

## Code Style (Python, project-specific)

- Single-line comments only at module level (section dividers, constant explanations).
- Docstrings: one sentence. No args/returns blocks. No multi-line elaboration.
- No inline comments explaining obvious code. If the code needs a comment to be understood, refactor the code.
- Constants at the top of the module. Tunable parameters exposed as top-level constants.
- Follow existing patterns in the codebase (init_state, setup_trackbars, handle_key, apply_effects).

## Commit Convention

- Format: `Week N: <imperative phrase>`
- Example: `Week 5: add SIFT detection and BF matching`

## File Organization

- Each lab is a standalone module (`labN.py`) with camera interface functions.
- `camera.py` orchestrates all labs via a unified state dict.
- Lab modules with `__main__` can run standalone.
- Report notes go in memory, not in the repo.
