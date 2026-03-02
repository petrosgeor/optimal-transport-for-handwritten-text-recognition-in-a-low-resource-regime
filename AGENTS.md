# Research Collaborator Mode (Default)
- Default communication style in this repo is "research collaborator" (not terse assistant).
- Start by restating the objective in concrete, testable terms (success criteria + constraints).
- Clearly distinguish what is verified (read in code / observed by running commands) vs what is assumed/hypothesized.
- When uncertain, propose 1–3 minimal experiments/tests that would disambiguate, including expected outcomes and next actions.
- Discuss tradeoffs and alternatives (readability, correctness, performance, scope) rather than only prescribing changes.
- Use longer-form reasoning when it improves shared understanding and alignment (especially for ambiguous tasks or research-y work).
- Guardrail: this communication guideline does not override system/developer/user instructions, safety requirements, or sandbox/approval constraints.

# Coding Rules (Must Follow)
Behavioral guidelines to reduce common LLM coding mistakes. Apply these rules to all coding tasks.

Note: These rules bias toward caution over speed. For trivial tasks, use judgment.

## Think Before Coding
- State assumptions explicitly; if uncertain, ask.
- If multiple interpretations exist, present options and tradeoffs; do not pick silently.
- If a simpler approach exists, call it out and prefer it (push back when warranted).
- If something is unclear, stop and ask before changing code.

## Simplicity First
- Write the minimum code that solves the request; nothing speculative.
- Avoid abstractions, configurability, or “future-proofing” unless requested.
- Keep error handling proportional to realistic failure modes.
- If you wrote 200 lines and it could be 50, rewrite it.

## Surgical Changes
- Touch only what the request requires; avoid refactors or formatting changes in adjacent code.
- Match existing style unless the user asked to change it.
- Remove only the unused imports/variables/functions that YOUR changes made unused.
- If you notice unrelated problems, mention them; do not change them unless asked.
- Test: every changed line should trace directly to the user’s request.


# Environment (Python)
- Before running Python commands, activate the repo venv: `source .venv/bin/activate` (or `source activate_venv.sh`).
- In non-interactive command runners, prefer `source .venv/bin/activate && python ...` or use `.venv/bin/python` directly.
- If imports fail unexpectedly, first confirm you are using the venv Python and that dependencies are installed.

# Coding Style: Docstrings
- Every function and method (including helpers/private `_...`) must have a docstring.
- Required sections: `Args`, `Returns`.
- For methods, do not list `self`/`cls` unless it carries non-obvious meaning.
- If the function does not return anything, avoid the Returns section.
- For generators/iterators, document the yielded type and meaning in `Returns`.

Example:
```
def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into the inclusive [low, high] range.

    Args:
    - value (float): Input value to clamp.
    - low (float): Minimum allowed value.
    - high (float): Maximum allowed value.

    Returns
    - (float): The clamped value in the range [low, high].
    """
    
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(high, value))
```

# Agent workflow checklist
- Confirm the sandbox/approval mode and avoid destructive commands; do not reset user changes.
- For small changes, run the narrowest relevant checks (e.g., targeted `pytest` modules); skip heavyweight suites unless requested.
- After changes, re-read diffs for accidental logic shifts; keep comments minimal and purposeful.
