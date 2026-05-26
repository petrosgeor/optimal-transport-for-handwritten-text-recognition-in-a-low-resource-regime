# Repo Context
This repo explores handwritten text recognition with an alignment-driven pseudo-labeling loop. The core experiment trains/refines an HTR backbone, trains a visual-to-word projector, uses OT alignment to pseudo-label more samples, and repeats.

The main flow is:

1. `HTRDataset` loads processed word images, transcriptions, initial aligned seeds, lexicon priors, and word embeddings.
2. `HTRNet` predicts CTC logits and optional descriptor heads such as PHOC.
3. `Projector` maps harvested visual features into the word-embedding space.
4. OT alignment assigns high-confidence pseudo-labels to currently unaligned samples.
5. The trainer logs pseudo-labels and test-set CER/WER after each round.

Important files:

- `alignment/trainer.py`: Main alternating refinement entrypoint.
- `alignment/alignment_utilities.py`: OT alignment, candidate selection, and pseudo-label assignment.
- `alignment/losses.py`: Projection losses and standalone OT/GW coupling helpers.
- `alignment/eval.py`: CER/WER evaluation.
- `alignment/utils/`: Shared helpers for CTC, checkpoint loading, reporting, and numeric checks.
- `htr_base/models.py`: `HTRNet` and `Projector`.
- `htr_base/utils/htr_dataset.py`: Dataset loading, initial seed selection, lexicon priors, and word embeddings.
- `htr_base/utils/phoc.py`: PHOC target construction.
- `alignment/alignment_configs/*.yaml`: Experiment behavior is heavily config-driven.

Experiment notes:

- `dataset.n_aligned` controls how many initially labeled/aligned seed samples are used.
- `dataset.word_prob_strategy` controls lexicon prior probabilities: `empirical`, `wordfreq`, or `uniform`.
- `results/`, `htr_base/data/`, and checkpoints are local/generated artifacts and are usually git-ignored.
- Round metrics files are TSV rows with: `round_idx`, `correct_pseudo`, `cer_test`, `wer_test`.
- CER is logged as a fraction. WER is logged as a percentage.

Common pitfalls:

- Do not infer behavior from result filenames alone; verify the active YAML config and code path.
- Do not reintroduce projector ensemble or variance-based pseudo-label logic unless explicitly requested.
- Prefer narrow, targeted tests/checks because full training runs can be expensive.

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
- Every function and method (including helpers/private `_...`) must have a docstring, following the Google style docstrings.
- Required sections: `Args`, `Returns`. 
- For methods, do not list `self`/`cls` unless it carries non-obvious meaning.
- If the function does not return anything, avoid the `Returns` section completely.
- For generators/iterators, document the yielded type and meaning in `Returns`.

Example:
```
def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into the inclusive [low, high] range.

    Args:
        value (float): Input value to clamp.
        low (float): Minimum allowed value.
        high (float): Maximum allowed value.

    Returns
        (float): The clamped value in the range [low, high].
    """
    
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(high, value))
```


# Agent workflow checklist
- Confirm the sandbox/approval mode and avoid destructive commands; do not reset user changes.
- For small changes, run the narrowest relevant checks (e.g., targeted `pytest` modules); skip heavyweight suites unless requested.
- After changes, re-read diffs for accidental logic shifts; keep comments minimal and purposeful.
