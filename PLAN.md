# Plan: correct the optimisation-criterion description in the LK methods

## Goal
`LucasKanade` and `DirectionalLucasKanade` claim to use the Zero Normalized
Cross Correlation (ZNCC) criterion, but both minimise the plain sum of squared
differences (SSD) between the current subset and the interpolated reference.
Make the documentation match the code. No behaviour changes.

## Steps
1. **Fix class docstrings.**
   Files: `pyidi/methods/_lucas_kanade.py`, `pyidi/methods/_directional_lucas_kanade.py`.
   Replace the ZNCC wording with "sum of squared differences (SSD)" and note
   that the method is therefore sensitive to brightness/contrast changes.
   Done when: `grep -rn "Zero Normalized" pyidi/methods/_*lucas_kanade.py` is empty.

2. **Clarify the `tol` parameter.**
   LK compares `tol` with the update norm ‖Δ‖; directional LK compares it with
   the squared scalar step. Document this in each `configure()` docstring.
   Done when: both docstrings state what `tol` is compared against.

3. **Check the docs.**
   Search `docs/source/` and `README.md` for ZNCC/"normalized cross" claims
   about LK and correct them.
   Done when: grep is clean, or every remaining hit is about `_dic.py` (ZNSSD).

4. **Verify.**
   Run `pytest tests/test_lucas_kanade_numba.py tests/test_directional_lucas_kanade_numba.py -q`
   and the flake8 command from CLAUDE.md.
   Done when: both pass.

5. **Changelog.**
   Add a "Documentation" entry to `CHANGELOG.md` under the unreleased section.

## Constraints
- Docstrings, docs and changelog only. Don't touch the algorithm or `_lk_kernels.py`.
- One commit per step.

## Out of scope (possible follow-up plan)
- Actually implementing a ZNSSD criterion in both LK methods (needs parity
  changes in `_lk_kernels.py` and new tests).

## Progress
Track in PROGRESS.md, following the rules in CLAUDE.md.
