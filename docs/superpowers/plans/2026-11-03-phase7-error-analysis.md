# Phase 7: saved-prediction error analysis

Implement the existing IMPLEMENTATION_GUIDE.md sections 9–11 only. Inputs are
the saved Phase 6 predictions, completion marker, frozen selection, and per-race
results. No model load, prediction, fitting, feature rebuilding or selection.

1. Add tested `evaluate.breakdown(df, pred_col, by)` with pooled lap MAE,
   equal-race mean MAE, and lap/race counts. Reject missing groups and invalid
   predictions so comparisons use identical rows.
2. Add analysis grouping and paired comparisons. Use configured tyre-age edges
   [0, 5, 10, 20, 30, 100]; reject out-of-range ages rather than dropping rows.
   Early/late proxy: <10 versus >=10 strictly earlier clean stint laps.
   Low/high signed trend: <=0 versus >0 s/lap. These are descriptive reporting
   groups fixed before examining their errors, not new model features.
3. Implement thin script 07 reading only saved predictions and metadata. Require
   the completed test marker and agreement with Phase 6 per-race scores. Save
   five paired tables (age, compound, circuit, clean-history phase, slope) and
   figures 10/11. Generate seven findings with support counts and caveats.
4. Run synthetic tests, the full suite, and a focused independent review; execute
   script 07, inspect both figures, reconcile saved tables and confirm frozen
   hashes unchanged. Append measured results and verification to the log.

Review focus: boundaries, race versus lap weighting, missing groups, fixed RF/B1
choice, descriptive rather than causal claims, marker untouched, no Phase 8 work.

Execution note: this workspace has no Git metadata, so Git-dependent worktree,
commit and ledger helper commands do not apply. Work in the shared authorized
workspace; this file records progress.

Progress: complete. Eight new tests observed failing before implementation and passing after; full suite 151 passed. Script 07 completed once. All five tables independently reconciled; both figures inspected; 13 frozen hashes unchanged. Independent review: no material issues. Deferred minor: explicit unit cases for negative gains and zero baseline MAE. Physical stint age is not inferred from cleaned history; the documented clean-history proxy is retained. Phase 8 untouched.
