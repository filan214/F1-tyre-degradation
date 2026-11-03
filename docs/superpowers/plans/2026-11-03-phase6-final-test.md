# Phase 6: final 2025 evaluation

Scope: implement the user's prescribed final evaluation only. The saved Phase 5
choice (RF) and Phase 4 baseline (B1) are fixed. Do not edit features, model
definitions, hyperparameters, fitted pipelines, or prepared inputs.

1. Test the opening marker, rerun refusal/explicit deviation, failure retention,
   saved prediction offsets, common evaluation rows, and negative-result logging
   with synthetic fixtures in temporary directories.
2. Implement a thin script 06 and reusable final-evaluation orchestration. Load
   existing pipelines; create the marker exclusively before reading test rows.
   Evaluate all three models and four baselines. Persist predictions from this
   single pass, per-race MAEs, the summary, fig09 and full experiment results.
3. Run the complete test suite and an independent read-only review before
   opening the real holdout. Resolve any correctness issues first.
4. Run script 06 exactly once without a rerun flag. Reconcile saved outputs
   without another prediction pass; inspect fig09 and compare frozen-input
   hashes before/after. Record completion and leave Phase 7 untouched.

No new model selection is permitted from the final results. Negative results
remain visible. An existing marker blocks the authorized first run; an actual
failure after opening retains the marker and is never automatically retried.
