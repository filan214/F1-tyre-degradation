# Phase 8: portfolio README and reproducibility evidence

Use the existing PRD and guide Phase 8 specification. Keep models, features,
hyperparameters, original predictions and original opening marker frozen.

1. Recompute headline metrics from saved predictions and reconcile tables; use
   these measured values to write the README in the prescribed eleven-section
   order with relative figure links, honest limitations and PowerShell commands.
2. Add the two deferred numerical edge-case tests (negative gain and zero
   baseline MAE), run all tests and dependency checks, verify packaged archive
   metadata and local documentation links.
3. Exercise scripts 01–07 in an isolated source copy starting with only verified
   raw cache inputs. Reuse the pinned installed environment; never overwrite the
   original models or results. Preserve the original test opening in the copy
   and explicitly mark any evaluation as a reproduction rerun, not a fresh test.
4. Compare reproduced outputs, record exact verification scope and limitations,
   update PRD checks supported by evidence, append the final results/portfolio
   bullet and Phase 8 entry without rewriting experiment history. Request one
   independent review of the final documentation and checks.

Constraint observed before execution: no Git metadata exists in this supplied
workspace. A pinned raw GitHub input download failed with WinError 10061 under
this session's network restrictions. A literal network-backed fresh clone and
clean dependency installation cannot be claimed from cached-copy verification.
The fresh-clone DoD item stays unchecked unless independent evidence proves it.
This is a verification limitation, not permission to invent success.

Execution: README rewritten in eleven sections; metrics recomputed for seven predictors; 153 tests passed in original and clean offline environments; pip check and package/catalog checks passed. Scripts 01–07 completed in the cached source copy; 23 CSVs, four Parquets, CV results and selection comparisons passed. All 13 original frozen hashes are unchanged. Independent review found no material issues. Ruling: literal fresh-clone verification remains unverified because there is no Git metadata and the network probe failed; leave that DoD item open. Cost: online fresh-clone reproducibility cannot be claimed. Both previously deferred edge-case tests now pass.
