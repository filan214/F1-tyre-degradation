# CLAUDE_CODE_PROMPTS — session prompts

The prompts for Sessions 0–8 were not in this folder when this file was created (2026-11-03), so only Session 9 is recorded here. See `EXPERIMENT_LOG.md`, Deviations.

Paste a prompt into a new Claude Code session opened in the project folder.

## Session 9 — build the Streamlit app

```
Read CLAUDE.md, PRD.md, IMPLEMENTATION_GUIDE.md and EXPERIMENT_LOG.md first. Phases 0–8 are complete and the 2025 test set is already opened. This session adds Phase 9: a read-only Streamlit app. Do not change any feature, model, selection, prediction or Phase 0–8 output.

Context:
- Data comes from the pinned TracingInsights archive (see Deviations), not live FastF1 downloads.
- Final results are in reports/tables/ (committed) and data/processed/test_predictions.parquet (gitignored).
- The app must read every number from files. Never hard-code metrics.

Rule exceptions I am explicitly authorising for this session (they override the matching CLAUDE.md rules for Phase 9 only):
- Stretch work: Phase 9 may start although the PRD §9 fresh-clone checkbox is still open (it is closed in the next session, after the repo is on GitHub).
- Dependency: add streamlit as a new optional extra in pyproject.toml ([project.optional-dependencies] app = ["streamlit==<exact version>"]). Core and dev dependencies stay unchanged.
- Test-set access: scripts/08_export_app_data.py may read the already-saved Phase 6 outputs (test_predictions.parquet, test tables). It must not load models, predict, refit or change any Phase 0–8 file.
- Committed data: app_data/ (outside data/) is committed on purpose. It holds only derived 2025 test predictions and result tables, no raw archive files. data/, models/ and caches stay gitignored.
- Paths: the export script takes APP_DATA_DIR from config.py (add it). The app itself must not import f1pace, so it resolves app_data/ relative to its own file location; no absolute paths.

Tasks:
1. Docs first.
   - CLAUDE.md: add a short "Phase 9 exceptions" section recording the five exceptions above.
   - IMPLEMENTATION_GUIDE.md §12: add a "Phase 9 — Streamlit app" section with a DoD.
   - CLAUDE_CODE_PROMPTS.md: add a Session 9 prompt.
   - PRD.md §13: add one line noting the app is a post-v1 stretch item (dashboards were a v1 non-goal).
   - EXPERIMENT_LOG.md: log all of this as a deviation, and fill the empty "Key results summary" table using only numbers already logged in Phases 4–6.

2. Export: scripts/08_export_app_data.py (thin), with logic in src/f1pace/app_data.py.
   - Inspect the real columns of test_predictions.parquet first. Stop and ask if anything below is missing.
   - Write app_data/test_predictions.parquet with only what the app needs: race_id, event_name, circuit, driver, team, lap_number, stint, compound, tyre_age, lap_time_s, pred_b1, pred_rf (also pred_b2–b4, pred_ridge, pred_xgb if the total stays under 5 MB).
   - Copy the small tables the app uses into app_data/: test_summary.csv, test_results_by_race.csv, error_by_tyre_age_bucket.csv, error_by_compound.csv, error_by_stint_phase.csv, error_by_slope_group.csv, error_by_circuit.csv, cv_summary.csv.
   - Write app_data/manifest.json: source commits, row counts, SHA-256 of every exported file, export timestamp.
   - Assert: 13,718 rows, 18 races, no 2023–2024 rows, no missing predictions, and per-race MAE recomputed from the export equals test_results_by_race.csv (atol 1e-12).
   - app_data/ must be committable (adjust .gitignore only if needed); total size under 10 MB.

3. App: app/streamlit_app.py (entry point) plus app/requirements.txt (used by Streamlit Community Cloud) pinning exact versions of only streamlit, pandas and pyarrow, matching the versions in pyproject.toml. Use Streamlit's built-in charts or Altair (ships with Streamlit). The app must not import fastf1, sklearn, xgboost, joblib or the f1pace package, must not load models, and must not download anything. Use @st.cache_data for reads. Pages (st.navigation, or a sidebar radio if unavailable):
   a. Overview: 3-sentence project summary; headline metric cards (model MAE, baseline MAE, improvement %, races won/lost) read from test_summary.csv; the full seven-predictor table; link to the GitHub README.
   b. Race explorer: pick race (show event name) and driver; chart of actual lap time vs pred_rf vs pred_b1 by lap number with stints marked; this race's per-race MAE; note that removed laps (pit, safety car, etc.) are not shown.
   c. Where the model helps: charts and tables from the error_by_* files, showing losses as clearly as wins (young tyres, Monaco and other losing circuits).
   d. Method and limits: honest framing from PRD §1; data attribution (FastF1-derived TracingInsights archive, linked pinned commits); "2025 held-out evaluation, not a live forecast"; descriptive-not-causal caveat.
   Every chart: title, units in seconds, source note "Data: FastF1-derived TracingInsights archive".

4. Tests (offline): tests/test_app_data.py covering the export assertions on a synthetic fixture; tests/test_streamlit_app.py using streamlit.testing.v1.AppTest to run every page against app_data/ and assert no exceptions. Run the full pytest suite.

5. Give me single-line PowerShell commands to install the app extra into the existing .venv (pip install -e ".[dev,app]") and launch with streamlit run app/streamlit_app.py. Run it if you can and confirm every page renders.

6. Read the licence of the TracingInsights 2023/2024/2025 repositories and report whether redistributing derived lap times and predictions in a public app is allowed. If it is unclear or not allowed, say so plainly; deployment waits for my decision.

7. Append a Phase 9 entry to EXPERIMENT_LOG.md: commands, exported file sizes and hashes, pytest count, reconciliation result, licence finding.

Do not create a git repository or deploy in this session.
```
