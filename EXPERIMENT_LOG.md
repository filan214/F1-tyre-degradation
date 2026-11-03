# EXPERIMENT_LOG — F1 Race Pace Forecasting

Rules for this file (see `CLAUDE.md`):

- **Append only.** Add new entries at the bottom of the relevant section; never rewrite history.
- **Only numbers produced by code that was actually run.** Include the command used.
- Negative results stay in.
- Any divergence from `IMPLEMENTATION_GUIDE.md` goes in **Deviations**.

---

## Environment

| Item | Value |
|---|---|
| OS / shell | Windows / PowerShell |
| Python | _fill in Phase 0_ |
| fastf1 | _fill in Phase 0_ |
| pandas | _fill in Phase 0_ |
| numpy | _fill in Phase 0_ |
| scikit-learn | _fill in Phase 0_ |
| xgboost | _fill in Phase 0_ |
| matplotlib | _fill in Phase 0_ |
| pyarrow | _fill in Phase 0_ |

### 2026-11-02 — Phase 0 — installed environment

The original unfilled template above is retained under the append-only rule.
This snapshot records the installed `.venv` after setup and compatibility checks.

| Item | Installed value |
|---|---|
| OS / shell | Windows / PowerShell |
| Python | 3.13.2 |
| fastf1 | 3.8.3 |
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| xgboost | 3.2.0 |
| matplotlib | 3.11.0 |
| pyarrow | 24.0.0 |
| pytest (dev dependency) | 9.1.1 |
| pip | 24.0 |

Version commands actually run:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip freeze | Tee-Object -FilePath 'data\setup\pip-freeze-phase0.txt'
.\.venv\Scripts\Activate.ps1; pip freeze | Tee-Object -FilePath 'data\setup\pip-freeze-phase0.txt'; python -c "import importlib.metadata as m; print('NumPy Python requirement:', m.metadata('numpy').get('Requires-Python'))"
```

The first install resolved NumPy 2.5.3, whose installed metadata required Python
>=3.12. To retain the guide's `requires-python = ">=3.11"`, NumPy 2.4.6 was
installed from cache and `pip freeze` was run again before pinning. Its metadata
reports `Requires-Python: >=3.11`. The initial freeze is retained locally at
`data/setup/pip-freeze-first-install.txt`; the final freeze is at
`data/setup/pip-freeze-phase0.txt`. These environment artifacts are gitignored.
Only the seven requested runtime libraries are pinned in `pyproject.toml`.

### 2026-11-02 — Phase 1 environment check

Command: `.\.venv\Scripts\python.exe -m pip --version`

Observed pip: **26.2.2**, following the user's terminal upgrade. The seven pinned
runtime distributions still match the Phase 0 snapshot. The obsolete `~ip/` and
`~ip-24.0.dist-info/` upgrade backups were moved, without deletion, into
`.venv/pip-upgrade-leftovers/`; a subsequent dependency check produced no invalid
distribution warning.

---

## Entry template

Copy this for each run:

```
### YYYY-MM-DD — Phase N — <short title>
Command: <exact single-line PowerShell command>
Data: <races / split / row counts used>
Results:
<table or bullet list of measured numbers>
Observations:
- <what the numbers or figures show, in plain language>
Next:
- <next action>
```

---

## Entries

### 2026-11-02 — Planning — project initialised
Command: none (planning only)
Data: none downloaded yet
Results: none. No code, dataset, trained model or measured results exist.
Observations:
- Scope fixed in PRD.md: next-lap prediction, dry races, train 2023–2024, test 2025, race-held-out evaluation.
- Model target is `target_delta = lap_time_s − mean_last3`. Errors are reported as lap-time MAE (Mean Absolute Error) in seconds.
- Expected strongest baseline: B4 stint trend. This is a hypothesis to verify, not a result.
Next:
- Session 0 (setup), then Session 1 (Milestone 1: one race, Bahrain 2024 candidate, to be confirmed dry).

### 2026-11-02 — Phase 0 — package scaffold and configuration verified

Commands actually run (each line is a separate PowerShell invocation):

```powershell
py -3.11 -m venv .venv
python -m venv --upgrade .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PIP_NO_INDEX = '0'; .\.venv\Scripts\python.exe -m pip install -e ".[dev]" --timeout 15 --retries 0 --cache-dir .venv\pip-cache
$env:PIP_FIND_LINKS = (python -c "from pathlib import Path; print(Path('data/setup/wheelhouse').resolve().as_uri())"); .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PIP_FIND_LINKS = (python -c "from pathlib import Path; print(Path('data/setup/wheelhouse').resolve().as_uri())"); .\.venv\Scripts\python.exe -m pip install numpy==2.4.6
.\.venv\Scripts\Activate.ps1; $env:PIP_FIND_LINKS = (python -c "from pathlib import Path; print(Path('data/setup/wheelhouse').resolve().as_uri())"); pip install -e ".[dev]"
```

Data: no race data downloaded or inspected; no features, predictions, model fits,
figures, or evaluation metrics were produced. `data/setup/` contains only local
environment artifacts and cached installation wheels.

Results:

- Created the §1 scaffold; later-phase modules, scripts, and tests contain only
  module docstrings. `config.py` implements §2.
- The initial isolated configuration test run observed 9 failures before the
  paths and `ensure_dirs()` were implemented. After implementation, the same
  tests passed. Initial tests used the available pytest via `PYTHONPATH` under
  Python 3.11 while installation was blocked; final tests use the installed
  editable package and pytest directly from `.venv` under Python 3.13.2.
- Editable installation succeeded both before and after the final seven pins
  were written, using cached wheels and the local sandbox compatibility hook.
- `python -m pytest -q`: 9 passed. Tests check all seven path types and directory
  creation with `tmp_path` and `monkeypatch`, both absent and existing
  directories, repeated calls, and preservation of an existing file.
- `python -m pip check`: `No broken requirements found.`
- All seven pins match installed distribution versions; importing `fastf1`,
  `pandas`, `numpy`, `sklearn`, `xgboost`, `matplotlib`, and `pyarrow` succeeded.
- Exact installed versions are recorded in Environment above. No race-pace
  results have been measured.

Observations:

- The folder has no Git metadata. Work was performed in the requested workspace;
  no repository, branch, or worktree was created.
- Network and temporary-directory restrictions required local environment setup
  adjustments; see Deviations below. The pip upgrade retained pip 24.0 while
  package-index access was disabled.

Next:

- Phase 0 is the stopping point for this session. Phase 1 remains unimplemented.


### 2026-11-02 — Phase 1 — exploration attempt failed: 2024_01

Timestamp: 2026-11-02T17:50:11+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/01_explore_one_race.py --year 2024 --round 1`

Results: DataNotLoadedError: The data you are trying to access has not been loaded yet. See `Session.load`

No complete real-data exploration result was produced. The Phase 1 real-data DoD remains unmet; no audit counts or figures are inferred from this failure.

Next: resolve the reported error, then rerun the same command. Do not start Phase 2.


### 2026-11-02 — Phase 1 — exploration attempt failed: 2024_01

Timestamp: 2026-11-02T17:51:42+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts\01_explore_one_race.py --year 2024 --round 1`

Results: DataNotLoadedError: The data you are trying to access has not been loaded yet. See `Session.load`

No complete real-data exploration result was produced. The Phase 1 real-data DoD remains unmet; no audit counts or figures are inferred from this failure.

Next: resolve the reported error, then rerun the same command. Do not start Phase 2.

### 2026-11-02 — Phase 1 — code verification and archive-access diagnosis

Commands actually run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\01_explore_one_race.py --help
.\.venv\Scripts\python.exe scripts\01_explore_one_race.py --year 2025 --round 1
.\.venv\Scripts\python.exe scripts\01_explore_one_race.py --year 2024 --round 1
```

Data: synthetic unit-test fixtures and actual failed FastF1 download attempts.
No complete real race timing frame was loaded; there are no real audit counts,
race figures, feature datasets, model fits, or prediction metrics to report.

Results:

- **54 tests passed.** New behavior was tested before implementation. Coverage
  includes loading flags/schema, backward weather alignment, C1–C10 first-removal
  attribution, additive counts, lap identity, sanity checks, PNG export,
  append-only experiment records, failure reporting, and the locked-year guard.
- `pip check`: **No broken requirements found.** Runtime pins are unchanged.
- The CLI accepts only 2023 and 2024; passing 2025 exits with argument error 2.
- All three figure layouts were rendered and visually inspected using explicitly
  labelled synthetic fixtures under ignored `.venv/phase1-preview/`, rather than
  the real reports directories. Annotation headroom was adjusted after review.
- The 2024 schedule was initially unavailable in this restricted process; later
  runs selected Bahrain round 1 once schedule/results responses were present.
  The user's normal-terminal run still could not load timing, weather, and race
  messages from either archive source. This process separately recorded a proxy
  refusal at `127.0.0.1:9` (WinError 10061).
- A regression test reproduced duplicate FastF1 console records and leaked DEBUG
  tracebacks; disabling FastF1-to-root propagation fixed that configuration.
  Missing timing now reports an archive/network access explanation instead of
  exposing a `DataNotLoadedError` property getter failure.

Observations:

- Passing synthetic tests verifies the transformations and orchestration; it
  does not establish real race eligibility or fulfill the real-data DoD.
- The installed lap schema contains all guide-required columns. No source
  column has been silently renamed or replaced.
- The current archive access requirement contradicts the planning assumption of
  free fresh timing downloads. The version pin and zero-cost constraint are
  preserved pending the user's data-access decision.

Next:

- Resolve archive access, rerun Bahrain through FastF1, then inspect the real
  figures and the automatically generated audit/log entry. Remain in Phase 1.


### 2026-11-02 — Phase 1 — One-race exploration: 2024_01

Timestamp: 2026-11-02T18:45:42+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts\01_explore_one_race.py --year 2024 --round 1`

Data: FastF1-derived TracingInsights archive (training split); Bahrain Grand Prix; 1129 raw laps, 988 clean laps.

Source: [TracingInsights/2024](https://github.com/TracingInsights/2024), commit `d1543a71370d9b617b627319f8ad80b2f85dfd5e`.

Raw inputs are verified by SHA-256 and cached under `data/raw/tracinginsights/2024_01/`.

| Source file | SHA-256 |
|---|---|
| [session_laptimes.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/session_laptimes.json) | `508f22e47936ac8de631e21524ed7b9808c1d7e0e52729d22dc5162bf8e04aae` |
| [weather.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/weather.json) | `50d14901d8b3982dd55190ba2467167c3593d63ed0e1167a3b379baba6483975` |
| [rcm.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/rcm.json) | `9c5b3cf273d2a2f25235f0e137e1505deb3fc5c50bd8424e98db2cc29ad64994` |
| [drivers.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/drivers.json) | `eb28f95809c6cd356f5d6b79147825dcb1d4d64255702c6befd3f314d2e03789` |

Archive notes: publisher lap times/positions include official Ergast/Jolpica corrections; durations have millisecond export precision. Per-lap archive weather and mini-sector additions are discarded; separate weather samples are joined backward at lap start. The status inventory is derived from lap codes and race-control red flags, not an original status timeline.

Results:

- Dry/no-red-flag checks passed.
- Raw lap TrackStatus counts: `{"1": 1087, "12": 40, "21": 2}`.

| Rule | Description | Removed | Remaining |
|---|---|---|---|
| raw | All input laps | 0 | 1129 |
| C1 | Missing lap timing | 0 | 1129 |
| C2 | Standing-start lap | 20 | 1109 |
| C3 | In-lap | 42 | 1067 |
| C4 | Out-lap | 43 | 1024 |
| C5 | Track status not exactly green | 18 | 1006 |
| C6 | Stewards-deleted lap | 18 | 988 |
| C7 | Unsynced lap timing | 0 | 988 |
| C8 | FastF1-generated lap | 0 | 988 |
| C9 | Non-dry or unknown compound | 0 | 988 |
| C10 | Slower than 107% of surviving race median | 0 | 988 |

Saved outputs:

- `reports/tables/audit_2024_01.csv`
- `reports/figures/fig01_laptime_vs_tyre_age_2024_01.png`
- `reports/figures/fig02_laptime_vs_lap_2024_01.png`
- `reports/figures/fig03_cleaning_audit_2024_01.png`

Observations:

- 87.5% of raw laps survived the ordered rules; each removed lap is counted once.
- Largest first-removal count: C4 (Out-lap), 43 laps.
- Tyre-age curves are descriptive: fuel burn, traffic, and other effects remain mixed with tyre ageing. No model or prediction metric was produced.

Next:

- Stop after Phase 1 and review the saved figures and audit before starting Phase 2.


### 2026-11-02 — Phase 1 — One-race exploration: 2024_01

Timestamp: 2026-11-02T18:52:56+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts\01_explore_one_race.py --year 2024 --round 1 --source archive`

Data: FastF1-derived TracingInsights archive (training split); Bahrain Grand Prix; 1129 raw laps, 988 clean laps.

Source: [TracingInsights/2024](https://github.com/TracingInsights/2024), commit `d1543a71370d9b617b627319f8ad80b2f85dfd5e`.

Raw inputs are verified by SHA-256 and cached under `data/raw/tracinginsights/2024_01/`.

| Source file | SHA-256 |
|---|---|
| [session_laptimes.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/session_laptimes.json) | `508f22e47936ac8de631e21524ed7b9808c1d7e0e52729d22dc5162bf8e04aae` |
| [weather.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/weather.json) | `50d14901d8b3982dd55190ba2467167c3593d63ed0e1167a3b379baba6483975` |
| [rcm.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/rcm.json) | `9c5b3cf273d2a2f25235f0e137e1505deb3fc5c50bd8424e98db2cc29ad64994` |
| [drivers.json](https://raw.githubusercontent.com/TracingInsights/2024/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race/drivers.json) | `eb28f95809c6cd356f5d6b79147825dcb1d4d64255702c6befd3f314d2e03789` |

Archive notes: publisher lap times/positions include official Ergast/Jolpica corrections; durations have millisecond export precision. Per-lap archive weather and mini-sector additions are discarded; separate weather samples are joined backward at lap start. The status inventory is derived from lap codes and race-control red flags, not an original status timeline.

Results:

- Dry/no-red-flag checks passed.
- Raw lap TrackStatus counts: `{"1": 1087, "12": 40, "21": 2}`.

| Rule | Description | Removed | Remaining |
|---|---|---|---|
| raw | All input laps | 0 | 1129 |
| C1 | Missing lap timing | 0 | 1129 |
| C2 | Standing-start lap | 20 | 1109 |
| C3 | In-lap | 42 | 1067 |
| C4 | Out-lap | 43 | 1024 |
| C5 | Track status not exactly green | 18 | 1006 |
| C6 | Stewards-deleted lap | 18 | 988 |
| C7 | Unsynced lap timing | 0 | 988 |
| C8 | FastF1-generated lap | 0 | 988 |
| C9 | Non-dry or unknown compound | 0 | 988 |
| C10 | Slower than 107% of surviving race median | 0 | 988 |

Saved outputs:

- `reports/tables/audit_2024_01.csv`
- `reports/figures/fig01_laptime_vs_tyre_age_2024_01.png`
- `reports/figures/fig02_laptime_vs_lap_2024_01.png`
- `reports/figures/fig03_cleaning_audit_2024_01.png`

Observations:

- 87.5% of raw laps survived the ordered rules; each removed lap is counted once.
- Largest first-removal count: C4 (Out-lap), 43 laps.
- Tyre-age curves are descriptive: fuel burn, traffic, and other effects remain mixed with tyre ageing. No model or prediction metric was produced.

Next:

- Stop after Phase 1 and review the saved figures and audit before starting Phase 2.

### 2026-11-02 — Phase 1 — free recovery final verification

Commands:

```powershell
.\.venv\Scripts\python.exe scripts\01_explore_one_race.py --year 2024 --round 1
.\.venv\Scripts\python.exe scripts\01_explore_one_race.py --year 2024 --round 1 --source archive
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Data: the pinned, checksum-verified Bahrain 2024 public archive used by the two
successful run entries above. No 2025 data was opened.

Results:

- Both source-selection commands completed successfully and produced the same
  audit: **1,129 input laps, 988 clean laps**. The per-rule counts and source
  fingerprints were automatically recorded by each run.
- Final test run: **67 passed**. Dependency check: **No broken requirements found.**
- An independent read-only review recomputed the audit and confirmed exact
  equality with the saved CSV, all four cached source hashes, dry/no-red
  eligibility, and no missing weather on clean laps.
- All three real PNGs were visually inspected by the implementer and reviewer.
  Source attribution, units, annotations and audit counts are readable.
- All seven dependency pins still match installed versions. Later-phase modules
  and scripts remain docstring-only placeholders.

Observations:

- The free archive resolves the Phase 1 access block; no subscription or version
  upgrade was used. Its documented export limitations still apply.
- Review found no Critical or Important issues. A minor retrieval-time wording
  issue was corrected: retrieval time is in `provenance.json`, while successful
  log entries contain the commit, URLs and hashes.

Phase 1 DoD:

- [x] Real script completes end to end.
- [x] Audit CSV and all three real figures saved to guide paths.
- [x] Cleaning and complete test suite pass.
- [x] Measured audit numbers, command and observations logged.

Next: stop after Phase 1. Phase 2 has not been started.


### 2026-11-02 — Phase 2 — race manifest

Timestamp: 2026-11-02T19:58:18+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/02_build_manifest.py --source archive`

Fixed eligibility: wet-compound share >5% or any recorded rainfall excludes the race. When both apply, `wet_compound` is the primary reason; both measurements remain in the manifest. Load/schema errors are isolated per race.

Included/excluded by year:

| year | split | included | scheduled | excluded |
|---|---|---|---|---|
| 2023 | train | 17 | 22 | 5 |
| 2024 | train | 19 | 24 | 5 |
| 2025 | test | 18 | 24 | 6 |

Counts by primary reason (each race counted once):

| year | split | decision | reason | races |
|---|---|---|---|---|
| 2023 | train | excluded | rainfall | 3 |
| 2023 | train | excluded | wet_compound | 2 |
| 2023 | train | included | included | 17 |
| 2024 | train | excluded | rainfall | 2 |
| 2024 | train | excluded | wet_compound | 3 |
| 2024 | train | included | included | 19 |
| 2025 | test | excluded | rainfall | 3 |
| 2025 | test | excluded | wet_compound | 3 |
| 2025 | test | included | included | 18 |

Source provenance:

| year | data_source | source_commit |
|---|---|---|
| 2023 | TracingInsights/2023 | 737c84d3ae14b9883a671ca248d015b69cf1cb58 |
| 2024 | TracingInsights/2024 | d1543a71370d9b617b627319f8ad80b2f85dfd5e |
| 2025 | TracingInsights/2025 | dfb3ea3f50b69857fe8e40cabe5e899743c59f6a |

Manual overrides:

No manual overrides were applied.

Races flagged for manual review (diagnostics only; no automatic override):

| race_id | year | event_name | include | exclude_reason | review_reasons | wet_compound_share | any_rainfall | n_laps_raw | n_laps_clean | error_detail | manual_override | manual_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023_03 | 2023 | Australian Grand Prix | True |  | red_flag_session | 0 | False | 1003 | 737 |  |  |  |
| 2023_08 | 2023 | Canadian Grand Prix | True |  | unknown_compound | 0 | False | 1317 | 1031 |  |  |  |
| 2023_09 | 2023 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1354 | 1087 |  |  |  |
| 2023_12 | 2023 | Belgian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 816 | 653 |  |  |  |
| 2023_13 | 2023 | Dutch Grand Prix | False | wet_compound | red_flag_session | 0.244229 | True | 1343 | 755 |  |  |  |
| 2023_19 | 2023 | Mexico City Grand Prix | True |  | red_flag_session | 0 | False | 1282 | 1048 |  |  |  |
| 2023_20 | 2023 | São Paulo Grand Prix | False | rainfall | rainfall_without_wet_threshold;red_flag_session | 0 | True | 1109 | 972 |  |  |  |
| 2024_04 | 2024 | Japanese Grand Prix | True |  | red_flag_session | 0 | False | 907 | 750 |  |  |  |
| 2024_08 | 2024 | Monaco Grand Prix | True |  | red_flag_session | 0 | False | 1237 | 1152 |  |  |  |
| 2024_10 | 2024 | Spanish Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1310 | 1192 |  |  |  |
| 2024_11 | 2024 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1405 | 1240 |  |  |  |
| 2024_21 | 2024 | São Paulo Grand Prix | False | wet_compound | no_clean_laps;red_flag_session | 1 | True | 1135 | 0 |  |  |  |
| 2025_06 | 2025 | Miami Grand Prix | False | rainfall | rainfall_without_wet_threshold;unknown_compound | 0 | True | 1005 | 478 |  |  |  |
| 2025_10 | 2025 | Canadian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1349 | 1153 |  |  |  |
| 2025_13 | 2025 | Belgian Grand Prix | False | wet_compound | unknown_compound;red_flag_session | 0.2719 | True | 879 | 548 |  |  |  |
| 2025_18 | 2025 | Singapore Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1229 | 1105 |  |  |  |

The user authorized 2025 ingestion for fixed Phase 2 preparation and quality counts only. No test prediction results, feature selection, tuning, or model evaluation were produced.

Saved:

- `data/processed/race_manifest.csv`
- `reports/tables/race_manifest.csv`
- `reports/tables/race_manifest_summary.csv`
- `reports/tables/race_manifest_review.csv`
- `reports/figures/fig04_race_manifest_summary.png`


### 2026-11-02 — Phase 2 — clean laps complete

Timestamp: 2026-11-02T19:58:32+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/03_build_dataset.py --source archive`

Fixed eligibility: wet-compound share >5% or any recorded rainfall excludes the race. When both apply, `wet_compound` is the primary reason; both measurements remain in the manifest. Load/schema errors are isolated per race.

Included/excluded by year:

| year | split | included | scheduled | excluded |
|---|---|---|---|---|
| 2023 | train | 17 | 22 | 5 |
| 2024 | train | 19 | 24 | 5 |
| 2025 | test | 18 | 24 | 6 |

Counts by primary reason (each race counted once):

| year | split | decision | reason | races |
|---|---|---|---|---|
| 2023 | train | excluded | rainfall | 3 |
| 2023 | train | excluded | wet_compound | 2 |
| 2023 | train | included | included | 17 |
| 2024 | train | excluded | rainfall | 2 |
| 2024 | train | excluded | wet_compound | 3 |
| 2024 | train | included | included | 19 |
| 2025 | test | excluded | rainfall | 3 |
| 2025 | test | excluded | wet_compound | 3 |
| 2025 | test | included | included | 18 |

Source provenance:

| year | data_source | source_commit |
|---|---|---|
| 2023 | TracingInsights/2023 | 737c84d3ae14b9883a671ca248d015b69cf1cb58 |
| 2024 | TracingInsights/2024 | d1543a71370d9b617b627319f8ad80b2f85dfd5e |
| 2025 | TracingInsights/2025 | dfb3ea3f50b69857fe8e40cabe5e899743c59f6a |

Manual overrides:

No manual overrides were applied.

Races flagged for manual review (diagnostics only; no automatic override):

| race_id | year | event_name | include | exclude_reason | review_reasons | wet_compound_share | any_rainfall | n_laps_raw | n_laps_clean | error_detail | manual_override | manual_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023_03 | 2023 | Australian Grand Prix | True |  | red_flag_session | 0 | False | 1003 | 737 |  |  |  |
| 2023_08 | 2023 | Canadian Grand Prix | True |  | unknown_compound | 0 | False | 1317 | 1031 |  |  |  |
| 2023_09 | 2023 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1354 | 1087 |  |  |  |
| 2023_12 | 2023 | Belgian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 816 | 653 |  |  |  |
| 2023_13 | 2023 | Dutch Grand Prix | False | wet_compound | red_flag_session | 0.244229 | True | 1343 | 755 |  |  |  |
| 2023_19 | 2023 | Mexico City Grand Prix | True |  | red_flag_session | 0 | False | 1282 | 1048 |  |  |  |
| 2023_20 | 2023 | São Paulo Grand Prix | False | rainfall | rainfall_without_wet_threshold;red_flag_session | 0 | True | 1109 | 972 |  |  |  |
| 2024_04 | 2024 | Japanese Grand Prix | True |  | red_flag_session | 0 | False | 907 | 750 |  |  |  |
| 2024_08 | 2024 | Monaco Grand Prix | True |  | red_flag_session | 0 | False | 1237 | 1152 |  |  |  |
| 2024_10 | 2024 | Spanish Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1310 | 1192 |  |  |  |
| 2024_11 | 2024 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1405 | 1240 |  |  |  |
| 2024_21 | 2024 | São Paulo Grand Prix | False | wet_compound | no_clean_laps;red_flag_session | 1 | True | 1135 | 0 |  |  |  |
| 2025_06 | 2025 | Miami Grand Prix | False | rainfall | rainfall_without_wet_threshold;unknown_compound | 0 | True | 1005 | 478 |  |  |  |
| 2025_10 | 2025 | Canadian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1349 | 1153 |  |  |  |
| 2025_13 | 2025 | Belgian Grand Prix | False | wet_compound | unknown_compound;red_flag_session | 0.2719 | True | 879 | 548 |  |  |  |
| 2025_18 | 2025 | Singapore Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1229 | 1105 |  |  |  |

The user authorized 2025 ingestion for fixed Phase 2 preparation and quality counts only. No test prediction results, feature selection, tuning, or model evaluation were produced.

Total clean laps per split:

| split | clean_laps |
|---|---|
| test | 17403 |
| train | 32897 |

Clean laps by year:

| year | split | races | clean_laps |
|---|---|---|---|
| 2023 | train | 17 | 15155 |
| 2024 | train | 19 | 17742 |
| 2025 | test | 18 | 17403 |

Dataset diagnostics (unmatched weather retained as missing):

None.

All 54 included races were assembled; 50300 clean laps; 594 audit rows; zero included-race load errors.

Saved:

- `data/processed/laps_clean.parquet`
- `reports/tables/audit_all_races.csv`
- `reports/tables/dataset_load_errors.csv`
- `reports/tables/dataset_review.csv`

Phase 2 DoD: manifest copies, fig04, complete parquet and combined audit saved; decision/reason counts and manual directives logged. Stop here: no Phase 3 features were built.


### 2026-11-02 — Phase 2 — race manifest

Timestamp: 2026-11-02T19:59:55+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/02_build_manifest.py --source archive`

Fixed eligibility: wet-compound share >5% or any recorded rainfall excludes the race. When both apply, `wet_compound` is the primary reason; both measurements remain in the manifest. Load/schema errors are isolated per race.

Included/excluded by year:

| year | split | included | scheduled | excluded |
|---|---|---|---|---|
| 2023 | train | 17 | 22 | 5 |
| 2024 | train | 19 | 24 | 5 |
| 2025 | test | 18 | 24 | 6 |

Counts by primary reason (each race counted once):

| year | split | decision | reason | races |
|---|---|---|---|---|
| 2023 | train | excluded | rainfall | 3 |
| 2023 | train | excluded | wet_compound | 2 |
| 2023 | train | included | included | 17 |
| 2024 | train | excluded | rainfall | 2 |
| 2024 | train | excluded | wet_compound | 3 |
| 2024 | train | included | included | 19 |
| 2025 | test | excluded | rainfall | 3 |
| 2025 | test | excluded | wet_compound | 3 |
| 2025 | test | included | included | 18 |

Source provenance:

| year | data_source | source_commit |
|---|---|---|
| 2023 | TracingInsights/2023 | 737c84d3ae14b9883a671ca248d015b69cf1cb58 |
| 2024 | TracingInsights/2024 | d1543a71370d9b617b627319f8ad80b2f85dfd5e |
| 2025 | TracingInsights/2025 | dfb3ea3f50b69857fe8e40cabe5e899743c59f6a |

Manual overrides:

No manual overrides were applied.

Races flagged for manual review (diagnostics only; no automatic override):

| race_id | year | event_name | include | exclude_reason | review_reasons | wet_compound_share | any_rainfall | n_laps_raw | n_laps_clean | error_detail | manual_override | manual_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023_03 | 2023 | Australian Grand Prix | True |  | red_flag_session | 0 | False | 1003 | 737 |  |  |  |
| 2023_08 | 2023 | Canadian Grand Prix | True |  | unknown_compound | 0 | False | 1317 | 1031 |  |  |  |
| 2023_09 | 2023 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1354 | 1087 |  |  |  |
| 2023_12 | 2023 | Belgian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 816 | 653 |  |  |  |
| 2023_13 | 2023 | Dutch Grand Prix | False | wet_compound | red_flag_session | 0.244229 | True | 1343 | 755 |  |  |  |
| 2023_19 | 2023 | Mexico City Grand Prix | True |  | red_flag_session | 0 | False | 1282 | 1048 |  |  |  |
| 2023_20 | 2023 | São Paulo Grand Prix | False | rainfall | rainfall_without_wet_threshold;red_flag_session | 0 | True | 1109 | 972 |  |  |  |
| 2024_04 | 2024 | Japanese Grand Prix | True |  | red_flag_session | 0 | False | 907 | 750 |  |  |  |
| 2024_08 | 2024 | Monaco Grand Prix | True |  | red_flag_session | 0 | False | 1237 | 1152 |  |  |  |
| 2024_10 | 2024 | Spanish Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1310 | 1192 |  |  |  |
| 2024_11 | 2024 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1405 | 1240 |  |  |  |
| 2024_21 | 2024 | São Paulo Grand Prix | False | wet_compound | no_clean_laps;red_flag_session | 1 | True | 1135 | 0 |  |  |  |
| 2025_06 | 2025 | Miami Grand Prix | False | rainfall | rainfall_without_wet_threshold;unknown_compound | 0 | True | 1005 | 478 |  |  |  |
| 2025_10 | 2025 | Canadian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1349 | 1153 |  |  |  |
| 2025_13 | 2025 | Belgian Grand Prix | False | wet_compound | unknown_compound;red_flag_session | 0.2719 | True | 879 | 548 |  |  |  |
| 2025_18 | 2025 | Singapore Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1229 | 1105 |  |  |  |

The user authorized 2025 ingestion for fixed Phase 2 preparation and quality counts only. No test prediction results, feature selection, tuning, or model evaluation were produced.

Saved:

- `data/processed/race_manifest.csv`
- `reports/tables/race_manifest.csv`
- `reports/tables/race_manifest_summary.csv`
- `reports/tables/race_manifest_review.csv`
- `reports/figures/fig04_race_manifest_summary.png`


### 2026-11-02 — Phase 2 — clean laps complete

Timestamp: 2026-11-02T20:02:28+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/03_build_dataset.py --source archive`

Fixed eligibility: wet-compound share >5% or any recorded rainfall excludes the race. When both apply, `wet_compound` is the primary reason; both measurements remain in the manifest. Load/schema errors are isolated per race.

Included/excluded by year:

| year | split | included | scheduled | excluded |
|---|---|---|---|---|
| 2023 | train | 17 | 22 | 5 |
| 2024 | train | 19 | 24 | 5 |
| 2025 | test | 18 | 24 | 6 |

Counts by primary reason (each race counted once):

| year | split | decision | reason | races |
|---|---|---|---|---|
| 2023 | train | excluded | rainfall | 3 |
| 2023 | train | excluded | wet_compound | 2 |
| 2023 | train | included | included | 17 |
| 2024 | train | excluded | rainfall | 2 |
| 2024 | train | excluded | wet_compound | 3 |
| 2024 | train | included | included | 19 |
| 2025 | test | excluded | rainfall | 3 |
| 2025 | test | excluded | wet_compound | 3 |
| 2025 | test | included | included | 18 |

Source provenance:

| year | data_source | source_commit |
|---|---|---|
| 2023 | TracingInsights/2023 | 737c84d3ae14b9883a671ca248d015b69cf1cb58 |
| 2024 | TracingInsights/2024 | d1543a71370d9b617b627319f8ad80b2f85dfd5e |
| 2025 | TracingInsights/2025 | dfb3ea3f50b69857fe8e40cabe5e899743c59f6a |

Manual overrides:

No manual overrides were applied.

Races flagged for manual review (diagnostics only; no automatic override):

| race_id | year | event_name | include | exclude_reason | review_reasons | wet_compound_share | any_rainfall | n_laps_raw | n_laps_clean | error_detail | manual_override | manual_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023_03 | 2023 | Australian Grand Prix | True |  | red_flag_session | 0 | False | 1003 | 737 |  |  |  |
| 2023_08 | 2023 | Canadian Grand Prix | True |  | unknown_compound | 0 | False | 1317 | 1031 |  |  |  |
| 2023_09 | 2023 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1354 | 1087 |  |  |  |
| 2023_12 | 2023 | Belgian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 816 | 653 |  |  |  |
| 2023_13 | 2023 | Dutch Grand Prix | False | wet_compound | red_flag_session | 0.244229 | True | 1343 | 755 |  |  |  |
| 2023_19 | 2023 | Mexico City Grand Prix | True |  | red_flag_session | 0 | False | 1282 | 1048 |  |  |  |
| 2023_20 | 2023 | São Paulo Grand Prix | False | rainfall | rainfall_without_wet_threshold;red_flag_session | 0 | True | 1109 | 972 |  |  |  |
| 2024_04 | 2024 | Japanese Grand Prix | True |  | red_flag_session | 0 | False | 907 | 750 |  |  |  |
| 2024_08 | 2024 | Monaco Grand Prix | True |  | red_flag_session | 0 | False | 1237 | 1152 |  |  |  |
| 2024_10 | 2024 | Spanish Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1310 | 1192 |  |  |  |
| 2024_11 | 2024 | Austrian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1405 | 1240 |  |  |  |
| 2024_21 | 2024 | São Paulo Grand Prix | False | wet_compound | no_clean_laps;red_flag_session | 1 | True | 1135 | 0 |  |  |  |
| 2025_06 | 2025 | Miami Grand Prix | False | rainfall | rainfall_without_wet_threshold;unknown_compound | 0 | True | 1005 | 478 |  |  |  |
| 2025_10 | 2025 | Canadian Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1349 | 1153 |  |  |  |
| 2025_13 | 2025 | Belgian Grand Prix | False | wet_compound | unknown_compound;red_flag_session | 0.2719 | True | 879 | 548 |  |  |  |
| 2025_18 | 2025 | Singapore Grand Prix | False | rainfall | rainfall_without_wet_threshold | 0 | True | 1229 | 1105 |  |  |  |

The user authorized 2025 ingestion for fixed Phase 2 preparation and quality counts only. No test prediction results, feature selection, tuning, or model evaluation were produced.

Total clean laps per split:

| split | clean_laps |
|---|---|
| test | 17403 |
| train | 32897 |

Clean laps by year:

| year | split | races | clean_laps |
|---|---|---|---|
| 2023 | train | 17 | 15155 |
| 2024 | train | 19 | 17742 |
| 2025 | test | 18 | 17403 |

Dataset diagnostics (unmatched weather retained as missing):

None.

All 54 included races were assembled; 50300 clean laps; 594 audit rows; zero included-race load errors.

Saved:

- `data/processed/laps_clean.parquet`
- `reports/tables/audit_all_races.csv`
- `reports/tables/dataset_load_errors.csv`
- `reports/tables/dataset_review.csv`

Phase 2 DoD: manifest copies, fig04, complete parquet and combined audit saved; decision/reason counts and manual directives logged. Stop here: no Phase 3 features were built.


### 2026-11-02 — Phase 2 — saved-output reconciliation

Timestamp: 2026-11-02T20:03:49+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/verify_phase2_outputs.py`

| scheduled_races | included_races | excluded_races | clean_laps | audit_rows | review_flags | manual_overrides | load_errors |
|---|---|---|---|---|---|---|---|
| 70 | 54 | 16 | 50300 | 594 | 16 | 0 | 0 |

Manifest copies are byte-identical; every included race occurs in parquet and audit; all raw/C10 counts and ordered additive audits reconcile. Driver-lap keys are unique; fixed season labels are correct; matched weather has no missing values. No features dataset or final-evaluation marker exists.

The complete pytest suite passed. An independent read-only review found no blocking issue. The real fig04 was visually inspected: legible labels, counts, primary reasons and source note.

DoD:

- [x] Both real scripts exit successfully with the verified free cache.
- [x] Both manifest copies, parquet, combined audit and fig04 exist.
- [x] Included/excluded and primary-reason counts logged per season.
- [x] Clean lap counts logged per split.
- [x] All flagged races and absence of manual overrides logged.
- [x] Tests pass; stop after Phase 2.


### 2026-11-02 — Phase 3 — features and eligibility counts

Timestamp: 2026-11-02T21:01:35+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/03_build_dataset.py --part features`

Input: `data/processed/laps_clean.parquet`; SHA-256 `eb8202b2a0a49988fe0321e195d8d6e54c19863e03b9daa128d5fc5866ff0cf9`.

Current lap cleanliness is guaranteed by the Phase 2 input: F1 records this precondition with zero additional removals. F2–F4 give ordered first-removal counts, so each ineligible clean lap is counted once. No cleaning threshold is recalculated during feature construction.

Eligibility before/after each rule, per split:

| split | rule_id | rule_description | rows_before | rows_dropped | rows_after |
|---|---|---|---|---|---|
| test | F1 | Current lap already clean (Phase 2 input) | 17403 | 0 | 17403 |
| train | F1 | Current lap already clean (Phase 2 input) | 32897 | 0 | 32897 |
| test | F2 | At least three prior clean laps in race/driver | 17403 | 1039 | 16364 |
| train | F2 | At least three prior clean laps in race/driver | 32897 | 2080 | 30817 |
| test | F3 | Prior three lap numbers exactly consecutive | 16364 | 2646 | 13718 |
| train | F3 | Prior three lap numbers exactly consecutive | 30817 | 5481 | 25336 |
| test | F4 | Prior three laps in the current stint | 13718 | 0 | 13718 |
| train | F4 | Prior three laps in the current stint | 25336 | 0 | 25336 |

Row counts per split:

| split | clean_laps | feature_rows | dropped_rows |
|---|---|---|---|
| test | 17403 | 13718 | 3685 |
| train | 32897 | 25336 | 7561 |

Row counts per year:

| year | clean_laps | split | feature_rows | dropped_rows |
|---|---|---|---|---|
| 2023 | 15155 | train | 11197 | 3958 |
| 2024 | 17742 | train | 14139 | 3603 |
| 2025 | 17403 | test | 13718 | 3685 |

Feature contract: lag1–lag3 are clean, consecutive and in the same stint; optional lag4/lag5 stay missing across a gap or stint change and are excluded from model inputs. Stint slope/count use only strictly earlier clean laps in that stint, including earlier laps before a gap. Current lap time appears only on the target side.

Fig05 uses configured training seasons only; median target delta by tyre age and compound is labelled associative, not causal. No test targets were plotted or analysed.

The user authorized the fixed Phase 3 transformation and row counts for both splits. No model was trained; no prediction, baseline score, tuning or evaluation metric was produced. The final-evaluation marker remains untouched.

Saved:

- `data/processed/features.parquet`
- `reports/tables/feature_eligibility_audit.csv`
- `reports/figures/fig05_degradation_by_compound.png`

Phase 3 outputs complete. Stop before Phase 4: no baselines or model training.


### 2026-11-02 — Phase 3 — saved-output verification

Timestamp: 2026-11-02T21:05:10+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/verify_phase3_outputs.py`

| split | feature_rows | races |
|---|---|---|
| test | 13718 | 18 |
| train | 25336 | 36 |

Saved outputs reconcile with the clean source: all lag identities, optional-history masks, target arithmetic, finite numeric inputs, sampled independent strictly-past OLS/counts, fixed season labels and ordered eligibility counts pass. No overlap of training/test race IDs. Fig05 was visually inspected and labelled training-only, associative and not causal. No final-evaluation marker or trained model was created.


### 2026-11-02 — Phase 3 — completion checks

Timestamp: 2026-11-02T21:07:20+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" -c`

| split | feature_rows |
|---|---|
| test | 13718 |
| train | 25336 |

DoD:

- [x] Features parquet, eligibility audit and training-only fig05 saved.
- [x] Requested hand-checked history, future-target mutation and GroupKFold split tests implemented.
- [x] Full pytest run: 121 passed; pip check: no broken requirements.
- [x] Saved-output reconciliation and independent review pass; no findings.
- [x] Row counts and ordered drop counts logged; no model or baseline trained.

Phase 3 complete; Phase 4 has not started.


### 2026-11-03 — Phase 4 — training baselines

Timestamp: 2026-11-03T00:03:55+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/04_run_baselines.py`

Training only: 25336 eligible laps across 36 races, years [2023, 2024]. Both parquet reads apply year/split predicates before returning rows. No 2025 rows were loaded for analysis or changed.

All baselines predict lap-time seconds on identical rows. Summary weights races equally; standard deviation is the sample standard deviation (ddof=1).

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s |
|---|---|---|---|---|
| pred_b1 | 36 | 0.353741 | 0.352114 | 0.0723122 |
| pred_b2 | 36 | 0.366352 | 0.353417 | 0.0869105 |
| pred_b3 | 36 | 0.388288 | 0.369237 | 0.100759 |
| pred_b4 | 36 | 0.461501 | 0.423259 | 0.129878 |

Best baseline (the bar to beat): **pred_b1**, mean race MAE **0.353741 s**. Selected by lowest mean race MAE; ties break by predictor name.

B3 fallback: **4161/25336 rows (16.423271%)**. Fallback uses B2 whenever either optional lag is missing.

B4 extrapolates the OLS line over all strictly earlier clean laps of the same race/driver/stint. Its intercept is reconstructed from past mean lap/time; early clean laps omitted by feature eligibility still contribute. These baseline-only statistics are not added to the feature parquet or model feature list.

Saved:

- `reports/tables/baselines_train_by_race.csv`
- `reports/tables/baselines_train_summary.csv`
- `reports/tables/baselines_train_b3_fallback.csv`
- `reports/figures/fig06_baseline_mae_train.png`

Phase 4 complete. No ML model, tuning, cross-validation or final test evaluation was run.


### 2026-11-03 — Phase 4 — verification and observation

Timestamp: 2026-11-03T00:05:32+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" -c`

All 132 tests passed. Saved tables reconcile to 36 training races and 25,336 identical eligible laps; B3 fallback is 4,161 rows. Fig06 was visually inspected. B4 mean race MAE exceeds the best baseline by 0.107760 seconds: the initial hypothesis that B4 would be strongest was not supported. No 2025 rows were analysed or changed.


### 2026-11-03 — Phase 5 — ridge search and OOF complete

Timestamp: 2026-11-03T00:15:14+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/05_train_models.py`

Training only: 25336 laps / 36 races. Best params: `{"model__alpha": 100}`.

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s |
|---|---|---|---|---|
| pred_ridge | 36 | 0.349495 | 0.347228 | 0.0795661 |
| pred_b1 | 36 | 0.353741 | 0.352114 | 0.0723122 |

Refitted full-training pipeline saved to `models/ridge.joblib`. Development CV uses parameters selected on the same folds (not nested CV).


### 2026-11-03 — Phase 5 — ridge search and OOF complete

Timestamp: 2026-11-03T00:16:26+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/05_train_models.py --estimator-jobs 1`

Training only: 25336 laps / 36 races. Best params: `{"model__alpha": 100}`.

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s |
|---|---|---|---|---|
| pred_ridge | 36 | 0.349495 | 0.347228 | 0.0795661 |
| pred_b1 | 36 | 0.353741 | 0.352114 | 0.0723122 |

Refitted full-training pipeline saved to `models/ridge.joblib`. Development CV uses parameters selected on the same folds (not nested CV).


### 2026-11-03 — Phase 5 — rf search and OOF complete

Timestamp: 2026-11-03T00:39:58+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/05_train_models.py --estimator-jobs 1`

Training only: 25336 laps / 36 races. Best params: `{"model__max_depth": 12, "model__min_samples_leaf": 20}`.

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s |
|---|---|---|---|---|
| pred_rf | 36 | 0.33503 | 0.325842 | 0.0725439 |
| pred_b1 | 36 | 0.353741 | 0.352114 | 0.0723122 |

Refitted full-training pipeline saved to `models/rf.joblib`. Development CV uses parameters selected on the same folds (not nested CV).


### 2026-11-03 — Phase 5 — xgb search and OOF complete

Timestamp: 2026-11-03T00:41:17+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/05_train_models.py --estimator-jobs 1`

Training only: 25336 laps / 36 races. Best params: `{"model__learning_rate": 0.03, "model__max_depth": 3, "model__n_estimators": 300}`.

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s |
|---|---|---|---|---|
| pred_xgb | 36 | 0.343791 | 0.322266 | 0.103342 |
| pred_b1 | 36 | 0.353741 | 0.352114 | 0.0723122 |

Refitted full-training pipeline saved to `models/xgb.joblib`. Development CV uses parameters selected on the same folds (not nested CV).


### 2026-11-03 — Phase 5 — training model selection complete

Timestamp: 2026-11-03T00:45:51+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/05_train_models.py --estimator-jobs 1`

Training only: 25336 laps / 36 races. Both parquet reads used training-year/split filters. No 2025 rows were loaded, predicted or evaluated.

Best parameters:

```json
{
  "rf": {
    "model__max_depth": 12,
    "model__min_samples_leaf": 20
  },
  "ridge": {
    "model__alpha": 100
  },
  "xgb": {
    "model__learning_rate": 0.03,
    "model__max_depth": 3,
    "model__n_estimators": 300
  }
}
```

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s | improvement_s | improvement_pct | races_won_vs_baseline |
|---|---|---|---|---|---|---|---|
| pred_rf | 36 | 0.33503 | 0.325842 | 0.0725439 | 0.0187106 | 5.28934 | 32 |
| pred_xgb | 36 | 0.343791 | 0.322266 | 0.103342 | 0.00994948 | 2.81265 | 30 |
| pred_ridge | 36 | 0.349495 | 0.347228 | 0.0795661 | 0.0042451 | 1.20006 | 27 |
| pred_b1 | 36 | 0.353741 | 0.352114 | 0.0723122 | 0 | 0 | 0 |

Chosen final ML model: **rf**, with mean race OOF MAE **0.335030 s**, the lowest among ML models (ties break by predictor name). Phase 4 baseline: **pred_b1**, **0.353741 s**. The chosen model beats the baseline in development CV.

Hyperparameters were selected by GridSearchCV negative MAE on five GroupKFold folds. OOF predictions use the selected parameters and same folds, so these are non-nested development estimates. Each saved pipeline is refitted on all training rows. Preprocessing is fitted inside each pipeline/fold; no global scaler or encoder. Outer jobs are serial, estimator jobs=1.

Permutation importance uses freshly fitted pipelines on each training fold and permutes only its held-out validation inputs, five repeats, seed 42. Delta MAE equals lap-time MAE with the original mean_last3 offset held fixed. Error bars are SD across fold/repeat values, not confidence intervals.

Saved:

- `reports/tables/cv_search_ridge.csv`
- `models/ridge.joblib`
- `reports/tables/cv_search_rf.csv`
- `models/rf.joblib`
- `reports/tables/cv_search_xgb.csv`
- `models/xgb.joblib`
- `reports/tables/cv_results_by_race.csv`
- `reports/tables/cv_summary.csv`
- `reports/figures/fig07_cv_model_vs_baseline.png`
- `reports/figures/fig08_feature_importance_cv.png`

Additional evidence: cv_fold_membership.csv, cv_importance_by_fold.csv, cv_feature_importance.csv, data/processed/cv_oof_train.parquet and models/selection.json. Phase 6 has not started; no test-opened marker created.


### 2026-11-03 — Phase 5 — saved-output verification

Timestamp: 2026-11-03T02:05:38+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/verify_phase5.py`

The completed artifacts reconcile: 25,336 training OOF rows, 36 whole-race groups, identical baseline/model rows, independently recomputed per-race MAEs and macro summaries, all 4/4/8 search candidates across five folds, three fitted seeded pipelines, and 425 validation-fold permutation records (17 features × 5 folds × 5 repeats). No 2025 data was loaded by this verification; no final-test marker exists.


### 2026-11-03 — Phase 5 — completion checks

Timestamp: 2026-11-03T02:06:54+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" -c`

DoD:

- [x] Full prescribed Ridge/RF/XGBoost grids and five race-held-out folds completed.
- [x] CV comparison tables and figures 07/08 saved and visually inspected.
- [x] Three full-training pipeline refits saved; Random Forest selected by lowest mean race OOF MAE.
- [x] Validation-fold permutation evidence saved for all 17 model inputs.
- [x] 139 tests passed; pip check clean; saved-output reconciliation passed.
- [x] Independent review found no production defects; the resource-override test observation was corrected.
- [x] No 2025 rows loaded, predicted or evaluated. No final-test marker created.

The full run used --estimator-jobs 1 after the documented Windows worker-pool failure. No grid was reduced. Phase 5 complete; Phase 6 remains unstarted.


### 2026-11-03 — Phase 6 — final test results

Timestamp: 2026-11-03T02:44:42+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/06_final_test.py`

**The test set is now opened.** All results below are final held-out 2025 results; no feature, model, hyperparameter or selection changes were made.

13,718 eligible laps across 18 races; identical rows for all seven predictors. Saved delta predictions were converted with the original mean_last3. B4 uses all strictly earlier clean laps in each stint.

Training-selected final model remains **rf**; the comparison baseline remains **pred_b1**, selected in Phase 4. No test-based reselection. The chosen model **beats** the baseline on mean per-race MAE.

Full summary (equal weight per race; sample SD, ddof=1; strict wins and explicit ties):

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s | comparison_baseline | improvement_s | improvement_pct | races_won_vs_baseline | races_lost_vs_baseline | races_tied_vs_baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| pred_rf | 18 | 0.309541 | 0.295355 | 0.0996261 | pred_b1 | 0.0128832 | 3.99574 | 13 | 5 | 0 |
| pred_xgb | 18 | 0.318468 | 0.299242 | 0.102769 | pred_b1 | 0.0039561 | 1.22699 | 11 | 7 | 0 |
| pred_ridge | 18 | 0.320331 | 0.302939 | 0.0995081 | pred_b1 | 0.00209293 | 0.649125 | 13 | 5 | 0 |
| pred_b1 | 18 | 0.322424 | 0.311816 | 0.0948605 | pred_b1 | 0 | 0 | 0 | 0 | 18 |
| pred_b2 | 18 | 0.334719 | 0.317821 | 0.109425 | pred_b1 | -0.0122954 | -3.81342 | 6 | 12 | 0 |
| pred_b3 | 18 | 0.355633 | 0.337784 | 0.120412 | pred_b1 | -0.0332092 | -10.2998 | 2 | 16 | 0 |
| pred_b4 | 18 | 0.433867 | 0.387391 | 0.151603 | pred_b1 | -0.111443 | -34.5642 | 0 | 18 | 0 |

Full per-race MAE in seconds:

| race_id | n_laps | pred_b1 | pred_b2 | pred_b3 | pred_b4 | pred_ridge | pred_rf | pred_xgb |
|---|---|---|---|---|---|---|---|---|
| 2025_02 | 758 | 0.296416 | 0.322011 | 0.344733 | 0.379596 | 0.353269 | 0.301789 | 0.311558 |
| 2025_03 | 853 | 0.237319 | 0.246463 | 0.26793 | 0.316339 | 0.265613 | 0.250169 | 0.249469 |
| 2025_04 | 690 | 0.342733 | 0.336376 | 0.353121 | 0.418273 | 0.303648 | 0.296313 | 0.291 |
| 2025_05 | 663 | 0.312554 | 0.328907 | 0.357138 | 0.481784 | 0.302229 | 0.294398 | 0.293502 |
| 2025_07 | 733 | 0.331521 | 0.358547 | 0.378094 | 0.474849 | 0.325349 | 0.316365 | 0.314597 |
| 2025_08 | 879 | 0.657899 | 0.737505 | 0.806628 | 0.974334 | 0.673136 | 0.681244 | 0.68842 |
| 2025_09 | 740 | 0.370597 | 0.395688 | 0.403424 | 0.504755 | 0.366877 | 0.361442 | 0.356702 |
| 2025_11 | 791 | 0.347375 | 0.342999 | 0.348977 | 0.41143 | 0.388899 | 0.322674 | 0.360265 |
| 2025_14 | 1095 | 0.358282 | 0.353142 | 0.391406 | 0.466712 | 0.336481 | 0.319897 | 0.320523 |
| 2025_15 | 672 | 0.241862 | 0.253956 | 0.266601 | 0.358758 | 0.231099 | 0.227079 | 0.234816 |
| 2025_16 | 679 | 0.211462 | 0.221585 | 0.245373 | 0.280653 | 0.214084 | 0.222319 | 0.223174 |
| 2025_17 | 605 | 0.335534 | 0.328801 | 0.332208 | 0.537059 | 0.311818 | 0.309878 | 0.308346 |
| 2025_19 | 657 | 0.311078 | 0.310697 | 0.318025 | 0.394653 | 0.299768 | 0.290151 | 0.304981 |
| 2025_20 | 899 | 0.323998 | 0.306435 | 0.310455 | 0.376478 | 0.304993 | 0.288728 | 0.388609 |
| 2025_21 | 879 | 0.291382 | 0.296776 | 0.315548 | 0.374119 | 0.287683 | 0.278485 | 0.288145 |
| 2025_22 | 616 | 0.287578 | 0.313631 | 0.34336 | 0.37372 | 0.283771 | 0.300083 | 0.292496 |
| 2025_23 | 690 | 0.245403 | 0.26106 | 0.286847 | 0.305972 | 0.226833 | 0.228744 | 0.222972 |
| 2025_24 | 819 | 0.300639 | 0.310369 | 0.331529 | 0.380128 | 0.290407 | 0.281976 | 0.282846 |

Negative results: predictors not beating pred_b1 on mean race MAE: pred_b2, pred_b3, pred_b4. Chosen-model race losses: 2025_02, 2025_03, 2025_08, 2025_16, 2025_22.

B3 fallback: 2,051/13,718 (14.951159%).

Saved: reports/tables/test_results_by_race.csv, reports/tables/test_summary.csv, reports/figures/fig09_test_model_vs_baseline_by_race.png, and data/processed/test_predictions.parquet. models/TEST_OPENED.txt records opening time and frozen model hashes. No training, refitting or new data download. Phase 7 has not started.


### 2026-11-03 — Phase 6 — saved-output verification and completion

Timestamp: 2026-11-03T02:45:40+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/verify_phase6.py`

Saved predictions only: independently recomputed all seven per-race MAEs, mean/median/sample SD, improvements, and race win/loss/tie counts. All agree with the two saved CSVs; 13,718 rows across 18 races. No models were loaded or predicted again.

SHA-256 comparison against the pre-run snapshot passed for all 9 frozen files: features.py, models.py, config.py, three fitted pipelines, selection.json, and both input parquets. The marker contains exactly one opening and one completion.

The full suite passed 143 tests before opening, including synthetic rerun refusal, explicit-rerun deviation, marker retention after failure, saved offsets, identical rows and negative results. Independent code review found no blocking issues. Fig09 was visually inspected: all 18 race pairs, legend, units and source note are readable.

Phase 6 DoD complete. The test set is now opened. No Phase 7 analysis or model changes were made.


### 2026-11-03 — Phase 7 — saved-prediction error analysis

Timestamp: 2026-11-03T15:33:30+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" scripts/07_error_analysis.py`

Analysis of saved Phase 6 predictions only. The test set remains opened; no model was loaded, fitted or predicted again. Frozen training selection and original per-race scores were verified.

Fixed grouping: configured tyre-age edges [0, 5, 10, 20, 30, 100]; early <10 versus late >=10 prior clean stint laps; low <=0 versus high >0 signed stint_slope (s/lap). Only observed groups appear; empty groups are not assigned zero error. Primary comparison and figures use equal-race mean MAE within each group. Positive improvement_s means the model wins.

1. The training-selected pred_rf has mean race MAE 0.309541 s versus 0.322424 s for pred_b1, winning 13 and losing 5 of 18 races. The comparison retains all 13,718 saved eligible laps; no model was selected using these results.

2. Tyre-age results: 1-5: model 0.3439 s vs baseline 0.3353 s (gain -0.0086 s; 446 laps, 18 races); 6-10: model 0.3039 s vs baseline 0.3182 s (gain +0.0142 s; 2,818 laps, 18 races); 11-20: model 0.3037 s vs baseline 0.3155 s (gain +0.0118 s; 5,861 laps, 18 races); 21-30: model 0.3158 s vs baseline 0.3323 s (gain +0.0165 s; 3,306 laps, 18 races); 31+: model 0.3571 s vs baseline 0.3733 s (gain +0.0162 s; 1,287 laps, 16 races). These describe performance by age, not the incremental effect of the tyre-age feature.

3. Compound results: HARD: model 0.3383 s vs baseline 0.3753 s (gain +0.0371 s; 6,175 laps, 18 races); MEDIUM: model 0.2966 s vs baseline 0.3088 s (gain +0.0121 s; 5,807 laps, 18 races); SOFT: model 0.3833 s vs baseline 0.4083 s (gain +0.0250 s; 1,736 laps, 13 races). Compound groups can contain different races and drivers, so their error differences do not establish tyre performance differences.

4. Early/late clean-history proxy: Early (<10 prior clean laps): model 0.3082 s vs baseline 0.3148 s (gain +0.0066 s; 4,931 laps, 18 races); Late (>=10 prior clean laps): model 0.3107 s vs baseline 0.3270 s (gain +0.0163 s; 8,787 laps, 18 races). Early means fewer than 10 strictly earlier clean laps in the stint; late means at least 10. Removed laps are not counted, so this is not physical stint age or proximity to a pit stop.

5. Signed prior stint trend: Low (<=0 s/lap): model 0.2850 s vs baseline 0.2893 s (gain +0.0043 s; 8,186 laps, 18 races); High (>0 s/lap): model 0.3658 s vs baseline 0.3960 s (gain +0.0303 s; 5,532 laps, 18 races). Low means <=0 s/lap and high means >0 s/lap; high describes rising historical lap times, not greater slope magnitude or proven degradation.

6. Across circuits, the largest model gain is Sakhir: +0.046420 s (690 laps, 1 races); the smallest gain is Monaco: -0.023345 s (879 laps, 1 races). Negative gains are losses; all circuits remain in error_by_circuit.csv.

7. These are descriptive, not causal results on the already opened test set. Each reported mean gives equal weight to races represented in that group; tables also give pooled lap MAE and support counts. Groups may differ in race composition and contain few laps. No confidence intervals or independent tyre-age contribution are established; fuel, traffic, pace management and track evolution remain mixed in. No feature or hyperparameter changes follow from this analysis.

Full tyre_age_bucket comparison (`reports/tables/error_by_tyre_age_bucket.csv`):

| tyre_age_bucket | n_laps | n_races | model_pooled_mae_s | model_mean_race_mae_s | baseline_pooled_mae_s | baseline_mean_race_mae_s | model | baseline | improvement_s | improvement_pct | races_won | races_lost | races_tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-5 | 446 | 18 | 0.328238 | 0.3439 | 0.324652 | 0.335254 | pred_rf | pred_b1 | -0.00864543 | -2.57877 | 8 | 10 | 0 |
| 6-10 | 2818 | 18 | 0.29688 | 0.303911 | 0.309543 | 0.318156 | pred_rf | pred_b1 | 0.0142451 | 4.4774 | 13 | 5 | 0 |
| 11-20 | 5861 | 18 | 0.298915 | 0.303661 | 0.311905 | 0.315453 | pred_rf | pred_b1 | 0.0117918 | 3.73807 | 13 | 5 | 0 |
| 21-30 | 3306 | 18 | 0.320463 | 0.315819 | 0.336164 | 0.332343 | pred_rf | pred_b1 | 0.0165245 | 4.97213 | 13 | 5 | 0 |
| 31+ | 1287 | 16 | 0.396839 | 0.357069 | 0.412218 | 0.373269 | pred_rf | pred_b1 | 0.0162 | 4.34004 | 10 | 6 | 0 |

Full compound comparison (`reports/tables/error_by_compound.csv`):

| compound | n_laps | n_races | model_pooled_mae_s | model_mean_race_mae_s | baseline_pooled_mae_s | baseline_mean_race_mae_s | model | baseline | improvement_s | improvement_pct | races_won | races_lost | races_tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HARD | 6175 | 18 | 0.330342 | 0.338266 | 0.340935 | 0.375323 | pred_rf | pred_b1 | 0.0370568 | 9.87331 | 13 | 5 | 0 |
| MEDIUM | 5807 | 18 | 0.294817 | 0.296625 | 0.308322 | 0.30875 | pred_rf | pred_b1 | 0.0121254 | 3.92724 | 14 | 4 | 0 |
| SOFT | 1736 | 13 | 0.318696 | 0.383297 | 0.34064 | 0.408285 | pred_rf | pred_b1 | 0.0249877 | 6.12018 | 8 | 5 | 0 |

Full circuit comparison (`reports/tables/error_by_circuit.csv`):

| circuit | n_laps | n_races | model_pooled_mae_s | model_mean_race_mae_s | baseline_pooled_mae_s | baseline_mean_race_mae_s | model | baseline | improvement_s | improvement_pct | races_won | races_lost | races_tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Austin | 657 | 1 | 0.290151 | 0.290151 | 0.311078 | 0.311078 | pred_rf | pred_b1 | 0.0209268 | 6.72719 | 1 | 0 | 0 |
| Baku | 605 | 1 | 0.309878 | 0.309878 | 0.335534 | 0.335534 | pred_rf | pred_b1 | 0.0256559 | 7.64629 | 1 | 0 | 0 |
| Barcelona | 740 | 1 | 0.361442 | 0.361442 | 0.370597 | 0.370597 | pred_rf | pred_b1 | 0.00915481 | 2.47028 | 1 | 0 | 0 |
| Budapest | 1095 | 1 | 0.319897 | 0.319897 | 0.358282 | 0.358282 | pred_rf | pred_b1 | 0.0383856 | 10.7138 | 1 | 0 | 0 |
| Imola | 733 | 1 | 0.316365 | 0.316365 | 0.331521 | 0.331521 | pred_rf | pred_b1 | 0.0151559 | 4.57162 | 1 | 0 | 0 |
| Jeddah | 663 | 1 | 0.294398 | 0.294398 | 0.312554 | 0.312554 | pred_rf | pred_b1 | 0.0181559 | 5.80888 | 1 | 0 | 0 |
| Las Vegas | 616 | 1 | 0.300083 | 0.300083 | 0.287578 | 0.287578 | pred_rf | pred_b1 | -0.0125053 | -4.34849 | 0 | 1 | 0 |
| Lusail | 690 | 1 | 0.228744 | 0.228744 | 0.245403 | 0.245403 | pred_rf | pred_b1 | 0.0166592 | 6.78852 | 1 | 0 | 0 |
| Mexico City | 899 | 1 | 0.288728 | 0.288728 | 0.323998 | 0.323998 | pred_rf | pred_b1 | 0.0352695 | 10.8857 | 1 | 0 | 0 |
| Monaco | 879 | 1 | 0.681244 | 0.681244 | 0.657899 | 0.657899 | pred_rf | pred_b1 | -0.0233448 | -3.54838 | 0 | 1 | 0 |
| Monza | 679 | 1 | 0.222319 | 0.222319 | 0.211462 | 0.211462 | pred_rf | pred_b1 | -0.0108565 | -5.13399 | 0 | 1 | 0 |
| Sakhir | 690 | 1 | 0.296313 | 0.296313 | 0.342733 | 0.342733 | pred_rf | pred_b1 | 0.0464201 | 13.5441 | 1 | 0 | 0 |
| Shanghai | 758 | 1 | 0.301789 | 0.301789 | 0.296416 | 0.296416 | pred_rf | pred_b1 | -0.00537341 | -1.8128 | 0 | 1 | 0 |
| Spielberg | 791 | 1 | 0.322674 | 0.322674 | 0.347375 | 0.347375 | pred_rf | pred_b1 | 0.0247019 | 7.111 | 1 | 0 | 0 |
| Suzuka | 853 | 1 | 0.250169 | 0.250169 | 0.237319 | 0.237319 | pred_rf | pred_b1 | -0.0128506 | -5.41489 | 0 | 1 | 0 |
| São Paulo | 879 | 1 | 0.278485 | 0.278485 | 0.291382 | 0.291382 | pred_rf | pred_b1 | 0.0128977 | 4.42638 | 1 | 0 | 0 |
| Yas Island | 819 | 1 | 0.281976 | 0.281976 | 0.300639 | 0.300639 | pred_rf | pred_b1 | 0.0186622 | 6.20753 | 1 | 0 | 0 |
| Zandvoort | 672 | 1 | 0.227079 | 0.227079 | 0.241862 | 0.241862 | pred_rf | pred_b1 | 0.014783 | 6.11216 | 1 | 0 | 0 |

Full stint_phase comparison (`reports/tables/error_by_stint_phase.csv`):

| stint_phase | n_laps | n_races | model_pooled_mae_s | model_mean_race_mae_s | baseline_pooled_mae_s | baseline_mean_race_mae_s | model | baseline | improvement_s | improvement_pct | races_won | races_lost | races_tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Early (<10 prior clean laps) | 4931 | 18 | 0.306341 | 0.30825 | 0.314074 | 0.314822 | pred_rf | pred_b1 | 0.00657232 | 2.08763 | 13 | 5 | 0 |
| Late (>=10 prior clean laps) | 8787 | 18 | 0.318033 | 0.310698 | 0.334397 | 0.327022 | pred_rf | pred_b1 | 0.0163239 | 4.99168 | 13 | 5 | 0 |

Full slope_group comparison (`reports/tables/error_by_slope_group.csv`):

| slope_group | n_laps | n_races | model_pooled_mae_s | model_mean_race_mae_s | baseline_pooled_mae_s | baseline_mean_race_mae_s | model | baseline | improvement_s | improvement_pct | races_won | races_lost | races_tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Low (<=0 s/lap) | 8186 | 18 | 0.276552 | 0.285006 | 0.278289 | 0.289281 | pred_rf | pred_b1 | 0.00427484 | 1.47775 | 12 | 6 | 0 |
| High (>0 s/lap) | 5532 | 18 | 0.368993 | 0.365755 | 0.399308 | 0.396006 | pred_rf | pred_b1 | 0.030251 | 7.63904 | 18 | 0 | 0 |


### 2026-11-03 — Phase 7 — verification and completion

Timestamp: 2026-11-03T15:34:51+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/verify_phase7.py`

All 151 tests passed. Independent review found no blocking issues. Recomputed every group in all five CSVs directly from saved predictions: lap/race counts, pooled MAE, equal-race mean MAE, gains, percentages and race wins/losses/ties reconcile. Both figures were inspected for readable labels, counts, units and provenance.

All 13 frozen file hashes are unchanged: feature/model/config code, fitted models, selection, opening marker, input parquets, saved test predictions and Phase 6 result tables. No predictions were rerun.

Minor review follow-up deferred: dedicated tests for negative gains and zero baseline MAE; implemented handling was correct by inspection, and real negative-gain outputs reconcile. Existing tests cover ties and positive gains.

Phase 7 DoD: five tables, figures 10/11, seven measured findings and passing tests complete. Phase 8 remains unstarted.


### 2026-11-03 — Phase 8 — README metrics reconciliation

Timestamp: 2026-11-03T15:42:42+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/build_phase8_readme.py`

Recomputed all seven test predictors directly from saved predictions; per-race mean, median and sample SD match test_summary.csv to absolute tolerance 1e-12. No models were loaded or predictions rerun. The eleven-section README is generated from these measured tables and links the full negative results.

| predictor | n_races | mean_mae_s | median_mae_s | std_mae_s | comparison_baseline | improvement_s | improvement_pct | races_won_vs_baseline | races_lost_vs_baseline | races_tied_vs_baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| pred_rf | 18 | 0.309541 | 0.295355 | 0.0996261 | pred_b1 | 0.0128832 | 3.99574 | 13 | 5 | 0 |
| pred_xgb | 18 | 0.318468 | 0.299242 | 0.102769 | pred_b1 | 0.0039561 | 1.22699 | 11 | 7 | 0 |
| pred_ridge | 18 | 0.320331 | 0.302939 | 0.0995081 | pred_b1 | 0.00209293 | 0.649125 | 13 | 5 | 0 |
| pred_b1 | 18 | 0.322424 | 0.311816 | 0.0948605 | pred_b1 | 0 | 0 | 0 | 0 | 18 |
| pred_b2 | 18 | 0.334719 | 0.317821 | 0.109425 | pred_b1 | -0.0122954 | -3.81342 | 6 | 12 | 0 |
| pred_b3 | 18 | 0.355633 | 0.337784 | 0.120412 | pred_b1 | -0.0332092 | -10.2998 | 2 | 16 | 0 |
| pred_b4 | 18 | 0.433867 | 0.387391 | 0.151603 | pred_b1 | -0.111443 | -34.5642 | 0 | 18 | 0 |

Data/feature counts, error strata and CV table were read from existing recorded outputs. Fresh online reproduction remains unverified; no checkbox is inferred from documentation alone.


### 2026-11-03 — Phase 8 — portfolio documentation and reproducibility verification

Timestamp: 2026-11-03T16:11:10+07:00

Command: `& "D:\TUGAS\C Path\Portfolio\ML\F1 tyre degradation\.venv\Scripts\python.exe" .venv/verify_phase8.py`

Completed the eleven-section README with 6 embedded relative figures, all local links verified, full seven-predictor test results, negative findings, data-source and retrospective-eligibility caveats, commands and project structure. Recomputed metrics from saved predictions; no new design or selection decision.

Validation: the original and clean offline-installed environments each passed 153 tests; pip check passed. The clean environment installed the editable package and all runtime/dev dependencies from the existing wheel cache using normal build isolation. A wheel built successfully and contains archive_catalog.json.

Isolated end-to-end cached reproduction: all scripts 01–07 exited zero. Compared 23 CSVs, four processed Parquets, all three CV search parameter/score/rank tables (excluding elapsed-time columns), and selection.json. All agree with original results (numeric rtol=1e-10, atol=1e-12). Eleven figures were regenerated.

| command | exit_code | elapsed_s |
|---|---|---|
| python scripts/01_explore_one_race.py --year 2024 --round 1 --source archive | 0 | 4.74 |
| python scripts/02_build_manifest.py --source archive | 0 | 10.44 |
| python scripts/03_build_dataset.py --source archive --part all | 0 | 5.93 |
| python scripts/04_run_baselines.py | 0 | 5.56 |
| python scripts/05_train_models.py --estimator-jobs 1 | 0 | 1882.39 |
| python scripts/06_final_test.py --i-understand-rerun | 0 | 6.84 |
| python scripts/07_error_analysis.py | 0 | 5.64 |

Reproduction source: `.venv/phase8_reproduction`; raw inputs only were copied from the verified cache, with no processed data or models seeded. The original opened marker was copied immediately before script 06, which explicitly used --i-understand-rerun and logged its deviation. This verifies reproducibility, not another independent holdout. Original 13 frozen file hashes remain unchanged.

Clean-install checks use `.venv/phase8_clean_install`, an editable install of the isolated source copy and the seven pinned runtime versions. The already documented temporary-directory compatibility hook was required in this Windows sandbox; it is local/ignored, not part of application code. An exploratory --no-build-isolation wheel command failed because the old ambient setuptools 65.5.0 is incompatible with Python 3.13; normal isolated builds successfully used the declared setuptools>=68 requirement from cache.

The live pinned GitHub input probe failed with WinError 10061. This supplied workspace has no Git metadata or confirmed repository URL. Therefore a literal online fresh clone and cold data/dependency download remain unverified. Five of six PRD success checks are supported; the fresh-clone checkbox remains open. Cached reproduction and a fresh offline dependency install are not presented as proof of a network-backed clone.

The two Phase 7 minor edge-case coverage gaps are now closed: negative gain and a zero baseline MAE (undefined percent, retained absolute loss). Independent review found no material issues; its request to distinguish offline installation from online cloning was incorporated.

Additional verification commands actually run, separately in PowerShell:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pip wheel --no-deps --no-index --find-links data/setup/wheelhouse --wheel-dir .venv/phase8_wheels .venv/phase8_reproduction
.venv\phase8_clean_install\Scripts\python.exe -m pip install --no-index --find-links data/setup/wheelhouse -e ".venv/phase8_reproduction[dev]"
.venv\phase8_clean_install\Scripts\python.exe -m pip check
```

The clean-environment pytest command used its Python executable from the isolated source directory. Full run execution evidence is retained locally in `.venv/phase8_reproduction/execution.json` and `run_00.log` through `run_07.log`.

Portfolio bullet:

> Built a leakage-safe next-lap forecasting pipeline across 54 dry F1 races (2023–2025); a training-selected Random Forest reduced mean per-race MAE by 4.00% versus the last-lap baseline on 18 held-out 2025 races, with error analysis retaining failures on young tyres and individual circuits.

---

## Key results summary

Fill in **only** from the entries above.

| Stage | Split | Best baseline (MAE s) | Best model (MAE s) | Improvement | Races won vs baseline |
|---|---|---|---|---|---|
| Baselines (Phase 4) | train | | — | — | — |
| CV (Phase 5) | train CV | | | | |
| Final test (Phase 6) | 2025 test | | | | |

---

## Race manifest decisions

| race_id | Decision | Reason | Date |
|---|---|---|---|
| | | | |

---

## Deviations from IMPLEMENTATION_GUIDE.md

| Date | Section | What changed | Why | Impact |
|---|---|---|---|---|
| | | | | |

### 2026-11-02 — Phase 0 setup deviations

| Date | Section | What changed | Why | Impact |
|---|---|---|---|---|
| 2026-11-02 | §1 environment setup | Used Python 3.13.2 in `.venv`, with compatible cached wheels staged under `data/setup/wheelhouse/`. | Python 3.11 was initially used, but the cache lacked its scikit-learn wheel. `PIP_NO_INDEX=1` disables the index; a bounded retry with `PIP_NO_INDEX=0` failed because the configured proxy refused the connection (WinError 10061). | No extra project dependencies; the interpreter still satisfies Python 3.11+. Installation used the offline source via process-local `PIP_FIND_LINKS`. |
| 2026-11-02 | §1 environment setup | Added an ignored local startup hook at `.venv/Lib/site-packages/000_sandbox_compat.pth` loading `.venv/sandbox-compat/sitecustomize.py`. | Python 3.13 applies private Windows ACLs to directories created with mode `0o700`; those ACLs blocked this sandbox's pip build and pytest temporary directories (WinError 5). The venv upgrade's ensurepip step also failed, but the retained pip 24.0 was usable. | The hook redirects TEMP/TMP into `.venv/tmp` and preserves inherited ACLs for `0o700` mkdir calls only inside this workspace. It does not change package or test code, dependencies, or paths outside the workspace. It is local to this ignored venv; normal fresh setup outside the sandbox needs no hook. |
| 2026-11-02 | §1 workspace cleanup | The initial failed Python 3.13 pytest run left `pytest-cache-files-hy7iy6u6/` and `pytest-cache-files-lbcc_dxe/` in the project root. | Their private Windows ACLs also denied nonrecursive deletion and an ACL reset from this sandbox. | Both generated cache directories remain. They contain no project deliverables and can be removed outside this restricted sandbox; `.gitignore` is kept exactly as specified. Subsequent test runs use the working `.pytest_cache/`. |

### 2026-11-02 — Phase 1 — archive access blocks the real-data DoD

[FastF1 3.9 release notes](https://github.com/theOehrly/Fast-F1/releases/tag/v3.9.0)
confirm that October 2026 archive authentication requires an F1 TV Access account
or higher. Installed 3.8.3 has no authentication support. Fresh-download
reproducibility under PRD §§4 and 11 therefore needs a data-access decision.
No paid service, alternate data source, or dependency upgrade has been introduced.
Existing cached responses and failed attempts are retained. Real-data DoD is
unchecked; synthetic previews are not substituted for race outputs.

### 2026-11-02 — Phase 1 — free public archive adopted

The user explicitly requested: keep the project free and resolve the missing real
audit/figures. `archive.py` now reads the public TracingInsights Bahrain 2024
export at immutable commit `d1543a71370d9b617b627319f8ad80b2f85dfd5e`.

- **Access deviation (§3/Phase 1):** use four public FastF1-derived JSON files
  instead of a new authenticated `Session.load()` download. All required source
  fields are present and explicitly mapped; no quality flags are fabricated.
  The successful run entry records URLs and SHA-256 hashes. The first files were
  fetched byte-for-byte through the available HTTP tool because this execution
  process's proxy prevents direct requests; the package also includes a standard
  library downloader for normal terminal use and reuses verified local files.
- **Source limits:** publisher race lap times/positions include official
  Ergast/Jolpica corrections; duration fields are exported to milliseconds. The
  audit describes these exported inputs, so its C1 count is not evidence about
  missing timing in a pristine native FastF1 download. The reader version is
  FastF1 3.8.3; the publisher's extraction runtime is not asserted.
- **Eligibility evidence:** the original track-status timeline is not supplied.
  A labelled inventory of observed lap codes, cross-checked against separate
  race-control red flags, verifies Phase 1 eligibility. The original per-lap
  TrackStatus is preserved for C5; the inventory is never substituted into laps.
- **Leakage protection:** publisher-attached weather is discarded. The existing
  backward merge uses the separate session weather data. Unused mini-sector
  extras are also discarded; no OpenF1 endpoint or telemetry file is loaded.
- **Unchanged constraints:** no new dependency, paid access, cleaning rule,
  random seed, target, training/test season, or later-phase implementation.

Source: [TracingInsights Bahrain archive](https://github.com/TracingInsights/2024/tree/d1543a71370d9b617b627319f8ad80b2f85dfd5e/Bahrain%20Grand%20Prix/Race).
Publisher extraction code and repository licence are retained with the local raw
source files for provenance; they are not executed by this project.

---

### 2026-11-02 — Phase 2 — free catalog and fixed test-season preparation

- **Access extension (§3/Phase 2):** retain the free public FastF1-derived
  TracingInsights source adopted for Phase 1, now for all 70 scheduled Grand Prix
  races. `archive_catalog.json` pins 280 input files, their URLs and SHA-256 hashes,
  and FastF1's published schedule metadata. Source commits: 2023
  `737c84d3ae14b9883a671ca248d015b69cf1cb58`; 2024
  `d1543a71370d9b617b627319f8ad80b2f85dfd5e`; 2025
  `dfb3ea3f50b69857fe8e40cabe5e899743c59f6a`. Schedule commit:
  `2b0c23cf46d59bbe1c1b8c77da32fe8f74d6fe52` in
  [theOehrly/f1schedule](https://github.com/theOehrly/f1schedule/tree/2b0c23cf46d59bbe1c1b8c77da32fe8f74d6fe52).
  The existing source correction/precision/status-inventory limitations apply.
  No dependency, paid service, telemetry or OpenF1 endpoint was added. This
  execution environment cached public bytes through the available HTTP runtime
  because its shell proxy refuses requests; normal terminal runs can download
  missing files through the package's standard-library loader.
- **Narrow test-lock exception:** the user's Phase 2 task explicitly requests
  `TRAIN_YEARS + TEST_YEARS` manifest construction and included-race cleaning.
  Accordingly, 2025 was read for fixed ingestion, eligibility and requested
  quality counts. No features, model fitting, tuning, predictions or evaluation
  metrics were produced. The final-evaluation script and test-opened marker remain
  untouched. No thresholds were chosen from test outcomes.
- **Audit extension (§5):** preserve automatic decisions separately from human
  directives. A manual exclusion of an otherwise dry race has the explicit extra
  reason `manual_override`, plus `manual_note`; failed loads cannot be forced in.
  No manual override was applied in the real run. Review flags are diagnostics,
  including a fixed 50% clean-retention warning, not additional selection rules.
- **Structure extension (§1):** add `dataset.py` for reusable clean-lap assembly
  and `reporting.py` for measured log/summary generation, keeping numbered scripts
  thin. The catalog is packaged with setuptools. Part B of script 03 and all
  feature/modelling modules remain unimplemented.

---

### 2026-11-02 — Phase 3 — fixed feature preparation and eligibility accounting

The user explicitly requested the predefined feature transform and per-split row
counts for Phase 3. This extends the Phase 2 fixed-ingestion exception to the
standing test-lock wording: 2025 is transformed identically, with no test target
analysis, feature selection, tuning, predictions or evaluation metrics. The
training-only figure filters configured years and split labels before aggregating
any target. No final-test marker or model was created.

Guide §6's clean-current-lap condition is a precondition of `build_features`:
its input is Phase 2's saved `laps_clean.parquet`. F1 records that input unchanged;
F2–F4 report ordered first-removal counts for available history, consecutiveness
and stint agreement. Reapplying race-median outlier cleaning after mutating targets
would change the eligible input, so feature construction deliberately does not
repeat that target-dependent operation. `build_features` keeps the prescribed
DataFrame return; `build_features_with_audit` adds a companion audit contract.
No target definition, model feature list, cleaning threshold or dependency changed.

---

### 2026-11-03 — Phase 5 — Windows worker-pool constraint

The first real run completed Ridge search/OOF and logged its measured results,
then failed when Random Forest's `n_jobs=-1` attempted to create a joblib
ThreadPool. Python's Windows multiprocessing queue could not open its named pipe:
`PermissionError: [WinError 5] Access is denied`. No complete model-selection
result was claimed for that attempt.

The public pipeline defaults still use `n_jobs=-1` where supported. The script
now accepts `--estimator-jobs 1`, used for this environment's rerun, including
fold-local permutation importance. Outer searches/folds run serially to avoid
nested parallelism. This deviates from §8's parallel execution preference only;
the full grids, data, five race folds, seed, scoring and selection rules remain
unchanged. No dependency or permission setting was changed.


### 2026-11-03 — Phase 8 — reproduction verification scope

The required fresh online clone cannot be claimed: the supplied workspace lacks Git metadata and the pinned raw-input network probe failed with WinError 10061. Verification instead used an isolated source copy, the verified raw cache, and a separate clean offline-installed environment. All scripts reproduced their tables; original artifacts were unchanged. A copied opening marker plus --i-understand-rerun made the repeated final evaluation explicit. This is not a new holdout. The first PRD success checkbox remains open.

---

## Error analysis findings (Phase 7)

Seven measured findings and all five full breakdown tables are recorded in the
Phase 7 saved-prediction error analysis entry above. Figures 10 and 11 accompany
those results. The original Phase 6 prediction pass and model selection remain unchanged.

---

## Portfolio (Phase 8)

Resume bullet (from logged numbers only):

> _to be written_


### 2026-11-03 — Phase 8 — measured portfolio bullet

> Built a leakage-safe next-lap forecasting pipeline across 54 dry F1 races (2023–2025); a training-selected Random Forest reduced mean per-race MAE by 4.00% versus the last-lap baseline on 18 held-out 2025 races, with error analysis retaining failures on young tyres and individual circuits.
