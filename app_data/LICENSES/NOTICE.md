# Third-party notices for app_data

The result files in `app_data/` are derived from public FastF1-derived TracingInsights archives. They are not the original archive files.

| Archive | Pinned commit | Licence copy |
|---|---|---|
| [TracingInsights/2023](https://github.com/TracingInsights/2023/tree/737c84d3ae14b9883a671ca248d015b69cf1cb58) | `737c84d3ae14b9883a671ca248d015b69cf1cb58` | `TracingInsights-2023-LICENSE.txt` (MIT) |
| [TracingInsights/2024](https://github.com/TracingInsights/2024/tree/d1543a71370d9b617b627319f8ad80b2f85dfd5e) | `d1543a71370d9b617b627319f8ad80b2f85dfd5e` | `TracingInsights-2024-LICENSE.txt` (MIT) |
| [TracingInsights/2025](https://github.com/TracingInsights/2025/tree/dfb3ea3f50b69857fe8e40cabe5e899743c59f6a) | `dfb3ea3f50b69857fe8e40cabe5e899743c59f6a` | `TracingInsights-2025-LICENSE.txt` (Apache-2.0) |

## Changes made to the source data

- `test_predictions.parquet` holds 2025 lap data after this project's cleaning and eligibility rules. Races and laps were removed, and only selected columns were kept. Lap-time predictions from this project's models and baselines were added.
- The CSV tables are this project's error summaries. The 2025 tables are computed from those predictions. `cv_summary.csv` is computed from the 2023 and 2024 archives.

The cleaning rules and method are described in the project README.

## Trade marks

This is an unofficial project, not associated with the Formula 1 companies or the FIA. F1, FORMULA 1 and GRAND PRIX are trade marks of Formula One Licensing B.V.
