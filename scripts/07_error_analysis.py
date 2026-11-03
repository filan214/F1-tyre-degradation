"""Phase 7: analyse saved final-test errors without running models again."""

import logging

import matplotlib
matplotlib.use('Agg')

from f1pace import reporting
from f1pace.error_analysis import run_error_analysis


def main() -> None:
    """Save descriptive comparisons and report seven measured findings."""
    reporting.configure_logging()
    try:
        _, findings = run_error_analysis()
    except Exception as exc:
        logging.getLogger(__name__).exception('Phase 7 failed: %s', exc)
        raise SystemExit(1) from exc
    for index, finding in enumerate(findings, 1):
        print(f'{index}. {finding}')


if __name__ == '__main__':
    main()
