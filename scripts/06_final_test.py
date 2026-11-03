"""Phase 6: evaluate frozen models on 2025 once; explicitly guard repeat runs."""

import argparse
import logging

import matplotlib
matplotlib.use('Agg')

from f1pace import reporting
from f1pace.final_test import run_final_test


def main() -> None:
    """Run the final evaluation and print its complete seven-predictor summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--i-understand-rerun', action='store_true', help='Explicitly repeat evaluation of an already opened test set; logged as a deviation')
    args = parser.parse_args()
    reporting.configure_logging()
    try:
        summary = run_final_test(understand_rerun=args.i_understand_rerun)
    except Exception as exc:
        logging.getLogger(__name__).exception('Phase 6 failed: %s', exc)
        raise SystemExit(1) from exc
    print(summary.to_string(index=False))
    print('The 2025 test set is now opened. Frozen training choices are unchanged.')


if __name__ == '__main__':
    main()
