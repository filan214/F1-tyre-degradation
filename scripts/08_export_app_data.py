"""Phase 9: export saved 2025 results for the read-only Streamlit app."""

import json
import logging
import sys

from f1pace.app_data import MANIFEST_FILE, export_app_data
from f1pace.config import APP_DATA_DIR


def main() -> None:
    """Write app_data/ from saved outputs and print what was written."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s", stream=sys.stdout)
    manifest = export_app_data()
    print(json.dumps(manifest, indent=2))
    manifest_bytes = (APP_DATA_DIR / MANIFEST_FILE).stat().st_size
    total = sum(details["bytes"] for details in manifest["files"].values()) + manifest_bytes
    print(f"{MANIFEST_FILE}: {manifest_bytes} bytes; total: {total} bytes")


if __name__ == "__main__":
    main()
