"""Download the offline Bibles: python scripts/download_bibles.py [--force]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vachan.bible.builder import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
