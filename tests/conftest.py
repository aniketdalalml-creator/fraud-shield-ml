from __future__ import annotations

import sys
from pathlib import Path


# Ensure the repository root is on sys.path so imports like `import src...` and
# `import api...` work reliably when pytest changes the working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

