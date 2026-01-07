import sys
from pathlib import Path


def pytest_configure() -> None:
    """Make `src/` importable in tests without requiring installation."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
