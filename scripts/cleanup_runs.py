"""Delete outdated isolated prompt-evaluation run folders."""

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_storage import DEFAULT_RUNS_ROOT, run_updated_at

DEFAULT_MAX_AGE_HOURS: float = 24.0


def cleanup_runs(
    runs_root: str = DEFAULT_RUNS_ROOT,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    dry_run: bool = False,
    now: datetime | None = None,
) -> int:
    """Delete direct child run folders older than the configured threshold.

    Args:
        runs_root (str): Parent directory containing isolated run folders.
        max_age_hours (float): Age threshold in hours.
        dry_run (bool): Print candidates without deleting them.
        now (datetime | None): Current time override used by tests.

    Returns:
        int: Number of cleanup failures.
    """
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative.")

    root: Path = Path(runs_root).resolve()
    if not root.exists():
        print(f"Runs root does not exist: {root}")
        return 0
    if not root.is_dir():
        raise ValueError(f"Runs root is not a directory: {root}")

    current_time: datetime = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    failures: int = 0

    for child in sorted(root.iterdir()):
        if child.is_symlink():
            print(f"Skipped symlink: {child}")
            continue
        if not child.is_dir():
            continue

        resolved_child: Path = child.resolve()
        if resolved_child.parent != root:
            print(f"Skipped unsafe path: {child}")
            failures += 1
            continue

        age_hours: float = (current_time - run_updated_at(child)).total_seconds() / 3600
        if age_hours <= max_age_hours:
            print(f"Skipped recent run: {child}")
            continue

        if dry_run:
            print(f"Would delete: {child}")
            continue

        try:
            shutil.rmtree(child)
            print(f"Deleted: {child}")
        except OSError as exc:
            failures += 1
            print(f"Failed to delete {child}: {exc}")

    return failures


def _build_parser() -> argparse.ArgumentParser:
    """Build the cleanup command-line parser."""
    parser = argparse.ArgumentParser(description="Delete outdated prompt-evaluation run folders.")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Run cleanup from command-line arguments."""
    args = _build_parser().parse_args()
    return cleanup_runs(args.runs_root, args.max_age_hours, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
