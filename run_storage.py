"""Shared artifact, manifest, and archive contract for evaluation runs."""

import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from json import dumps, loads
from pathlib import Path
from typing import Any

DATASET_FILE: str = "evaluation_dataset.json"
SOLUTIONS_FILE: str = "solutions.json"
EVALUATION_FILE: str = "evaluation_results.json"
BEST_DATASET_FILE: str = "best_dataset.json"
REFINED_DATASET_PATTERN: str = "refined_dataset_{iteration}.json"
REFINED_SOLUTIONS_PATTERN: str = "refined_solutions_{iteration}.json"
REFINED_EVALUATION_PATTERN: str = "refined_evaluation_results_{iteration}.json"
DEFAULT_CHECKPOINT_DB: str = "langgraph_checkpoints.sqlite"
DEFAULT_RUNS_ROOT: str = "runs"
RUN_MANIFEST_FILE: str = "run_manifest.json"
RUN_ARCHIVE_FILE: str = "artifacts.zip"


@dataclass(frozen=True)
class RunArtifactPaths:
    """Paths for every artifact owned by one evaluation run."""

    dataset: Path
    solutions: Path
    evaluation: Path
    best_dataset: Path
    refined_dataset_pattern: Path
    refined_solutions_pattern: Path
    refined_evaluation_pattern: Path
    checkpoint: Path
    manifest: Path
    archive: Path

    @classmethod
    def in_directory(cls, directory: str | Path, prefix: str = "") -> "RunArtifactPaths":
        """Build the complete artifact layout beneath a directory.

        Args:
            directory (str | Path): Parent directory for artifacts.
            prefix (str): Optional filename prefix, such as ``"demo_"``.

        Returns:
            RunArtifactPaths: Complete artifact path set.
        """
        root: Path = Path(directory)
        return cls(
            dataset=root / f"{prefix}{DATASET_FILE}",
            solutions=root / f"{prefix}{SOLUTIONS_FILE}",
            evaluation=root / f"{prefix}{EVALUATION_FILE}",
            best_dataset=root / f"{prefix}{BEST_DATASET_FILE}",
            refined_dataset_pattern=root / f"{prefix}{REFINED_DATASET_PATTERN}",
            refined_solutions_pattern=root / f"{prefix}{REFINED_SOLUTIONS_PATTERN}",
            refined_evaluation_pattern=root / f"{prefix}{REFINED_EVALUATION_PATTERN}",
            checkpoint=root / f"{prefix}{DEFAULT_CHECKPOINT_DB}",
            manifest=root / RUN_MANIFEST_FILE,
            archive=root / RUN_ARCHIVE_FILE,
        )

    def state_paths(self) -> dict[str, str]:
        """Return graph-state artifact path values.

        Returns:
            dict[str, str]: Artifact paths keyed by their ``AgentState`` names.
        """
        return {
            "dataset_file": str(self.dataset),
            "solutions_file": str(self.solutions),
            "evaluation_file": str(self.evaluation),
            "best_dataset_file": str(self.best_dataset),
            "refined_dataset_pattern": str(self.refined_dataset_pattern),
            "refined_solutions_pattern": str(self.refined_solutions_pattern),
            "refined_evaluation_pattern": str(self.refined_evaluation_pattern),
        }

    def manifest_artifacts(self) -> dict[str, str]:
        """Return artifact names recorded in a run manifest.

        Returns:
            dict[str, str]: Run-local artifact filenames keyed by artifact role.
        """
        return {
            "dataset": self.dataset.name,
            "solutions": self.solutions.name,
            "evaluation": self.evaluation.name,
            "best_dataset": self.best_dataset.name,
            "refined_dataset_pattern": self.refined_dataset_pattern.name,
            "refined_solutions_pattern": self.refined_solutions_pattern.name,
            "refined_evaluation_pattern": self.refined_evaluation_pattern.name,
            "archive": self.archive.name,
        }


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format.

    Returns:
        str: Current UTC time with a ``Z`` suffix.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp as an aware UTC datetime.

    Args:
        value (Any): Candidate timestamp value.

    Returns:
        datetime | None: Parsed UTC timestamp, or ``None`` when invalid.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed: datetime = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a run manifest with an updated timestamp.

    Args:
        path (Path): Manifest output path.
        manifest (dict[str, Any]): Mutable manifest values to persist.
    """
    manifest["updated_at"] = utc_timestamp()
    path.write_text(dumps(manifest, indent=4), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a run manifest.

    Args:
        path (Path): Existing manifest path.

    Returns:
        dict[str, Any]: Decoded manifest object.

    Raises:
        ValueError: If the manifest does not contain a JSON object.
    """
    manifest: Any = loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must contain a JSON object: {path}")
    return manifest


def run_updated_at(run_directory: Path) -> datetime:
    """Return manifest update time or the directory modification time.

    Args:
        run_directory (Path): Run directory to inspect.

    Returns:
        datetime: Aware UTC timestamp used to determine run age.
    """
    try:
        timestamp: datetime | None = parse_timestamp(load_manifest(run_directory / RUN_MANIFEST_FILE).get("updated_at"))
        if timestamp is not None:
            return timestamp
    except (OSError, ValueError):
        pass
    return datetime.fromtimestamp(run_directory.stat().st_mtime, tz=timezone.utc)


def create_run_archive(paths: RunArtifactPaths) -> Path:
    """Create a ZIP containing the run manifest and generated JSON artifacts.

    Args:
        paths (RunArtifactPaths): Artifact layout for the run to package.

    Returns:
        Path: Absolute path to the rebuilt ZIP archive.
    """
    with zipfile.ZipFile(paths.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(paths.archive.parent.glob("*.json")):
            if artifact.is_file():
                archive.write(artifact, arcname=artifact.name)
    return paths.archive.resolve()
