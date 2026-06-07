import json
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

from run_storage import RunArtifactPaths, create_run_archive, load_manifest, parse_timestamp, write_manifest


class RunStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_artifact_paths_define_state_and_manifest_names_once(self):
        paths = RunArtifactPaths.in_directory(self.root, prefix="demo_")

        self.assertEqual(paths.state_paths()["dataset_file"], str(self.root / "demo_evaluation_dataset.json"))
        self.assertEqual(
            paths.state_paths()["refined_evaluation_pattern"],
            str(self.root / "demo_refined_evaluation_results_{iteration}.json"),
        )
        self.assertEqual(paths.manifest_artifacts()["archive"], "artifacts.zip")

    def test_manifest_round_trip_updates_timestamp(self):
        paths = RunArtifactPaths.in_directory(self.root)
        manifest = {"run_id": "run-one", "status": "running"}

        write_manifest(paths.manifest, manifest)
        loaded = load_manifest(paths.manifest)

        self.assertEqual(loaded["run_id"], "run-one")
        self.assertIsInstance(parse_timestamp(loaded["updated_at"]), datetime)

    def test_archive_contains_json_artifacts_but_not_checkpoint(self):
        paths = RunArtifactPaths.in_directory(self.root)
        write_manifest(paths.manifest, {"run_id": "run-one"})
        paths.dataset.write_text(json.dumps([{"task": "task"}]), encoding="utf-8")
        paths.checkpoint.write_text("checkpoint", encoding="utf-8")

        archive_path = create_run_archive(paths)

        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        self.assertEqual(names, {"evaluation_dataset.json", "run_manifest.json"})


if __name__ == "__main__":
    unittest.main()
