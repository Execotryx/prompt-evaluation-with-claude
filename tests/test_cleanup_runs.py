import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.cleanup_runs import cleanup_runs


class CleanupRunsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "runs"
        self.root.mkdir()
        self.now = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_run(self, name: str, age_hours: float, valid_manifest: bool = True) -> Path:
        run_directory = self.root / name
        run_directory.mkdir()
        timestamp = self.now - timedelta(hours=age_hours)
        manifest = run_directory / "run_manifest.json"
        if valid_manifest:
            manifest.write_text(json.dumps({"updated_at": timestamp.isoformat()}), encoding="utf-8")
        else:
            manifest.write_text("{invalid", encoding="utf-8")
            epoch = timestamp.timestamp()
            os.utime(run_directory, (epoch, epoch))
        return run_directory

    def test_deletes_only_runs_older_than_boundary(self):
        old = self.create_run("old", 25)
        boundary = self.create_run("boundary", 24)
        recent = self.create_run("recent", 2)

        failures = cleanup_runs(str(self.root), max_age_hours=24, now=self.now)

        self.assertEqual(failures, 0)
        self.assertFalse(old.exists())
        self.assertTrue(boundary.exists())
        self.assertTrue(recent.exists())

    def test_dry_run_and_malformed_manifest_fallback(self):
        old = self.create_run("old-malformed", 25, valid_manifest=False)

        failures = cleanup_runs(str(self.root), max_age_hours=24, dry_run=True, now=self.now)

        self.assertEqual(failures, 0)
        self.assertTrue(old.exists())

    def test_skips_directory_symlinks(self):
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        link = self.root / "linked-run"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks are unavailable in this environment.")

        failures = cleanup_runs(str(self.root), max_age_hours=0, now=self.now)

        self.assertEqual(failures, 0)
        self.assertTrue(outside.exists())
        self.assertTrue(link.exists())


if __name__ == "__main__":
    unittest.main()
