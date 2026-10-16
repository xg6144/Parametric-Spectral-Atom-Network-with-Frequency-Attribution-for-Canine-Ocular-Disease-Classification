import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import deduplicate_dataset as dedup


class DeduplicateDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.root = self.base / "raw"
        (self.root / "A").mkdir(parents=True)
        (self.root / "B").mkdir(parents=True)

    def tearDown(self):
        self.tempdir.cleanup()

    def _manifest(self, out: Path):
        with (out / "dedup_manifest.csv").open(newline="", encoding="utf-8") as handle:
            return {row["path"]: row for row in csv.DictReader(handle)}

    def test_exact_pixel_duplicates_and_cross_class_conflict(self):
        y, x = np.mgrid[:64, :64]
        array1 = np.stack(((x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256), axis=2).astype(np.uint8)
        image1 = Image.fromarray(array1, "RGB")
        image1.save(self.root / "A" / "a.png", compress_level=1)
        image1.save(self.root / "A" / "a_reencoded.png", compress_level=9)
        shutil.copy2(self.root / "A" / "a.png", self.root / "B" / "a_copy.png")

        array2 = np.stack(((x * 3) % 256, (y * 2) % 256, ((x * y) % 256)), axis=2).astype(np.uint8)
        image2 = Image.fromarray(array2, "RGB")
        image2.save(self.root / "A" / "c.png")
        shutil.copy2(self.root / "A" / "c.png", self.root / "A" / "d.png")

        out = self.base / "audit"
        rc = dedup.main([
            "--root", str(self.root), "--out", str(out),
            "--workers", "1", "--exact-only",
        ])
        self.assertEqual(rc, 0)
        rows = self._manifest(out)
        for path in ("A/a.png", "A/a_reencoded.png", "B/a_copy.png"):
            self.assertEqual(rows[path]["status"], "exclude_label_conflict")
        intra = [rows["A/c.png"]["status"], rows["A/d.png"]["status"]]
        self.assertEqual(sorted(intra), ["exclude_duplicate", "keep"])

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["inventory"]["images"], 5)
        self.assertEqual(summary["result"]["kept_images"], 1)

    def test_near_duplicates_are_report_only_until_explicitly_accepted(self):
        rng = np.random.default_rng(42)
        array = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        image = Image.fromarray(array, "RGB")
        image.save(self.root / "A" / "quality95.jpg", quality=95)
        image.save(self.root / "A" / "quality90.jpg", quality=90)

        report_out = self.base / "report"
        dedup.main([
            "--root", str(self.root), "--out", str(report_out), "--workers", "1",
            "--max-mae", "0.05", "--max-rmse", "0.10", "--min-correlation", "0.95",
        ])
        report_rows = self._manifest(report_out)
        self.assertEqual({row["status"] for row in report_rows.values()}, {"keep"})

        accepted_out = self.base / "accepted"
        dedup.main([
            "--root", str(self.root), "--out", str(accepted_out), "--workers", "1",
            "--max-mae", "0.05", "--max-rmse", "0.10", "--min-correlation", "0.95",
            "--accept-verified-near",
        ])
        accepted_rows = self._manifest(accepted_out)
        self.assertEqual(
            sorted(row["status"] for row in accepted_rows.values()),
            ["exclude_duplicate", "keep"],
        )

    def test_clique_cover_does_not_apply_transitive_chain(self):
        edges = {(0, 1), (1, 2)}
        self.assertEqual(dedup.clique_cover([0, 1, 2], edges), [[0, 1]])

    def test_every_accepted_near_edge_is_resolved(self):
        records = [
            dedup.ImageRecord(
                index=i, rel_path=f"A/{i}.png", class_name="A", extension=".png",
                file_size=10 + i, mtime_ns=0, sha256=str(i), pixel_sha256=str(i),
                width=10, height=10, phash=f"{i:016x}", dhash=f"{i:016x}", error="",
            )
            for i in range(3)
        ]
        status, _, _, _ = dedup.resolve_groups(
            records, [[0], [1], [2]], [(0, 1), (0, 2)], True
        )
        self.assertEqual(status.count("keep"), 1)
        for left, right in ((0, 1), (0, 2)):
            self.assertFalse(status[left] == status[right] == "keep")


if __name__ == "__main__":
    unittest.main()
