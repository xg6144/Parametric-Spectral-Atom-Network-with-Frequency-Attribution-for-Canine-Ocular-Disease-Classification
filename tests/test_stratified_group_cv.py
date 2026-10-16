import csv
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from prepare_stratified_group_folds import (
    assign_stratified_group_folds,
    load_kept_records,
    validate_fold_records,
)


class StratifiedGroupFoldTests(unittest.TestCase):
    def test_group_never_crosses_fold_and_all_classes_are_present(self):
        rows = []
        for class_name in ("A", "B", "C"):
            for group_number in range(10):
                group_id = f"{class_name}-G{group_number:02d}"
                for member in range(2):
                    rows.append(
                        {
                            "path": f"{class_name}/{group_number}_{member}.jpg",
                            "class_name": class_name,
                            "sha256": f"{class_name}{group_number}{member}".lower(),
                            "visual_group": group_id,
                            "group_id": group_id,
                        }
                    )

        records, classes = assign_stratified_group_folds(
            rows,
            n_splits=5,
            seed=42,
        )
        validate_fold_records(records, classes, n_splits=5)

        group_folds = defaultdict(set)
        fold_classes = defaultdict(set)
        for record in records:
            group_folds[record.group_id].add(record.fold)
            fold_classes[record.fold].add(record.class_name)
        self.assertTrue(all(len(folds) == 1 for folds in group_folds.values()))
        self.assertEqual(set(fold_classes), {1, 2, 3, 4, 5})
        self.assertTrue(all(names == set(classes) for names in fold_classes.values()))

    def test_only_keep_rows_and_singletons_get_stable_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "audit.csv"
            fields = ["path", "class", "sha256", "status", "visual_group"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "path": "A/a.jpg",
                            "class": "A",
                            "sha256": "abc",
                            "status": "keep",
                            "visual_group": "",
                        },
                        {
                            "path": "A/b.jpg",
                            "class": "A",
                            "sha256": "def",
                            "status": "keep",
                            "visual_group": "V000001",
                        },
                        {
                            "path": "A/c.jpg",
                            "class": "A",
                            "sha256": "ghi",
                            "status": "exclude_duplicate",
                            "visual_group": "V000002",
                        },
                    ]
                )

            rows = load_kept_records(manifest, verify_files=False)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["group_id"], "S:abc")
            self.assertEqual(rows[1]["group_id"], "V000001")

    def test_too_few_groups_for_a_class_fails(self):
        rows = [
            {
                "path": f"A/{index}.jpg",
                "class_name": "A",
                "sha256": str(index),
                "visual_group": "same-group",
                "group_id": "same-group",
            }
            for index in range(10)
        ]
        with self.assertRaisesRegex(ValueError, "at least n_splits"):
            assign_stratified_group_folds(rows, n_splits=5, seed=42)


if __name__ == "__main__":
    unittest.main()
