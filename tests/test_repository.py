"""Read-only checks for a standalone Lab I checkout; no fitting or simulation."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/calibration"), str(ROOT / "src/tgsim")]
import idm_core
import tgsim_core


class RepositoryTests(unittest.TestCase):
    def test_original_source_manifest(self):
        manifest = json.loads((ROOT / "review/source_manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_human_collection_registry_hashes(self):
        registry = json.loads((ROOT / "data/run_registry.json").read_text())
        self.assertEqual(set(registry["runs"]), {"run_1", "run_2"})
        for record in registry["runs"].values():
            source = ROOT / record["source"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), record["sha256"])

    def test_selected_calibration_events_load(self):
        events = idm_core.load_event_data(ROOT)["run_2"]
        self.assertEqual(len(events), 8)
        self.assertEqual(sum(event.n for _, event in events), 1140)

    def test_real_world_inputs_load(self):
        observations = tgsim_core.raw_unit_check()
        self.assertEqual(set(observations), {"HV-HV", "HV-AV"})
        for pair, record in observations.items():
            self.assertGreater(record["n_follower_samples"], 0)
            self.assertTrue((ROOT / record["source"]).is_file())

    def test_validation_record_identity(self):
        expected = {"152115": (737, 312, 17, 5), "155015": (749, 319, 0, 0)}
        for suffix, values in expected.items():
            path = ROOT / f"data/validation/idm_test_20260922_{suffix}/commands.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [r for r in csv.DictReader(stream) if r["role"] == "follower"]
            flags = [float(r["emergency"]) > 0 for r in rows]
            episodes = sum(flag and (i == 0 or not flags[i-1]) for i, flag in enumerate(flags))
            self.assertEqual((len(rows), sum(r["mode"] == "idm_straight" for r in rows),
                              sum(flags), episodes), values)

    def test_repository_assets_are_local(self):
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or not path.is_symlink():
                continue
            self.assertTrue(path.exists(), str(path))
            self.assertTrue(path.resolve().is_relative_to(ROOT), str(path))
        for name in ["environment/idm_controller.py", "environment/6e68_map.yaml",
                     "data/raw/tgsim/HV-HV.csv", "data/raw/tgsim/HV-AV.csv"]:
            self.assertTrue((ROOT / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
