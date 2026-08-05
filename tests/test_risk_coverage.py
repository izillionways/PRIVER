import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "risk_coverage.py"
SPEC = importlib.util.spec_from_file_location("risk_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RiskCoverageTests(unittest.TestCase):
    def test_confidence_features_use_only_final_top_k_results(self):
        retrieved = [
            {"priver_score": 2.0, "bbox": [0, 0, 10, 10], "patch_size": 512},
            {"priver_score": 1.0, "bbox": [1, 1, 11, 11], "patch_size": 1024},
            {
                "priver_score": -100.0,
                "bbox": [1000, 1000, 1010, 1010],
                "patch_size": 512,
            },
        ]
        row = {"retrieved": retrieved}

        top_two = MODULE.retrieval_confidence_features(row, top_k=2)
        explicitly_truncated = MODULE.retrieval_confidence_features(
            {"retrieved": retrieved[:2]}, top_k=2
        )
        all_three = MODULE.retrieval_confidence_features(row, top_k=3)

        self.assertEqual(top_two, explicitly_truncated)
        self.assertNotEqual(
            top_two["neg_spatial_center_std"],
            all_three["neg_spatial_center_std"],
        )


if __name__ == "__main__":
    unittest.main()
