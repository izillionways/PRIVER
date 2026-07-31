import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_diversity_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_diversity_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DiversityMetricTests(unittest.TestCase):
    def test_ndcg_rewards_earlier_positive_patch(self):
        positives = {"positive"}
        early = MODULE.ndcg_at_k(["positive", "negative"], positives, 2)
        late = MODULE.ndcg_at_k(["negative", "positive"], positives, 2)
        self.assertEqual(early, 1.0)
        self.assertLess(late, early)

    def test_semantic_nms_suppresses_overlapping_candidate(self):
        candidates = [
            {"patch_id": "a", "bbox": [0, 0, 10, 10]},
            {"patch_id": "b", "bbox": [1, 1, 11, 11]},
            {"patch_id": "c", "bbox": [20, 20, 30, 30]},
        ]
        selected = MODULE.semantic_nms(candidates, threshold=0.5, top_k=2)
        self.assertEqual([row["patch_id"] for row in selected], ["a", "c"])

    def test_object_recall_counts_unique_objects(self):
        query = {
            "query_id": "q",
            "image_id": "i",
            "class_name": "plane",
            "positive_object_ids": ["o1", "o2"],
        }
        selected = [{"patch_id": "p1"}, {"patch_id": "p2"}]
        result = MODULE.evaluate(
            query,
            selected,
            positive_patches={"p1", "p2"},
            patch_objects={"p1": {"o1"}, "p2": {"o1"}},
            top_k=2,
        )
        self.assertEqual(result["precision_at_10"], 1.0)
        self.assertEqual(result["unique_objects_at_10"], 1)
        self.assertEqual(result["object_recall_at_10"], 0.5)

    def test_cluster_bootstrap_preserves_positive_paired_delta(self):
        baseline = [
            {"query_id": "q1", "image_id": "i1", "ndcg_at_10": 0.2},
            {"query_id": "q2", "image_id": "i2", "ndcg_at_10": 0.4},
        ]
        method = [
            {"query_id": "q1", "image_id": "i1", "ndcg_at_10": 0.3},
            {"query_id": "q2", "image_id": "i2", "ndcg_at_10": 0.5},
        ]
        result = MODULE.paired_cluster_bootstrap(
            baseline, method, "ndcg_at_10", n_bootstrap=100, seed=42
        )
        self.assertAlmostEqual(result["delta"], 0.1)
        self.assertGreater(result["delta_ci95_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
