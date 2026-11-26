#!/usr/bin/env python3
"""
tests/test_brepnet_retrieval.py — brepnet_retrieval 的自动化测试。

覆盖：加载、评分、排序、索引校验、合成数据自身检索、结果导出、CLI。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brepnet_retrieval as br


class TestLoadEmbeddings(unittest.TestCase):
    def test_load_txt_multi_face(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.embeddings"
            p.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
            arr = br.load_embeddings_file(p)
            self.assertEqual(arr.shape, (2, 3))

    def test_load_txt_single_face_is_2d(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "single.embeddings"
            p.write_text("1 2 3\n", encoding="utf-8")
            arr = br.load_embeddings_file(p)
            self.assertEqual(arr.shape, (1, 3))


class TestScoring(unittest.TestCase):
    def test_self_score_is_zero(self):
        emb = np.array([[0.0, 0.0], [1.0, 0.0]])
        s = br.score_query_against_candidate(emb, emb)
        self.assertAlmostEqual(s, 0.0, places=9)

    def test_directed_distance_value(self):
        q = np.array([[0.0, 0.0]])
        t = np.array([[3.0, 4.0]])  # 距离 5
        s = br.score_query_against_candidate(q, t)
        self.assertAlmostEqual(s, 5.0, places=9)

    def test_dim_mismatch_raises(self):
        q = np.zeros((1, 2))
        t = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            br.score_query_against_candidate(q, t)


class TestRanking(unittest.TestCase):
    def test_stable_order(self):
        scores = [5.0, 0.0, 3.0]
        order = br.rank_candidates(scores)
        self.assertEqual(list(order), [1, 2, 0])


class TestValidateFaces(unittest.TestCase):
    def test_all_faces(self):
        emb = np.zeros((4, 2))
        self.assertEqual(br.validate_faces(emb, "all", "m"), [0, 1, 2, 3])

    def test_duplicate_raises(self):
        emb = np.zeros((4, 2))
        with self.assertRaises(ValueError):
            br.validate_faces(emb, [0, 0], "m")

    def test_out_of_range_raises(self):
        emb = np.zeros((4, 2))
        with self.assertRaises(ValueError):
            br.validate_faces(emb, [0, 5], "m")


class TestSyntheticSelfRetrieval(unittest.TestCase):
    def test_all_six_self_top1(self):
        emb_map = br.make_synthetic_embeddings(seed=20260902)
        summary = br.run_all_queries(emb_map, faces="all", top_k=5)
        self.assertEqual(summary["n_queries"], 6)
        self.assertEqual(summary["self_top1_hits"], 6)
        self.assertAlmostEqual(summary["self_top1_rate"], 1.0, places=9)
        self.assertAlmostEqual(summary["mean_self_rank"], 1.0, places=9)
        self.assertTrue(summary["all_zero_score"])


class TestExportResults(unittest.TestCase):
    def test_export_files(self):
        emb_map = br.make_synthetic_embeddings(seed=20260902)
        summary = br.run_all_queries(emb_map, faces="all", top_k=5)
        with tempfile.TemporaryDirectory() as d:
            results_dir = br.export_results(summary, Path(d))
            self.assertTrue((results_dir / "summary.json").exists())
            self.assertTrue((results_dir / "per_query.csv").exists())
            self.assertTrue((results_dir / "top_k.csv").exists())
            with open(results_dir / "summary.json", encoding="utf-8") as f:
                sj = json.load(f)
            self.assertEqual(sj["self_top1_hits"], 6)
            self.assertAlmostEqual(sj["self_top1_rate"], 1.0, places=9)


class TestCLI(unittest.TestCase):
    def test_demo_end_to_end(self):
        import subprocess
        root = Path(__file__).resolve().parent.parent
        script = root / "brepnet_retrieval.py"
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, str(script), "demo",
                 "--output-dir", str(Path(d) / "run"),
                 "--top-k", "5"],
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("100.00%", r.stdout)
            sj_path = Path(d) / "run" / "results" / "summary.json"
            self.assertTrue(sj_path.exists())


if __name__ == "__main__":
    unittest.main()
