#!/usr/bin/env python3
"""
brepnet_retrieval.py — BRepNet 模型自身检索实验的独立检索层。

背景
----
BRepNet (Lambourne et al., CVPR 2021) 是一个面向 CAD 边界表示 (B-Rep) 的
拓扑消息传递网络，原始任务是「逐面语义分割」。在测试阶段，官方代码可以把
每个面的隐藏表示 (embedding) 导出为纯文本 `.embeddings` 文件：
    每一行 = 一个面；每一列 = 一个嵌入维度。

本项目只回答一个刻意收窄的问题：
    给定由 BRepNet 生成的「面级嵌入」，当查询模型本身也位于候选库中时，
    检索系统能否把它自己找回来（自身 Top-1）？

这个检索层**不依赖** OCC / PyTorch / BRepNet，只用 NumPy，因此可以在任意
Python 3.9+ 环境里独立运行和复现。它的作用是打通「读取嵌入 → 距离计算 →
候选排序 → 结果统计与导出」这条链路，作为后续真实 STEP 推理前的 sanity check。

评分公式（与官方 similarity-search notebook 一致）
---------------------------------------------------
设查询模型被选中的面嵌入集合为 Q={q_i}，候选模型全部面嵌入为 T={t_j}：
    score(Q, T) = Σ_i  min_j  || q_i - t_j ||_2
分数越小越相似。这是「查询 → 候选」的定向距离。

当候选 T 就是查询模型自身时，每个 q_i 都能在 T 中找到完全相同的向量，
最小欧氏距离为 0，求和仍为 0，因此自身取得唯一最低分（若无重复嵌入）。

工程修正（相对官方 notebook）
-----------------------------
1. Top-k 排序：官方用 `np.argpartition` 只保证选出最小 k 个、不保证顺序；
   本项目改用完整稳定排序 (`np.argsort`)，保证输出顺序即真实排名。
2. 单面模型：加载函数统一保证 `faces × features` 的二维形状。
3. 输入校验：检查面索引重复/越界、特征维度不一致。
4. 结果留存：导出 JSON（summary）、查询汇总 CSV、Top-k CSV。

命令
----
    # 合成演示：生成 6 个合成模型，验证自身 Top-1
    python brepnet_retrieval.py demo --output-dir experiment_runs/synthetic_demo --top-k 5

    # 真实推理：读取 BRepNet 导出的 .embeddings 目录，做全库自身检索
    python brepnet_retrieval.py evaluate \
        --embeddings-dir <path/to/embeddings> \
        --output-dir real_results --faces all --top-k 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# 数据加载与校验
# ---------------------------------------------------------------------------

def load_embeddings_file(path):
    """
    读取单个 `.embeddings`（或 `.txt` / `.npy`）文件为二维数组 (faces, dim)。

    官方 `.embeddings` 是纯文本：每行一个面，列用空白分隔。
    对单面模型（只有一行）也统一返回二维形状。
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float64)
    else:
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    # 单面模型的保障：若误读成一维，扩成 (1, dim)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"期望二维 (faces, dim)，实际为 {arr.ndim} 维: {path}")
    return arr


def validate_faces(embeddings, faces, model_name):
    """
    校验查询面索引：非空、无重复、无越界。
    faces 为 None 或 "all" 时返回全部面索引。
    """
    n_faces = embeddings.shape[0]
    if faces is None or (isinstance(faces, str) and faces.lower() == "all"):
        return list(range(n_faces))
    faces = [int(f) for f in faces]
    if len(faces) == 0:
        raise ValueError(f"{model_name}: 查询面索引为空")
    if len(set(faces)) != len(faces):
        raise ValueError(f"{model_name}: 查询面索引存在重复: {faces}")
    if any(f < 0 or f >= n_faces for f in faces):
        raise ValueError(f"{model_name}: 查询面索引越界 (面数={n_faces}): {faces}")
    return faces


# ---------------------------------------------------------------------------
# 评分与排序
# ---------------------------------------------------------------------------

def score_query_against_candidate(query_emb, cand_emb):
    """
    计算查询模型对候选模型的定向距离。

    query_emb: (Q, dim)  被选中的查询面
    cand_emb:  (T, dim)  候选模型全部面
    返回标量 score = Σ_i min_j || q_i - t_j ||_2
    """
    if query_emb.shape[1] != cand_emb.shape[1]:
        raise ValueError(
            f"特征维度不一致: query={query_emb.shape[1]}, cand={cand_emb.shape[1]}"
        )
    # (Q, T) 两两欧氏距离
    diff = query_emb[:, None, :] - cand_emb[None, :, :]
    dists = np.linalg.norm(diff, axis=2)          # (Q, T)
    nearest = dists.min(axis=1)                    # (Q,)
    return float(nearest.sum())


def rank_candidates(scores):
    """
    稳定排序：分数从小到大，返回 (名次数组, 排序索引)。
    使用 np.argsort(kind='stable') 保证并列分数时顺序稳定。
    """
    order = np.argsort(scores, kind="stable")
    return order


# ---------------------------------------------------------------------------
# 检索主流程
# ---------------------------------------------------------------------------

def retrieve(embeddings_map, query_name, faces="all", top_k=5):
    """
    以 query_name 为查询，遍历所有候选模型，返回有序检索结果。

    embeddings_map: {模型名: (faces, dim) 数组}
    返回 dict:
        query_name, n_faces, results: [ {rank, name, score, is_self}, ... ]
        self_rank, self_score, top_k_hits
    """
    query_emb = embeddings_map[query_name]
    q_indices = validate_faces(query_emb, faces, query_name)
    query_faces = query_emb[q_indices]

    scores = {}
    for cand_name, cand_emb in embeddings_map.items():
        scores[cand_name] = score_query_against_candidate(query_faces, cand_emb)

    order = rank_candidates(list(scores.values()))
    names = list(scores.keys())
    ranked = []
    for rank_i, idx in enumerate(order, start=1):
        name = names[idx]
        ranked.append({
            "rank": rank_i,
            "name": name,
            "score": scores[name],
            "is_self": (name == query_name),
        })

    self_rank = next(r["rank"] for r in ranked if r["is_self"])
    self_score = scores[query_name]
    return {
        "query_name": query_name,
        "n_query_faces": len(q_indices),
        "results": ranked,
        "self_rank": self_rank,
        "self_score": self_score,
        "top_k_hits": ranked[:top_k],
    }


def run_all_queries(embeddings_map, faces="all", top_k=5):
    """每个模型轮流作为查询，汇总全库自身检索统计。"""
    per_query = []
    for qname in embeddings_map:
        per_query.append(retrieve(embeddings_map, qname, faces=faces, top_k=top_k))

    n = len(per_query)
    n_top1 = sum(1 for r in per_query if r["self_rank"] == 1)
    self_ranks = [r["self_rank"] for r in per_query]
    self_scores = [r["self_score"] for r in per_query]

    return {
        "per_query": per_query,
        "n_queries": n,
        "self_top1_hits": n_top1,
        "self_top1_rate": (n_top1 / n) if n else 0.0,
        "mean_self_rank": float(np.mean(self_ranks)) if n else 0.0,
        "self_scores": self_scores,
        "all_zero_score": all(abs(s) < 1e-9 for s in self_scores),
    }


# ---------------------------------------------------------------------------
# 合成数据（demo）
# ---------------------------------------------------------------------------

def make_synthetic_embeddings(seed=20260902):
    """
    生成 6 个合成模型的面嵌入，用于验证数据形状与距离计算。
    不同模型用不同均值的随机向量构造，保证彼此可区分、自身零距离。
    """
    rng = np.random.default_rng(seed)
    specs = {
        "bracket_A": 4,
        "bracket_B": 5,
        "cover_A":   6,
        "flange_A":  4,
        "housing_A": 5,
        "shaft_A":   6,
    }
    dim = 16
    embeddings_map = {}
    for name, n_faces in specs.items():
        mean = rng.normal(0, 10, size=dim)
        embeddings_map[name] = rng.normal(mean, 0.5, size=(n_faces, dim))
    return embeddings_map


# ---------------------------------------------------------------------------
# 结果导出
# ---------------------------------------------------------------------------

def export_results(summary, output_dir):
    """导出 summary.json、per-query.csv、top-k.csv。"""
    output_dir = Path(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    summary_payload = {
        "n_queries": summary["n_queries"],
        "self_top1_hits": summary["self_top1_hits"],
        "self_top1_rate": summary["self_top1_rate"],
        "mean_self_rank": summary["mean_self_rank"],
        "self_scores": summary["self_scores"],
        "all_zero_score": summary["all_zero_score"],
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # per-query.csv
    import csv
    with open(results_dir / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "n_faces", "self_rank", "self_score",
                    "nearest_nonself", "nonself_score"])
        for r in summary["per_query"]:
            nonself = next((x for x in r["results"] if not x["is_self"]), None)
            w.writerow([
                r["query_name"], r["n_query_faces"], r["self_rank"],
                f"{r['self_score']:.6f}",
                nonself["name"] if nonself else "",
                f"{nonself['score']:.6f}" if nonself else "",
            ])

    # top-k.csv（每个查询的前 k 个候选）
    with open(results_dir / "top_k.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "rank", "candidate", "score", "is_self"])
        for r in summary["per_query"]:
            for hit in r["top_k_hits"]:
                w.writerow([r["query_name"], hit["rank"], hit["name"],
                            f"{hit['score']:.6f}", int(hit["is_self"])])

    return results_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_demo(args):
    print("=" * 64)
    print("BRepNet 自身检索 — 合成演示 (synthetic demo)")
    print("=" * 64)
    embeddings_map = make_synthetic_embeddings(seed=20260902)
    print(f"生成 {len(embeddings_map)} 个合成模型:")
    for name, arr in embeddings_map.items():
        print(f"  {name:<12s} {arr.shape[0]} faces × {arr.shape[1]} dims")

    summary = run_all_queries(embeddings_map, faces="all", top_k=args.top_k)

    print("\n" + "─" * 64)
    print("逐查询结果:")
    print(f"{'查询':<12s} {'面数':>4s} {'自身名次':>8s} {'自身分':>10s}")
    for r in summary["per_query"]:
        print(f"{r['query_name']:<12s} {r['n_query_faces']:>4d} "
              f"{r['self_rank']:>8d} {r['self_score']:>10.6f}")

    print("\n" + "─" * 64)
    print("总体结果:")
    print(f"  查询总数         : {summary['n_queries']}")
    print(f"  自身 Top-1 命中   : {summary['self_top1_hits']}/{summary['n_queries']}")
    print(f"  自身 Top-1 命中率 : {summary['self_top1_rate']*100:.2f}%")
    print(f"  平均自身排名     : {summary['mean_self_rank']:.3f}")
    print(f"  自身得分全为 0   : {summary['all_zero_score']}")

    results_dir = export_results(summary, args.output_dir)
    print(f"\n结果已导出 → {results_dir}")
    print("=" * 64)
    return summary


def cmd_evaluate(args):
    print("=" * 64)
    print("BRepNet 自身检索 — 真实 embeddings 评估")
    print("=" * 64)
    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.is_dir():
        sys.exit(f"错误: embeddings 目录不存在: {emb_dir}")

    files = sorted(emb_dir.glob("*.embeddings")) or sorted(emb_dir.glob("*.txt"))
    if not files:
        sys.exit(f"错误: 在 {emb_dir} 中未找到 .embeddings / .txt 文件")

    print(f"发现 {len(files)} 个 embeddings 文件")
    embeddings_map = {}
    for fp in files:
        name = fp.stem
        arr = load_embeddings_file(fp)
        embeddings_map[name] = arr
        print(f"  {name:<24s} {arr.shape[0]} faces × {arr.shape[1]} dims")

    summary = run_all_queries(embeddings_map, faces=args.faces, top_k=args.top_k)

    print("\n" + "─" * 64)
    print("总体结果:")
    print(f"  查询总数         : {summary['n_queries']}")
    print(f"  自身 Top-1 命中   : {summary['self_top1_hits']}/{summary['n_queries']}")
    print(f"  自身 Top-1 命中率 : {summary['self_top1_rate']*100:.2f}%")
    print(f"  平均自身排名     : {summary['mean_self_rank']:.3f}")
    print(f"  自身得分全为 0   : {summary['all_zero_score']}")

    # 并列/异常诊断
    nonzero = [r for r in summary["per_query"] if abs(r["self_score"]) > 1e-9]
    if nonzero:
        print(f"\n  ⚠ 发现 {len(nonzero)} 个自身得分非零的查询（文件错配/数值变化）:")
        for r in nonzero:
            print(f"    {r['query_name']}: self_score={r['self_score']:.6f}")

    results_dir = export_results(summary, args.output_dir)
    print(f"\n结果已导出 → {results_dir}")
    print("=" * 64)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="BRepNet 模型自身检索实验的独立检索层"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="合成数据演示自身 Top-1")
    p_demo.add_argument("--output-dir", default="experiment_runs/synthetic_demo")
    p_demo.add_argument("--top-k", type=int, default=5)
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="对真实 .embeddings 目录做全库自身检索")
    p_eval.add_argument("--embeddings-dir", required=True)
    p_eval.add_argument("--output-dir", default="real_results")
    p_eval.add_argument("--faces", default="all")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
