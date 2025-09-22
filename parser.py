import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from recipe.tspo.sampling_tree import SamplingTree
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict


def _plot_hist(data: list[float], title: str, bin_width: float, show_freq: bool, save_path: str):
    plt.figure()

    # 自动生成区间范围
    min_val, max_val = min(data), max(data)
    bins = np.arange(np.floor(min_val/bin_width) * bin_width,
                     np.ceil(max_val/bin_width) * bin_width + bin_width,
                     bin_width)

    # 绘制直方图
    counts, edges, _ = plt.hist(data, bins=bins.tolist(), edgecolor='black')

    plt.title(title)
    plt.xlabel("Range")
    plt.ylabel("Frequency")

    # 在柱子上标注频次
    if show_freq:
        for count, edge in zip(counts, edges):
            if count > 0:
                plt.text(edge + bin_width/2, count, str(int(count)),
                         ha='center', va='bottom', fontsize=9)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 高分辨率保存
    print(f"Image has been saved to: {save_path}")

    counts = counts.tolist()
    edges = edges.tolist()
    total = sum(counts)

    res = {}
    for i in range(len(edges) - 1):
        res[str((edges[i], edges[i + 1]))] = (counts[i], round(counts[i] / total, 4))
    res["avg"] = round(sum(data) / len(data), 4)

    # Save result to a json file
    json_file = save_path.replace(".png", ".json")
    with open(json_file, 'w') as f:
        json.dump(res, f, indent=2)

    plt.close()


def _plot_two_hists_in_one_figure(
        data1: list[float], data2: list[float], 
        labels: tuple[str, str],
        title: str,
        bin_width: float,
        save_path: str):
    
    plt.figure()

    # 自动生成区间范围（覆盖两组数据）
    min_val = min(min(data1), min(data2))
    max_val = max(max(data1), max(data2))
    bins = np.arange(np.floor(min_val/bin_width) * bin_width,
                     np.ceil(max_val/bin_width) * bin_width + bin_width,
                     bin_width)

    # 绘制直方图，两组数据透明度不同以便重叠显示
    counts1, edges1, _ = plt.hist(data1, bins=bins.tolist(), edgecolor='black', 
                             alpha=0.6, label=labels[0])
    counts2, edges2, _ = plt.hist(data2, bins=bins.tolist(), edgecolor='black', 
                             alpha=0.6, label=labels[1])

    plt.title(title)
    plt.xlabel("Range")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 高分辨率保存
    print(f"Image has been saved to: {save_path}")

    plt.close()


class SolutionStats:
    """存储 correct/incorrect 的统计结果"""
    def __init__(self):
        self.correct = defaultdict(list)
        self.incorrect = defaultdict(list)
        self.all = defaultdict(list)

    def add(self, node, stats):
        target = self.correct if node.acc else self.incorrect
        for k, v in stats.items():
            target[k].append(v)
            self.all[k].append(v)
    
    # ---------- 序列化 ----------
    def to_dict(self):
        return {
            "correct": dict(self.correct),
            "incorrect": dict(self.incorrect),
            "all": dict(self.all),
        }

    def to_json(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ---------- 反序列化 ----------
    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.correct.update(data.get("correct", {}))
        obj.incorrect.update(data.get("incorrect", {}))
        obj.all.update(data.get("all", {}))
        return obj

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self):
        return (f"SolutionStats(correct_keys={list(self.correct.keys())}, "
                f"incorrect_keys={list(self.incorrect.keys())})")

    def plot_hist(self, save_dir):
        keys = self.all.keys()
        for key in keys:
            all_data = self.all[key]
            correct_data = self.correct[key]
            incorrect_data = self.incorrect[key]

            _plot_hist(all_data, str.capitalize(f"{key}"), bin_width=0.1, show_freq=False,
                      save_path=os.path.join(save_dir, f"{key}.png"))
            
            _plot_two_hists_in_one_figure(correct_data, incorrect_data,
                                          labels=("Correct", "Incorrect"),
                                          title=str.capitalize(f"{key}") + " (Correct vs Incorrect)",
                                          bin_width=0.1,
                                          save_path=os.path.join(save_dir, f"{key}_contrast.png"))
            

def calc_solution_attributes_of_tree(tree_json_file):
    tree = SamplingTree.from_json(tree_json_file)
    assert tree.leaf_nodes is not None

    stats_container = SolutionStats()

    def compute_stats(path):
        """计算路径上的统计值"""
        path_entropy = [e for nd in path for e in nd.entropy]
        path_log_prob = [lp for nd in path for lp in nd.log_prob]

        step_entropy = [sum(nd.entropy) / len(nd.entropy) for nd in path]
        step_ppl = [math.exp(-sum(nd.log_prob) / len(nd.log_prob)) for nd in path]

        return {
            "entropy": round(sum(path_entropy) / len(path_entropy), 4),
            "ppl": round(math.exp(-sum(path_log_prob) / len(path_log_prob)), 4),
            "mean_step_entropy": round(sum(step_entropy) / len(step_entropy), 4),
            "max_step_entropy": round(max(step_entropy), 4),
            "mean_step_ppl": round(sum(step_ppl) / len(step_ppl), 4),
            "max_step_ppl": round(max(step_ppl), 4),
        }

    for node in tree.leaf_nodes:
        stats = compute_stats(node.get_ancestors())
        stats_container.add(node, stats)

    return stats_container


def calc_solution_attributes_of_gen_step(gen_step_dir):
    files = os.listdir(gen_step_dir)
    json_files = [file for file in files if file.endswith("json") and file.startswith("tree")]
    tree_json_files = [os.path.join(gen_step_dir, json_file) for json_file in json_files]

    for tree_json_file in tree_json_files:
        print(f"Processing {tree_json_file}...")
        stats_container = calc_solution_attributes_of_tree(tree_json_file)

        # 合并 correct/incorrect 结果
        if 'all_stats' not in locals():
            all_stats = stats_container
        else:
            for k in stats_container.correct:
                all_stats.correct[k].extend(stats_container.correct[k])
            for k in stats_container.incorrect:
                all_stats.incorrect[k].extend(stats_container.incorrect[k])
            for k in stats_container.all:
                all_stats.all[k].extend(stats_container.all[k])
    
    # 保存结果到文件
    stats_file = os.path.join(gen_step_dir, "solution_stats.json")
    all_stats.to_json(stats_file)
    print(f"Saved solution stats to {stats_file}")

    all_stats.plot_hist(gen_step_dir)


def calc_problem_attributes_of_tree(tree_json_file):
    tree = SamplingTree.from_json(tree_json_file)
    
    assert tree.leaf_nodes is not None

    entropys = []
    log_probs = []

    for node in tree.all_nodes:
        if node.parent is None:
            continue
        assert node.entropy is not None and node.log_prob is not None
        entropys.extend(node.entropy)
        log_probs.extend(node.log_prob)
    
    avg_entropy = sum(entropys) / len(entropys)
    perplexity = math.exp(-sum(log_probs) / len(log_probs))

    correct_leaf_nodes = [node for node in tree.leaf_nodes if node.acc]
    acc = len(correct_leaf_nodes) / len(tree.leaf_nodes)

    return avg_entropy, perplexity, acc


def calc_problem_attributes_of_step(gen_step_dir):
    files = os.listdir(gen_step_dir)
    json_files = [file for file in files if file.endswith("json") and file.startswith("tree")]
    tree_json_files = [os.path.join(gen_step_dir, json_file) for json_file in json_files]

    avg_entropys = []
    perplexities = []
    accs = []
    for tree_json_file in tree_json_files:
        avg_entropy, perplexity, acc = calc_problem_attributes_of_tree(tree_json_file)
        avg_entropys.append(avg_entropy)
        perplexities.append(perplexity)
        accs.append(acc)
    
    avg_entropys = [round(avg_entropy, 4) for avg_entropy in avg_entropys]
    perplexities = [round(perplexity, 4) for perplexity in perplexities]
    accs = [round(acc, 4) for acc in accs]

    avg_entropys_file = os.path.join(gen_step_dir, "problem_entropys.json")
    with open(avg_entropys_file, 'w') as f:
        json.dump(avg_entropys, f, indent=2)
    print(f"Saved avg entropys file to {avg_entropys_file}")

    perplexities_file = os.path.join(gen_step_dir, "problem_perplexities.json")
    with open(perplexities_file, 'w') as f:
        json.dump(perplexities, f, indent=2)
    print(f"Saved perplexities file to {perplexities_file}")

    accs_file = os.path.join(gen_step_dir, "problem_accs.json")
    with open(accs_file, 'w') as f:
        json.dump(accs, f, indent=2)
    print(f"Saved accs file to {accs_file}")


def show_high_entropy_tokens_of_solution(data: dict, top_k: int, save_path: str | None = None):
    indices = np.argsort(data["entropy"])[-top_k:][::-1]

    res = []
    for idx in indices:
        context_start = idx - 10 if idx - 10 >= 0 else 0
        context_end = idx + 10 if idx + 10 <= len(data["output"]) else len(data["output"])
        context = data["output"][context_start:context_end]
        ctxt_entropy = [round(float(x), 4) for x in data["entropy"][context_start:context_end]]
        res.append({
            "index": int(idx),
            "token": data["output"][idx],
            "entropy": round(float(data["entropy"][idx]), 4),
            "log_prob": round(float(data["log_prob"][idx]), 4),
            "context": context,
            "ctxt_entropy": ctxt_entropy,
            "candidates": data["candidate_tokens"][idx],
            "candidate_log_probs": [round(float(lp), 4) for lp in data["candidate_log_probs"][idx]],
        })
    
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"High entropy tokens have been saved to: {save_path}")

    return res



if __name__ == "__main__":
    with open("/user/hxu4/u16814/rollout/COLLECT_DATA/DAPO-32/1_top10.jsonl", 'r') as f:
        lines = json.load(f)
    
    print(lines[0]["input"])
    print("".join(lines[0]["output"]))