import re
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_tree_data(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    match = re.search(r'const\s+treeData\s*=\s*(\{.*?meta_info.*?\});', html, re.S)
    if not match:
        raise ValueError("未找到 treeData 结构")

    js_code = match.group(1).rstrip(';').strip()

    try:
        return json.loads(js_code)
    except json.JSONDecodeError:
        try:
            import demjson3 as demjson
            return demjson.decode(js_code)
        except ImportError:
            import js2py
            context = js2py.EvalJs()
            context.execute("var treeData = " + js_code)
            return context.treeData.to_dict()

def calc_first_level_leaf_rate(tree_data):
    first_level_nodes = tree_data["tree_data"]["children"]
    num_leaf = sum(1 for node in first_level_nodes if node["children"] == [])
    return num_leaf / len(first_level_nodes) if first_level_nodes else 0

def calc_fllr_of_step(step_dir):
    html_files = [os.path.join(step_dir, f) for f in os.listdir(step_dir) if f.endswith(".html")]
    tree_data = [extract_tree_data(html_file) for html_file in html_files]
    fllr = [calc_first_level_leaf_rate(data) for data in tree_data]
    return sum(fllr) / len(fllr) if fllr else 0

def calc_fllr_of_exp(exp_dir, scope="all", max_workers=None):
    """
    scope: "all" | "10-20"
    max_workers: 并行进程数，默认 None -> 使用 CPU 核心数
    """
    step_dirs = sorted(os.listdir(exp_dir))  # 确保顺序一致
    if '-' in scope:
        start, end = [int(x) for x in scope.split('-')]
        step_dirs = step_dirs[start:end]

    step_paths = [os.path.join(exp_dir, step_dir) for step_dir in step_dirs]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_step = {executor.submit(calc_fllr_of_step, step): step for step in step_paths}
        for future in as_completed(future_to_step):
            step = future_to_step[future]
            try:
                result = round(future.result(), 4)
            except Exception as e:
                print(f"Error in {step}: {e}")
                result = 0
            results.append((step, result))

    # 按 step_dirs 原始顺序返回
    results.sort(key=lambda x: step_dirs.index(os.path.basename(x[0])))
    return [r[1] for r in results]


if __name__ == "__main__":
    fllr_1 = calc_fllr_of_exp(
        "/user/hxu4/u16813/sampling_tree/TSPOvsDAPO-Qwen3-4B-MATH-17K/TSPO-4x2x2x2-Smt-StepReward",
        "40-80",
        max_workers=40  # 可手动设定，比如服务器 16 核
    )
    print(fllr_1)
