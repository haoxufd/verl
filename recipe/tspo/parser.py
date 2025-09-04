import re
import json
import os

def extract_tree_data(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 正则匹配 const treeData = {...};
    match = re.search(r'const\s+treeData\s*=\s*(\{.*?meta_info.*?\});', html, re.S)
    if not match:
        raise ValueError("未找到 treeData 结构")

    js_code = match.group(1).rstrip(';').strip()

    # 尝试严格 JSON 解析
    try:
        return json.loads(js_code)
    except json.JSONDecodeError:
        # 如果失败，尝试宽松解析
        try:
            import demjson3 as demjson
            print(js_code)
            print(html_path)
            return demjson.decode(js_code)
        except ImportError:
            import js2py
            context = js2py.EvalJs()
            context.execute("var treeData = " + js_code)
            return context.treeData.to_dict()

def calc_first_level_leaf_rate(tree_data):
    first_level_nodes = tree_data["tree_data"]["children"]
    num_leaf = 0

    for node in first_level_nodes:
        if node["children"] == []:
            num_leaf += 1
    
    return num_leaf / len(first_level_nodes)

def calc_fllr_of_step(step_dir):
    html_files = [os.path.join(step_dir, f) for f in os.listdir(step_dir)]
    tree_data = [extract_tree_data(html_file) for html_file in html_files]
    fllr = [calc_first_level_leaf_rate(data) for data in tree_data]

    return sum(fllr) / len(fllr)

def calc_fllr_of_exp(exp_dir):
    step_dirs = os.listdir(exp_dir)
    fllr = [calc_fllr_of_step(os.path.join(exp_dir, step_dir)) for step_dir in step_dirs]
    fllr = [round(value, 4) for value in fllr]

    return fllr


# 🔹 使用示例
if __name__ == "__main__":
    fllr_1 = calc_fllr_of_exp("/user/hxu4/u16813/sampling_tree/TSPOvsDAPO-Qwen3-4B-MATH-17K/TSPO-4x2x2x2-Smt")
    fllr_2 = calc_fllr_of_exp("/user/hxu4/u16813/sampling_tree/TSPOvsDAPO-Qwen3-4B-MATH-17K/TSPO-4x2x2x2-Smt-StepReward")
    print(fllr_1)
    print(fllr_2)