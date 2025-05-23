import torch
import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.gsm8k_step_seperate import extract_solution as extract_solution_gsm8k

from graphviz import Digraph

import json
import os
from pathlib import Path
from typing import Dict, Any

class ChildrenFullError(Exception):
    """
    Custom exception to indicate that a node's children are full.
    
    Attributes:
        message (str): Error message
    """
    
    def __init__(self, message="Node's children are full."):
        self.message = message
        super().__init__(self.message)


class Node:
    """
    Tree node class that can be used to represent a node in an n-ary tree.
    
    Attributes:
        children (list[Node]): References to all child nodes
        parent (Node): Reference to the parent node
        content (str): The question or problem-solving step corresponding to this node
        children_is_full (bool): Indicates whether the node's children are full (reached n)
        is_leaf (bool): Indicates whether the node is a leaf node
    """
    
    def __init__(self, 
                 token_sequence: list[int], 
                 max_children: int, 
                 parent=None, 
                 children=None, 
                 children_is_full=False, 
                 is_root=False,
                 is_leaf=True,
                 text=None,
                 score=None,
                 depth=0) -> None:
        """
        Initialize a tree node.
        
        Args:
            parent (Node): Reference to parent node, default is None
        """
        self.children = [] if children is None else children
        self.parent = parent
        self.token_sequence = token_sequence
        self.children_is_full = children_is_full
        self.is_root = is_root
        self.is_leaf = is_leaf
        self.max_children = max_children
        self.text = text
        self.score = score
        self.advantage = None
        self.depth = depth
    
    def add_child(self, child_node):
        """
        Add a child node to the current node.
        
        Args:
            child_node (Node): The child node to add
            
        Returns:
            bool: True if the child was added successfully, False otherwise
        """
        # Check if the node can accept more children
        if self.max_children is not None and len(self.children) >= self.max_children:
            raise ChildrenFullError(f"Cannot add child to node since children are full.")
        
        # Add the child
        self.children.append(child_node)
        child_node.parent = self

        # Check if self's children are full
        if len(self.children) == self.max_children:
            self.children_is_full = True
        
        # No longer a leaf node once we have children
        self.is_leaf = False

        child_node.depth = self.depth + 1
    
    def get_ancestors(self):
        """
        Get all ancestors of the current node.
        
        Returns:
            list[Node]: List of ancestor nodes
        """
        ancestors = []
        current_node = self
        while current_node.parent is not None:
            ancestors.append(current_node.parent)
            current_node = current_node.parent
        ancestors.reverse()
        return ancestors


class SamplingTree:
    """
    Sampling tree.
    """

    def __init__(self, root_node, order, max_prompt_length, max_response_length, data_source, ability, reward_model, index, extra_info, tokenizer) -> None:
        self.root = root_node
        self.order = order
        self.extra_info = extra_info
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.data_source = data_source
        self.ability = ability
        self.reward_model = reward_model
        self.index = index
        self.all_nodes = [root_node]
        self.final_answer = reward_model["ground_truth"]
    
    
    def add_node(self, node1: Node, node2: Node):
        """
        Add a node to the tree.
        
        Args:
            node1 (Node): The parent node
            node2 (Node): The child node to add
        """

        node1.add_child(node2)
        self.all_nodes.append(node2)
    
    def add_path(self, node: Node, path: list[Node]):
        """
        Add a path to the tree.
        
        Args:
            path (list[Node]): The path to add
        """
        
        parent_node = node
        for nd in path:
            self.add_node(parent_node, nd)
            parent_node = nd
    
    def add_paths(self, node: Node, paths: list[list[Node]]):
        """
        Add multiple paths to the tree.
        
        Args:
            paths (list[list[Node]]): The paths to add
        """
        for path in paths:
            self.add_path(node, path)
    
    def compute_scores(self):
        """
        Compute scores for all nodes in the tree.
        """
        compute_scores(self.root, self.final_answer)
        
    def compute_advantages(self):
        """
        Compute advantages for all nodes in the tree.
        """
        compute_advantages(self.root)
    
    def assemble_tensor_data(self, input_nodes, output_nodes):
        """
        Assemble tensor data for the input and output nodes.
        
        Args:
            input_nodes (list[Node]): The input nodes
            output_nodes (list[Node]): The output nodes
            
        Returns:
            tuple: A tuple containing the input and output tensors
        """
        response = []
        response_mask = []
        prompt = []

        for data in input_nodes:
            prompt.extend(data.token_sequence)
        prompt_length = len(prompt)
        prompt = prompt[:self.max_prompt_length] if len(prompt) > self.max_prompt_length else [self.tokenizer.pad_token_id] * (self.max_prompt_length - len(prompt)) + prompt
        prompt = torch.tensor(prompt, dtype=torch.int64)

        for node in output_nodes:
            response.extend(node.token_sequence)
        response_length = len(response)
        response = response[:self.max_response_length] if len(response) > self.max_response_length else response + [self.tokenizer.pad_token_id] * (self.max_response_length - len(response))
        response = torch.tensor(response, dtype=torch.int64)

        response_mask = torch.zeros(self.max_response_length, dtype=torch.int64)
        response_mask[:response_length] = 1

        input_id = torch.cat([prompt, response], dim=0)
        attention_mask = torch.zeros(self.max_prompt_length, dtype=torch.int64)
        attention_mask[-prompt_length:] = 1
        attention_mask = torch.cat([attention_mask, response_mask], dim=0)

        position_id = get_position_ids_from_attention_mask(attention_mask.reshape(1, -1))
        position_id = position_id.reshape(-1)

        advantage = []
        for node in output_nodes:
            advantage.extend([node.advantage] * len(node.token_sequence))
        advantage = advantage[:self.max_response_length] if len(advantage) > self.max_response_length else advantage + [0.0] * (self.max_response_length - len(advantage))
        advantage = torch.tensor(advantage, dtype=torch.float32)

        score = []
        for node in output_nodes:
            score.extend([node.score] * len(node.token_sequence))
        score = score[:self.max_response_length] if len(score) > self.max_response_length else score + [0.0] * (self.max_response_length - len(score))
        score = torch.tensor(score, dtype=torch.float32)

        reward = score

        return {
            "input_id": input_id,
            "attention_mask": attention_mask,
            "position_id": position_id,
            "prompt": prompt,
            "response": response,
            "response_mask": response_mask,
            "advantage": advantage,
            "return": advantage,
            "score": score,
            "reward": reward
        }
  
    def collect_batch_data(self):
        """
        Collect batch data from the tree.
        """
        leaf_nodes = [node for node in self.all_nodes if node.is_leaf]

        visited = {}
        for node in self.all_nodes:
            visited[node] = False
        visited[self.root] = True

        tensor_data = {
            "input_ids": [], 
            "attention_mask": [], 
            "position_ids": [], 
            "prompts": [], 
            "responses": [], 
            "response_mask": [], 
            "advantages": [], 
            "returns": [],
            "token_level_scores": [],
            "token_level_rewards": []}
        batch_key_to_piece_key = {
            "input_ids": "input_id",
            "attention_mask": "attention_mask",
            "position_ids": "position_id",
            "prompts": "prompt",
            "responses": "response",
            "response_mask": "response_mask",
            "advantages": "advantage",
            "returns": "return",
            "token_level_scores": "score",
            "token_level_rewards": "reward"
        }

        for node in leaf_nodes:
            path = node.get_ancestors() + [node]
            
            # Find the first unvisited node in the path
            pos = 0
            while pos < len(path) and visited[path[pos]]:
                pos += 1
            
            assert pos < len(path)

            input_nodes = path[:pos]
            output_nodes = path[pos:]

            single_tensor_data = self.assemble_tensor_data(input_nodes, output_nodes)
            for key in tensor_data.keys():
                tensor_data[key].append(single_tensor_data[batch_key_to_piece_key[key]])
            
            for nd in path[pos:]:
                visited[nd] = True
        
        tensor_batch_dict = {}
        for key in tensor_data.keys():
            tensor_batch_dict[key] = torch.stack(tensor_data[key], dim=0)
        
        num_data = len(leaf_nodes)
        non_tensor_batch_dict = {
            "data_source": np.array([self.data_source] * num_data),
            "ability": np.array([self.ability] * num_data),
            "index": np.array([self.index] * num_data),
            "reward_model": np.array([self.reward_model] * num_data),
            "extra_info": np.array([self.extra_info] * num_data)
        }
        
        return DataProto.from_single_dict({**tensor_batch_dict, **non_tensor_batch_dict})
    
    def visualize(self, output_file="tree_visualization.html", auto_open=True):
        """
        生成交互式HTML树形可视化
        
        Args:
            output_file (str): 输出HTML文件名
            auto_open (bool): 是否自动在浏览器中打开
        """
        # 1. 将树转换为JSON格式
        tree_data = self._tree_to_json()
        
        # 2. 生成HTML内容
        html_content = self._generate_html_from_template(tree_data)
        
        # 3. 保存为HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"可视化文件已保存为: {output_file}")
        
    
    def _tree_to_json(self) -> Dict[str, Any]:
        """将树结构转换为D3.js友好的JSON格式"""
        
        def node_to_dict(node, node_id=0):
            # 处理文本显示（截断过长文本）
            display_text = ""
            full_text = node.text or ""
            if full_text:
                display_text = full_text[:50] + "..." if len(full_text) > 50 else full_text
            
            # 处理token序列显示
            token_str = str(node.token_sequence)
            display_tokens = token_str[:30] + "..." if len(token_str) > 30 else token_str
            
            node_data = {
                "id": node_id,
                "name": f"Node {node.depth}" if not node.is_root else "Root",
                "score": round(node.score, 4) if node.score is not None else "N/A",
                "advantage": round(node.advantage, 4) if node.advantage is not None else "N/A",
                "text": display_text,
                "full_text": full_text,
                "token_sequence": node.token_sequence,
                "display_tokens": display_tokens,
                "depth": node.depth,
                "is_root": node.is_root,
                "is_leaf": node.is_leaf,
                "children_count": len(node.children),
                "children": []
            }
            
            # 递归处理子节点
            child_id = node_id + 1
            for child in node.children:
                child_data, child_id = node_to_dict(child, child_id)
                node_data["children"].append(child_data)
            
            return node_data, child_id
        
        root_data, _ = node_to_dict(self.root)
        
        # 添加树的元信息
        tree_info = {
            "tree_data": root_data,
            "meta_info": {
                "final_answer": getattr(self, 'final_answer', 'N/A'),
                "total_nodes": len(self.all_nodes),
                "max_depth": max(node.depth for node in self.all_nodes),
                "order": getattr(self, 'order', 'N/A')
            }
        }
        
        return tree_info
    
    def _generate_html_from_template(self, tree_data: Dict[str, Any]) -> str:
        """从模板文件生成HTML内容"""
        # 获取模板文件路径
        template_dir = Path(__file__).parent / "templates"
        template_path = template_dir / "tree_visualization.html"
        
        # 如果模板文件不存在，提示用户
        if not template_path.exists():
            raise FileNotFoundError(
                f"模板文件未找到: {template_path}\n"
                f"请确保 templates/tree_visualization.html 文件存在"
            )
        
        # 读取模板文件
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # 替换数据占位符
        tree_json = json.dumps(tree_data, ensure_ascii=False, indent=2)
        html_content = template.replace('{tree_json}', tree_json)
        
        return html_content
    

def print_tree(node, level=0):
    """
    Recursively print the tree starting from the given node.
    
    Args:
        node (Node): The current node to print from. If None, starts from root.
        level (int): The current depth level in the tree.
    """
    indent = "  " * level
    print(f"{indent}- score: {node.score}, advantage: {node.advantage}")
    print(f"{indent}  text: {repr(node.text)}")
    print(f"{indent}  token_sequence: {node.token_sequence}")
    
    for child in getattr(node, "children", []):
        print_tree(child, level + 1)


def get_position_ids_from_attention_mask(attention_mask):
    """
    Convert attention_mask to position_ids where position ids start from 0 at the 
    first non-padding token, and increment continuously across the sequence.
    
    Args:
        attention_mask: An int64 tensor of shape [batch_size, seq_len],
                        with values 0 (padding) or 1 (valid tokens)
                        
    Returns:
        position_ids: An int64 tensor of shape [batch_size, seq_len],
                      starting from 0 at the first 1 in the mask, and incrementing continuously
    """
    batch_size, seq_len = attention_mask.shape
    position_ids = torch.zeros_like(attention_mask)

    for i in range(batch_size):
        # Find the first index where attention_mask == 1
        first_valid = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(first_valid) > 0:
            start_idx = first_valid[0].item()
            # Create a range starting from 0
            position_ids[i, start_idx:] = torch.arange(seq_len - start_idx, dtype=attention_mask.dtype)
    
    return position_ids


def extract_raw_responses(responses, response_mask):
    """
    Extract elements from responses where the corresponding mask value is 1.
    
    Args:
        responses: Tensor of shape (bz, length) containing response values
        response_mask: Tensor of shape (bz, length) containing mask values (0 or 1)
    
    Returns:
        raw_responses: List of lists, where each inner list contains integers from responses
                      at positions where response_mask is 1
    """
    # Convert tensors to CPU and numpy if they're on GPU
    if isinstance(responses, torch.Tensor):
        responses = responses.cpu().numpy()
        response_mask = response_mask.cpu().numpy()
    
    batch_size = responses.shape[0]
    raw_responses = []
    
    for i in range(batch_size):
        # Extract values where mask is 1
        valid_indices = response_mask[i] == 1
        raw_response = responses[i][valid_indices].tolist()
        raw_responses.append(raw_response)
    
    return raw_responses


def compute_response_mask(responses, attention_mask):
    response_length = responses.size(1)

    return attention_mask[:, -response_length:]


def split_list_by_positions(lst, positions):
    """
    Split a list into sublists based on specified positions.
    
    Args:
        lst: The list to be split
        positions: A list of indices where the splits should occur
    
    Returns:
        A list of sublists
    """
    all_positions = positions + [len(lst) - 1]
    result = []
    start = 0

    for end in all_positions:
        result.append(lst[start : end + 1])
        start = end + 1

    if result[-1] == []:
        result.pop()
    
    return result


def distribute_evenly(total, parts):
    """
    Distributes a total number into parts as evenly as possible.
    
    Args:
        total (int): The total number to be distributed
        parts (int): Number of parts to distribute into
        
    Returns:
        list: A list of length 'parts' containing the distribution values
        
    Examples:
        >>> distribute_evenly(7, 3)
        [3, 2, 2]
        >>> distribute_evenly(7, 4)
        [2, 2, 2, 1]
    """
    if parts <= 0:
        raise ValueError("Number of parts must be greater than 0")
    
    if total < 0:
        raise ValueError("Total value cannot be negative")
        
    # Calculate the average value per part
    avg = total / parts
    
    # Get the base value by flooring the average
    base = int(avg)
    
    # Calculate how many parts need to be base+1
    remainder = total - base * parts
    
    # Build the result array
    result = []
    for i in range(parts):
        if i < remainder:
            result.append(base + 1)  # First 'remainder' parts get base+1
        else:
            result.append(base)      # Remaining parts get base
            
    return result


def find_step_split_token_positions(input_ids, tokenizer, step_split_str="\n\n"):
    text = tokenizer.decode(input_ids)
    encoding = tokenizer(text, return_offsets_mapping=True)
    
    split_str_len = len(step_split_str)
    step_split_str_end_positions = [i + split_str_len for i in range(len(text) - split_str_len) if text[i : i + split_str_len] == step_split_str]
    
    newline_token_positions = []
    start_idx = 0
    for end_pos in step_split_str_end_positions:
        for i in range(start_idx, len(encoding.offset_mapping)):
            if encoding.offset_mapping[i][1] == end_pos:
                newline_token_positions.append(i)
                start_idx = i + 1
                break
            elif encoding.offset_mapping[i][0] > end_pos:
                start_idx = i
                break
    
    return newline_token_positions


def convert_split_positions(split_positions, num_parts):
    """
    positions: a list of indices where the split token ids should occur.
    num_parts: the number of parts to split into.
    """
    total = len(split_positions) + 1
    if num_parts == 1:
        return []
    if num_parts >= total:
        return split_positions
    
    parts = distribute_evenly(total, num_parts)
    picked_positions = []
    for i in range(num_parts - 1):
        picked_positions.append(split_positions[sum(parts[:i + 1]) - 1])
    
    return picked_positions


def build_sampling_trees(batch_dict, actor_rollout_wg, tokenizer, config):
    start_input_ids = batch_dict["raw_prompt_ids"]
    order = config.actor_rollout_ref.rollout.order
    batch_size = len(start_input_ids)
    start_texts = tokenizer.batch_decode(start_input_ids, skip_special_tokens=False)
    root_nodes = [Node(token_sequence=start_input_ids[i], max_children=order, is_root=True, text=start_texts[i]) for i in range(batch_size)]
    sampling_trees = [
        SamplingTree(
            root_nodes[i], 
            order,
            config.data.max_prompt_length,
            config.data.max_response_length,
            batch_dict["data_source"][i],
            batch_dict["ability"][i],
            batch_dict["reward_model"][i],
            batch_dict["index"][i],
            batch_dict["extra_info"][i],
            tokenizer) for i in range(batch_size)
    ]

    while True:
        input_ids = []
        target_node_to_tree = {}
        all_target_nodes = []

        num_to_be_added_paths = order - 1
        for i, tree in enumerate(sampling_trees):
            # Collect input_ids for generation from all trees
            # For a certain tree, the input_ids are token sequences from the nodes whose children are not full
            # For example, if order is 2 and the tree is
            #                              0
            #                            /    \
            #                           1      2
            #                          /
            #                         3
            # token sequence of Node 0 is [1, 3, 5] and that of Node 1 is [2, 4]
            # So the input_ids for generation are [[1, 3, 5, 2, 4]]
            # Suppose -1 is a special token to seperate steps. If the generation result is [6, 7, -1, 8, 9], 
            # then there will be two new nodes 4 and 5, whose token sequences are [6, 7] and [8, 9] respectively,
            # as shown below:
            #                              0
            #                            /    \
            #                           1      2
            #                         /   \
            #                        3     4
            #                            /
            #                           5
            # The new nodes are added to the tree and the input_ids for generation are updated in next iteration.

            # Find all non-leaf nodes of tree whose children are not full
            target_nodes = [node for node in tree.all_nodes if (node.is_root or not node.is_leaf) and not node.children_is_full]
            all_target_nodes.extend(target_nodes)
            if target_nodes and target_nodes[0].is_root:
                num_to_be_added_paths = order
            
            for target_node in target_nodes:
                target_node_to_tree[target_node] = tree

                # Concatenate the token sequences in the path from root to target_node
                path = target_node.get_ancestors() + [target_node]
                seq = []
                for nd in path:
                    seq.extend(nd.token_sequence)
                input_ids.extend([seq for _ in range(num_to_be_added_paths)])
        
        # If no input_ids are generated, break the loop
        if not all_target_nodes:
            break
        
        batch_size = len(input_ids)
        # Create ndarray of raw_prompt_ids
        raw_prompt_ids = np.array(input_ids, dtype='O')
        # Create tensors of input_ids, attention_mask, and position_ids
        input_ids = [[tokenizer.pad_token_id] * (config.data.max_prompt_length - len(seq)) + seq for seq in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int64)
        position_ids = get_position_ids_from_attention_mask(attention_mask)
        
        gen_batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids
        }
        gen_batch = DataProto.from_single_dict(gen_batch_dict, auto_padding=True)

        gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)
        responses = gen_batch_output.batch["responses"][: batch_size, :]
        attention_mask = gen_batch_output.batch["attention_mask"][: batch_size, :]
        response_mask = compute_response_mask(responses, attention_mask)
        raw_responses = extract_raw_responses(responses, response_mask)

        paths = []
        config.data.step_split_str = "\n\n"
        depths= [node.depth for node in all_target_nodes]
        for idx, raw_response in enumerate(raw_responses):
            positions = find_step_split_token_positions(raw_response, tokenizer, config.data.step_split_str)
            # Remove the last position since it is "\n\n<answer>VALUE<answer>"
            positions = convert_split_positions(positions[:-1], config.actor_rollout_ref.rollout.max_tree_depth - depths[idx // num_to_be_added_paths])
            sequences = split_list_by_positions(raw_response, positions)
            texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]
            path = [Node(token_sequence=sequences[i], max_children=order, text=texts[i]) for i in range(len(sequences))]
            paths.append(path)
        for i, target_node in enumerate(all_target_nodes):
            tree = target_node_to_tree[target_node]
            paths_to_be_added = paths[i * (order - 1) : (i + 1) * (order - 1)] if not target_node.is_root else paths[i * order : (i + 1) * order]
            tree.add_paths(target_node, paths_to_be_added)
    
    return sampling_trees


def compute_scores(node, final_answer):
    if node.is_leaf:
        node.score = 1.0 if extract_solution_gsm8k(node.text) == final_answer else 0.0
        return 1, node.score
    
    # Num of leaf nodes for each subtree
    leaf_node_nums = []
    # Total score of all leaf nodes for each subtree
    scores = []
    for child in node.children:
        leaf_num, score = compute_scores(child, final_answer)
        leaf_node_nums.append(leaf_num)
        scores.append(score)
    
    node.score = sum(scores) / sum(leaf_node_nums)
    
    return sum(leaf_node_nums), sum(scores)


def compute_advantages(node: Node):
    if node.is_leaf:
        return
    
    assert node.children_is_full

    scores = np.array([_node.score for _node in node.children])
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    advantages = ((scores - mean) / (std + 1e-6)).tolist()

    for i, child in enumerate(node.children):
        child.advantage = advantages[i]
    
    for child in node.children:
        compute_advantages(child)