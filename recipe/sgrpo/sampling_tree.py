import torch
import numpy as np
import torch
from typing import Union, List

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import json
from pathlib import Path
from typing import Dict, Any

import os

from typing import List, Tuple
import torch
import numpy as np

from pprint import pprint
import numpy as np
import torch
from typing import Union, List

from verl import DataProto
import os

from dataclasses import dataclass
from typing import Generator, List, Tuple
from typing import List, Tuple


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
        token_sequence (list[int]): The token sequence of a question or a problem-solving step corresponding to this node
        text (str): The result of decoding the token_sequence
        children_is_full (bool): Indicates whether the node's children are full (reached max_children)
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
                 depth=0,
                 in_correct_path=None) -> None:
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
        self.is_first_incorrect = None
        self.state = (self.token_sequence, self.text) if self.is_root else None
        self.in_correct_path = in_correct_path
    
    def add_child(self, child_node):
        # Check if the node can accept more children
        if self.max_children is not None and len(self.children) >= self.max_children:
            raise ChildrenFullError(f"Cannot add child to node since children are full.")
        
        # Add the child
        self.children.append(child_node)
        child_node.parent = self
        assert self.state is not None, "State must be set before adding children."
        child_node.state = (self.state[0] + child_node.token_sequence, self.state[1] + child_node.text)

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
    An n-ary tree representing the rollout (sampling) process of step-wise grpo. Each SamplingTree instance corresponds to a single data, or a single question in the dataset.
    """
    def __init__(self, root_node, order, max_prompt_length, max_response_length, data_source, ability, reward_model, index, extra_info, tokenizer) -> None:
        """
        root_node (Node): The root node of the tree
        order (int): The number of children each non-leaf node has
        max_prompt_length (int): The maximum length of the prompt that the generator can handle
        max_response_length (int): The maximum length of the response that the generator can output
        data_source (str): The source of the data, e.g., "gsm8k"
        ability (str): The ability level of the data, e.g., "easy", "medium", "hard"
        reward_model (dict): The reward model used for evaluation, e.g., {"ground_truth": "42"}
        index (int): The index of the data in the dataset
        extra_info (dict): Additional information about the data, e.g., {"question": "Calculate 21 * 2", "answer": "42"}
        tokenizer: The tokenizer used to process the text data
        """
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
        self.first_incorrect_nodes = []
    
    def get_redundant_generation_input(self):
        input_ids = []
        leaf_nodes = [node for node in self.all_nodes if node.is_leaf]
        path_num = 0
        res = {}
        for leaf_node in leaf_nodes:
            if leaf_node.score == 1:
                continue
            path = leaf_node.get_ancestors()[1:] + [leaf_node]
            path_res = {}
            for node in path:
                node_input_ids = [node.state[0]] * (self.order - 1)
                path_res[node] = {"input_ids": node_input_ids}
                if not node.is_leaf:
                    input_ids.extend(node_input_ids)
            res[f"path_{path_num}"] = path_res
            path_num += 1
        
        return input_ids, res
    
    
    def add_node(self, node1: Node, node2: Node):
        node1.add_child(node2)
        self.all_nodes.append(node2)
    
    def add_path(self, node: Node, path: list[Node]):
        parent_node = node
        for nd in path:
            self.add_node(parent_node, nd)
            parent_node = nd
    
    def add_paths(self, node: Node, paths: list[list[Node]]):
        for path in paths:
            self.add_path(node, path)
    
    def _compute_scores(self, node):
        if node.is_leaf:
            node.score = compute_score(self.data_source, node.state[1], self.final_answer)
            return 1, node.score

        # Num of leaf nodes for each subtree
        leaf_node_nums = []
        # Total score of all leaf nodes for each subtree
        scores = []
        for child in node.children:
            leaf_num, score = self._compute_scores(child)
            leaf_node_nums.append(leaf_num)
            scores.append(score)

        node.score = sum(scores) / sum(leaf_node_nums)

        return sum(leaf_node_nums), sum(scores)
    
    def compute_scores(self):
        """
        Compute scores for all nodes in the tree.
        """
        self._compute_scores(self.root)
    
    def _compute_advantages(self, node: Node):
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
            self._compute_advantages(child)
        
    def compute_advantages(self):
        """
        Compute advantages for all nodes in the tree.
        """
        self._compute_advantages(self.root)
    
    def assemble_tensor_data(self, input_nodes, output_nodes, adjust_input=False):
        """
        Assemble tensor data based on input and output nodes. The input nodes are for the prompt part, and the output nodes are for the response part.
        This function processes the token sequences of the input and output nodes, constructs the input_id, attention_mask, response, response_mask and other necessary tensors for training.
        
        Args:
            input_nodes (list[Node]): The input nodes
            output_nodes (list[Node]): The output nodes
            
        Returns:
            A dict containing all the necessary tensors
        """
        response = []
        response_mask = []
        prompt = []

        for input_node in input_nodes:
            input_node.advantage = None

        if adjust_input:
            for node in input_nodes:
                prompt.extend(node.token_sequence)
        else:
            prompt = input_nodes[0].token_sequence
        prompt_length = len(prompt)
        prompt = to_fixed_length_tensor(prompt, self.max_prompt_length, self.tokenizer.pad_token_id, "left", torch.int32)

        advantage_pieces = []
        target_output_nodes = output_nodes if adjust_input else input_nodes[1:] + output_nodes
        for node in target_output_nodes:
            response.extend(node.token_sequence)
            advantage_pieces.append([node.advantage if node.advantage is not None else 0.0] * len(node.token_sequence))
        response_length = len(response)
        response = to_fixed_length_tensor(response, self.max_response_length, self.tokenizer.pad_token_id, "right", torch.int32)

        response_mask = torch.zeros(self.max_response_length, dtype=torch.int32)
        response_mask[:response_length] = 1

        input_id = torch.cat([prompt, response], dim=0)
        attention_mask = torch.zeros(self.max_prompt_length, dtype=torch.int32)
        attention_mask[-prompt_length:] = 1
        attention_mask = torch.cat([attention_mask, response_mask], dim=0)

        position_id = get_position_ids_from_attention_mask(attention_mask.reshape(1, -1)).reshape(-1)

        advantage = [adv_value for adv_piece in advantage_pieces for adv_value in adv_piece]
        assert len(advantage) == response_length
        advantage = to_fixed_length_tensor(advantage, self.max_response_length, 0, "right", torch.float32)

        advantage_mask = []
        for target_output_node in target_output_nodes:
            if target_output_node.advantage is not None:
                advantage_mask.extend([1.0] * len(target_output_node.token_sequence))
            else:
                advantage_mask.extend([0.0] * len(target_output_node.token_sequence))
        advantage_mask = to_fixed_length_tensor(advantage_mask, self.max_response_length, 0.0, "right", torch.float32)

        return {
            "input_id": input_id,
            "attention_mask": attention_mask,
            "position_id": position_id,
            "prompt": prompt,
            "response": response,
            "response_mask": response_mask,
            "advantage": advantage,
            "return": advantage,
            "advantage_mask": advantage_mask
        }
  
    def collect_batch_data(self, adjust_input=False):
        """
        Collect batch data from the tree for training.
        After the tree is built and advantages and scores are computed, this function collects all the necessary tensors for training.
        The number of leaf nodes in the tree is the batch size, and each leaf node corresponds to a single piece of data.
        For example, if the tree is like:
                                      0
                                    /   \
                                   1     2
                                  / \
                                 3   4
        Then we would get 3 pieces of data, which are:
        0 -> 1 3
        0 1 -> 4
        0 -> 2
        where the nodes before the arrow are input nodes and those after the arrow are output nodes.
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
            "returns": []
        }
        
        batch_key_to_single_key = {
            "input_ids": "input_id",
            "attention_mask": "attention_mask",
            "position_ids": "position_id",
            "prompts": "prompt",
            "responses": "response",
            "response_mask": "response_mask",
            "advantages": "advantage",
            "returns": "return"
        }

        input_output_pairs = []

        for node in leaf_nodes:
            path = node.get_ancestors() + [node]
            
            # Find the first unvisited node in the path
            pos = 0
            while pos < len(path) and visited[path[pos]]:
                pos += 1
            
            assert pos < len(path)

            input_nodes = path[:pos]
            output_nodes = path[pos:]
            input_output_pairs.append((input_nodes, output_nodes))
            
            for nd in path[pos:]:
                visited[nd] = True
        
        return self.collect_batch_data_from_input_output_pairs(input_output_pairs, adjust_input)
    
    def collect_batch_data_from_input_output_pairs(self, input_output_pairs: List[Tuple[Node, Node]], adjust_input=False):
        tensor_data = {
            "input_ids": [], 
            "attention_mask": [], 
            "position_ids": [], 
            "prompts": [], 
            "responses": [], 
            "response_mask": [], 
            "advantages": [], 
            "returns": [],
            "advantage_mask": []
        }
        
        batch_key_to_single_key = {
            "input_ids": "input_id",
            "attention_mask": "attention_mask",
            "position_ids": "position_id",
            "prompts": "prompt",
            "responses": "response",
            "response_mask": "response_mask",
            "advantages": "advantage",
            "returns": "return",
            "advantage_mask": "advantage_mask"
        }

        for input_nodes, output_nodes in input_output_pairs:
            single_tensor_data = self.assemble_tensor_data(input_nodes, output_nodes, adjust_input)
            for key in tensor_data.keys():
                tensor_data[key].append(single_tensor_data[batch_key_to_single_key[key]])
        
        tensor_batch_dict = {}
        for key in tensor_data.keys():
            tensor_batch_dict[key] = torch.stack(tensor_data[key], dim=0)
        
        num_data = len(input_output_pairs)
        non_tensor_batch_dict = {
            "data_source": np.array([self.data_source] * num_data),
            "ability": np.array([self.ability] * num_data),
            "index": np.array([self.index] * num_data),
            "reward_model": np.array([self.reward_model] * num_data),
            "extra_info": np.array([self.extra_info] * num_data)
        }

        if adjust_input:
            non_tensor_batch_dict["response_length"] = np.array([len(output_nodes[-1].state[0]) - len(input_nodes[0].state[0]) for _, output_nodes in input_output_pairs])
        
        return DataProto.from_single_dict({**tensor_batch_dict, **non_tensor_batch_dict})
    
    def compute_scores_pruned(self):
        filtered_first_incorrect_nodes = []
        parents_of_first_incorrect_nodes = []
        for first_incorrect_node in self.first_incorrect_nodes:
            if first_incorrect_node.parent not in parents_of_first_incorrect_nodes:
                filtered_first_incorrect_nodes.append(first_incorrect_node)
                parents_of_first_incorrect_nodes.append(first_incorrect_node.parent)
        
        for first_incorrect_node in filtered_first_incorrect_nodes:
            parent = first_incorrect_node.parent
            brothers_and_self = parent.children
            for node in brothers_and_self:
                node.score = 0.0 if node.is_first_incorrect else 1.0
    
    def compute_advantages_pruned(self):
        filtered_first_incorrect_nodes = []
        parents_of_first_incorrect_nodes = []
        for first_incorrect_node in self.first_incorrect_nodes:
            if first_incorrect_node.parent not in parents_of_first_incorrect_nodes:
                filtered_first_incorrect_nodes.append(first_incorrect_node)
                parents_of_first_incorrect_nodes.append(first_incorrect_node.parent)
        
        for first_incorrect_node in filtered_first_incorrect_nodes:
            parent = first_incorrect_node.parent
            brothers_and_self = parent.children

            scores = np.array([node.score for node in brothers_and_self])
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            # Here std cannot be 0 because self node is a first incorrect node and its score is 0.
            # At the same time, there are at leat one correct node in its brother nodes, otherwise its
            # parent node would be an incorrect node ahead of it and it couldn't have become a first
            # incorrect node. This is why we don't need to add 1e-6 to std to avoid division by zero.
            advantages = ((scores - mean) / (std)).tolist()
            
            for idx, node in enumerate(brothers_and_self):
                node.advantage = advantages[idx]

    def get_path_to_leaf(self, node: Node):
        """
        Suppose this is a pruned sampling tree. node is a brother node of a first incorrect node.
        If node is also a first incorrect node, then it has only one path forward.
        If node is not a first incorrect node, then it must be a correct node and it must have
        at lease one correct path forward. In that path, there is only one line and no other branches.
        We need to find that path.
        """
        path = [node]
        iter_node = node

        if node.is_first_incorrect:
            while iter_node.children:
                iter_node = iter_node.children[0]
                path.append(iter_node)
        else:
            while iter_node.children:
                exist_correct_node = False
                for child in iter_node.children:
                    if not child.is_first_incorrect:
                        iter_node = child
                        exist_correct_node = True
                        break
                assert exist_correct_node
                path.append(iter_node)
        
        return path

    def collect_batch_data_pruned(self, adjust_input=False):
        filtered_first_incorrect_nodes = []
        parents_of_first_incorrect_nodes = []

        for first_incorrect_node in self.first_incorrect_nodes:
            if first_incorrect_node.parent not in parents_of_first_incorrect_nodes:
                filtered_first_incorrect_nodes.append(first_incorrect_node)
                parents_of_first_incorrect_nodes.append(first_incorrect_node.parent)

        input_and_output_pairs = []
        for first_incorrect_node in filtered_first_incorrect_nodes:
            parent = first_incorrect_node.parent
            brothers_and_self = parent.children
            paths = []
            for node in brothers_and_self:
                paths.append(self.get_path_to_leaf(node))
            input_nodes = parent.get_ancestors() + [parent]
            for output_nodes in paths:
                input_and_output_pairs.append((input_nodes, output_nodes))

        return self.collect_batch_data_from_input_output_pairs(input_and_output_pairs, adjust_input)

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

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 3. 保存为HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
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
                "is_first_incorrect": node.is_first_incorrect,
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


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    score =  _default_compute_score(data_source, solution_str, ground_truth, extra_info)
    
    return score


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

    first_loop = True
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
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int32)
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
        depths= [node.depth for node in all_target_nodes]
        for idx, raw_response in enumerate(raw_responses):
            positions = find_step_split_token_positions(raw_response, tokenizer, config.trainer.step_split_str)
            positions = pick_split_token_positions(positions, config.actor_rollout_ref.rollout.max_tree_depth - depths[idx // num_to_be_added_paths])
            sequences = split_list_by_positions(raw_response, positions)
            texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]
            path = [Node(token_sequence=sequences[i], max_children=order, text=texts[i]) for i in range(len(sequences))]
            paths.append(path)
        for i, target_node in enumerate(all_target_nodes):
            tree = target_node_to_tree[target_node]
            paths_to_be_added = paths[i * (order - 1) : (i + 1) * (order - 1)] if not target_node.is_root else paths[i * order : (i + 1) * order]
            tree.add_paths(target_node, paths_to_be_added)
    
        # In the first loop, drop those trees whose leaf nodes are all correct or all incorrect.
        if first_loop:
            filtered_sampling_trees = []
            for sampling_tree in sampling_trees:
                leaf_nodes = [node for node in sampling_tree.all_nodes if node.is_leaf]
                scores = [compute_score(sampling_tree.data_source, node.state[1], sampling_tree.final_answer) for node in leaf_nodes]
                if sum(scores) > 0 and sum(scores) < len(leaf_nodes):
                    filtered_sampling_trees.append(sampling_tree)
            sampling_trees = filtered_sampling_trees

            first_loop = False

    
    return sampling_trees


def build_pruned_sampling_trees(batch_dict, actor_rollout_wg, tokenizer, config):
    start_input_ids = batch_dict["raw_prompt_ids"]
    order = config.actor_rollout_ref.rollout.order
    question_num = len(start_input_ids)
    start_texts = tokenizer.batch_decode(start_input_ids, skip_special_tokens=False)
    root_nodes = [Node(token_sequence=start_input_ids[i], max_children=order, is_root=True, text=start_texts[i]) for i in range(question_num)]

    input_ids = []
    for root_node in root_nodes:
        input_ids.extend([root_node.token_sequence] * order)

    gen_batch_output = generate(input_ids, tokenizer, actor_rollout_wg, config)
    entropys = actor_rollout_wg.compute_log_prob(gen_batch_output).batch["entropys"]
    raw_responses = gen_batch_output.non_tensor_batch["raw_responses"]

    sampling_trees = []
    for i in range(question_num):
        text_responses = tokenizer.batch_decode(raw_responses[i * order: (i + 1) * order], skip_special_tokens=True)
        scores = [compute_score(batch_dict["data_source"][i], text, batch_dict["reward_model"][i]["ground_truth"]) for text in text_responses]
        if all([score == 0  for score in scores]) or all([score == 1  for score in scores]):
            continue

        paths = generate_paths(batch_dict["data_source"][i], input_ids[i * order: (i + 1) * order], raw_responses[i * order: (i + 1) * order], entropys[i * order: (i + 1) * order], tokenizer, config, batch_dict["reward_model"][i]["ground_truth"], config.trainer.top_n_entropy_tokens)
        
        sampling_tree = SamplingTree(
                root_nodes[i], 
                order,
                config.data.max_prompt_length,
                config.data.max_response_length,
                batch_dict["data_source"][i],
                batch_dict["ability"][i],
                batch_dict["reward_model"][i],
                batch_dict["index"][i],
                batch_dict["extra_info"][i],
                tokenizer
            )
        
        sampling_tree.add_paths(sampling_tree.root, paths)
        path_scores = [path[-1].score for path in paths]
        sampling_tree.root.path_scores = path_scores
        sampling_trees.append(sampling_tree)

    finder_stype = config.actor_rollout_ref.rollout.finder_style
    if finder_stype == "redundant":
        find_first_incorrect_step_with_redundant_generation(sampling_trees, tokenizer, config, actor_rollout_wg)
    elif finder_stype == "binary_search":
        find_first_incorrect_step_with_python_generator(sampling_trees, tokenizer, config, actor_rollout_wg)

    return sampling_trees


def find_first_incorrect_step_with_redundant_generation(sampling_trees: List[SamplingTree], tokenizer, config, actor_rollout_wg):
    order = config.actor_rollout_ref.rollout.order

    all_input_ids = []
    all_structure_data = {}
    for sampling_tree in sampling_trees:
        input_ids, structure_tree_data = sampling_tree.get_redundant_generation_input()
        all_input_ids.extend(input_ids)
        all_structure_data[sampling_tree] = structure_tree_data
    
    gen_batch_output = generate(all_input_ids, tokenizer, actor_rollout_wg, config)
    entropys = actor_rollout_wg.compute_log_prob(gen_batch_output).batch["entropys"]

    raw_responses = gen_batch_output.non_tensor_batch["raw_responses"]
    response_texts = tokenizer.batch_decode(raw_responses.tolist(), skip_special_tokens=False)
    prompt_texts = tokenizer.batch_decode(all_input_ids, skip_special_tokens=False)
    
    cnt = 0
    for tree, structure_tree_data in all_structure_data.items():
        for path, structure_path_data in structure_tree_data.items():
            node_idx = 0
            find_first_incorrect = False
            for node, structure_node_data in structure_path_data.items():
                is_first_incorrect = False
                if not find_first_incorrect and node.is_leaf:
                    is_first_incorrect = True
                elif not find_first_incorrect and not node.is_leaf:
                    structure_node_data["prompt_plus_response_texts"] = prompt_texts[cnt * (order - 1) : (cnt + 1) * (order - 1)] + response_texts[cnt * (order - 1) : (cnt + 1) * (order - 1)]
                    scores = [compute_score(tree.data_source, text, tree.final_answer) for text in structure_node_data["prompt_plus_response_texts"]]
                    is_first_incorrect = all(score == 0 for score in scores)

                if is_first_incorrect:
                    node.is_first_incorrect = True
                    tree.first_incorrect_nodes.append(node)
                    find_first_incorrect = True
                    if node_idx > 0:
                        paths = generate_paths(tree.data_source, all_input_ids[(cnt - 1) * (order - 1) : (cnt) * (order - 1)], raw_responses[(cnt - 1) * (order - 1) : (cnt) * (order - 1)], entropys[(cnt - 1) * (order - 1) : (cnt) * (order - 1)], tokenizer, config, tree.final_answer, config.trainer.top_n_entropy_tokens - node.parent.depth)
                        tree.add_paths(node.parent, paths)
                        path_scores = [path[-1].score for path in paths]
                        node.parent.path_scores = [0.0] + path_scores
                
                if not node.is_leaf:
                    cnt += 1
                node_idx += 1

    # After finding the first incorrect steps, we also need to evaluate the correctness of their brother nodes
    input_ids = []
    all_structure_data = {}
    for sampling_tree in sampling_trees:
        tree_structure_data = {}
        for node in sampling_tree.all_nodes:
            if node.is_first_incorrect:
                for idx, brother in enumerate(node.parent.children):
                    # 如果 first_incorrect_node depth 为 1，那么在判断其兄弟节点的正确性时，若兄弟节点所在的 path 为 incorrect，其兄弟节点的正确性一定已被判断过了，无须重复判断
                    if node.parent.path_scores[idx] == 1 or brother.depth == 1:
                        continue
                    node_structure_data = {"input_ids": [brother.state[0]] * (order - 1)}
                    input_ids.extend(node_structure_data["input_ids"])
                    tree_structure_data[brother] = node_structure_data
        all_structure_data[sampling_tree] = tree_structure_data
    
    gen_batch_output = generate(input_ids, tokenizer, actor_rollout_wg, config)
    raw_responses = gen_batch_output.non_tensor_batch["raw_responses"]
    assert raw_responses.shape[0] == len(input_ids)
    response_texts = tokenizer.batch_decode(raw_responses.tolist(), skip_special_tokens=False)
    prompt_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

    cnt = 0
    for sampling_tree, structure_tree_data in all_structure_data.items():
        for node, structure_node_data in structure_tree_data.items():
            structure_node_data["prompt_plus_response_texts"] = prompt_texts[cnt * (order - 1) : (cnt + 1) * (order - 1)] + response_texts[cnt * (order - 1) : (cnt + 1) * (order - 1)]
            scores = [compute_score(tree.data_source, text, sampling_tree.final_answer) for text in structure_node_data["prompt_plus_response_texts"]]
            if all(score == 0 for score in scores):
                node.is_first_incorrect = True
                sampling_tree.first_incorrect_nodes.append(node)
            else:
                node.is_first_incorrect = False
            cnt += 1


def to_fixed_length_tensor(
    seq: Union[List[int], List[List[int]]], 
    max_length: int, 
    pad_token_id: int | float, 
    pad_direction: str = 'right', 
    dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    """
    Convert a sequence to a fixed-length tensor by padding or truncating.
    
    Args:
        seq: A list of integers or a list of lists of integers representing the sequence(s)
        max_length: The maximum length of each sequence
        pad_token_id: The token ID used for padding
        pad_direction: The direction of padding, 'left' or 'right' (default: 'right')
        dtype: The data type of the tensor (default: torch.int32)
    
    Returns:
        - If seq is List[int]: A torch.Tensor of shape [max_length]
        - If seq is List[List[int]]: A torch.Tensor of shape [num_sequences, max_length]
    """
    def _process_single_sequence(single_seq: List[int]) -> List[int]:
        """Process a single sequence by truncating or padding."""
        if len(single_seq) > max_length:
            # Truncate if sequence is too long
            return single_seq[:max_length]
        elif len(single_seq) < max_length:
            # Pad if sequence is too short
            padding_length = max_length - len(single_seq)
            if pad_direction == 'left':
                return [pad_token_id] * padding_length + single_seq
            elif pad_direction == 'right':
                return single_seq + [pad_token_id] * padding_length
            else:
                raise ValueError(f"Invalid pad_direction: {pad_direction}. Must be 'left' or 'right'.")
        else:
            # Sequence is already the correct length
            return single_seq
    
    # Check if input is 1D (list of integers) or 2D (list of lists of integers)
    if len(seq) == 0:
        # Handle empty input
        if isinstance(seq, list) and all(isinstance(item, list) for item in seq):
            # Empty 2D list
            return torch.empty((0, max_length), dtype=dtype)
        else:
            # Empty 1D list
            return torch.empty((0,), dtype=dtype)
    
    # Determine if this is a 2D input (list of lists)
    if isinstance(seq[0], list):
        # 2D case: List[List[int]]
        processed_sequences = []
        for single_seq in seq:
            processed_seq = _process_single_sequence(single_seq)
            processed_sequences.append(processed_seq)
        
        return torch.tensor(processed_sequences, dtype=dtype)
    
    else:
        # 1D case: List[int]
        processed_seq = _process_single_sequence(seq)
        return torch.tensor(processed_seq, dtype=dtype)


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


def generate_paths(data_source, prompts, raw_responses, entropys, tokenizer, config, final_answer, top_n_entropy_tokens):
    paths = []
    for i in range(len(raw_responses)):
        paths.append(generate_path(data_source, prompts[i], raw_responses[i], entropys[i], tokenizer, config, final_answer, top_n_entropy_tokens))
    return paths


def extract_raw_responses(responses: torch.Tensor, response_mask: torch.Tensor) -> np.ndarray:
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
    np_responses = responses.cpu().numpy()
    np_response_mask = response_mask.cpu().numpy()
    
    batch_size = np_responses.shape[0]
    raw_responses = []
    
    for i in range(batch_size):
        # Extract values where mask is 1
        valid_indices = np_response_mask[i] == 1
        raw_response = np_responses[i][valid_indices].tolist()
        raw_responses.append(raw_response)
    
    return np.array(raw_responses, dtype='O')


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
    """
    Find positions of the step split token in the input_ids. Note that we cannot directly encode the step_split_str and find the encoding result in input_ids. For example, we seperately encode step_split_str to get step_split_token_ids but this token id sequence may not be in the input_ids, since the input_ids is the result of encoding the whole text. Even if the whole text contains the step_split_str, the tokenization may not match exactly.
    """
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


def pick_split_token_positions(split_positions, num_parts):
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


def get_top_entropy_tokens_single(entropy, response, tokenizer, top_n=5):
    """
    从单个序列中选择 entropy 最高的前 n 个 token
    
    Args:
        entropy: [response_len] 形状的 tensor，单个序列的entropy值
        response: 单个序列的 token ids
        tokenizer: 用于解码的 tokenizer
        top_n: 要选择的 top token 数量
    
    Returns:
        result: 包含高entropy token信息的字典
    """
    if isinstance(response, list):
        response = torch.tensor(response)

    response_len = entropy.shape[0]
    
    # 获取 top_n 个最高 entropy 的索引
    top_indices = torch.topk(entropy, min(top_n, response_len), dim=0)
    top_entropy_values = top_indices.values  # entropy 值
    top_positions = top_indices.indices      # 位置索引
    
    batch_results = []
    for i in range(len(top_positions)):
        pos = top_positions[i].item()
        entropy_val = top_entropy_values[i].item()
        
        # 获取对应位置的 token id
        if pos < len(response):
            token_id = response[pos].item()
            # 解码 token
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            
            batch_results.append({
                'position': pos,
                'entropy': entropy_val,
                'token_id': token_id,
                'token_str': token_str
            })
    
    return {
        'top_entropy_tokens': batch_results
    }


def get_top_entropy_tokens_batch(responses, entropys, tokenizer, top_n=5):
    """
    从每个序列中选择 entropy 最高的前 n 个 token (批处理版本)
    
    Args:
        entropys: [bsz, response_len] 形状的 tensor
        gen_batch_output: 生成的 batch 输出，包含 token ids
        tokenizer: 用于解码的 tokenizer
        top_n: 要选择的 top token 数量
    
    Returns:
        results: 包含每个序列高entropy token信息的列表
    """
    bsz, response_len = entropys.shape
    results = []
    
    for batch_idx in range(bsz):
        # 获取当前序列的 entropy 值和对应的 response
        seq_entropy = entropys[batch_idx]  # [response_len]
        seq_response = responses[batch_idx]
        
        # 调用单序列处理函数
        single_result = get_top_entropy_tokens_single(
            entropy=seq_entropy,
            response=seq_response,
            tokenizer=tokenizer,
            top_n=top_n
        )
        
        # 添加 batch_idx 信息
        single_result['batch_idx'] = batch_idx
        results.append(single_result)
    
    return results


def generate(input_ids, tokenizer, actor_rollout_wg, config):
    batch_size = len(input_ids)

    # Create ndarray of raw_prompt_ids
    raw_prompt_ids = np.array(input_ids, dtype='O')

    # Create tensors of input_ids, attention_mask, and position_ids
    input_ids = to_fixed_length_tensor(input_ids, config.data.max_prompt_length, tokenizer.pad_token_id, pad_direction='left', dtype=torch.int32)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int32)
    position_ids = get_position_ids_from_attention_mask(attention_mask)
    
    gen_batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "raw_prompt_ids": raw_prompt_ids
    }
    gen_batch = DataProto.from_single_dict(gen_batch_dict, auto_padding=True)

    gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)
    world_size = actor_rollout_wg.world_size
    if len(gen_batch_output) % world_size != 0:
        gen_batch_output.padding(world_size - (len(gen_batch_output) % world_size), "last")
    
    gen_batch_output = gen_batch_output[:batch_size]
    response_mask = compute_response_mask(gen_batch_output.batch["responses"], gen_batch_output.batch["attention_mask"])
    gen_batch_output.batch["response_mask"] = response_mask
    gen_batch_output.non_tensor_batch["raw_responses"] = extract_raw_responses(gen_batch_output.batch["responses"], response_mask)

    return gen_batch_output


def generate_path(data_source, prompt, response, entropy, tokenizer, config, final_answer, top_n_entropy_tokens):
    if not isinstance(prompt, list):
        prompt = prompt.tolist()
    if not isinstance(response, list):
        response = response.tolist()

    if config.trainer.entropy_driven_step_split:
        result = get_top_entropy_tokens_single(entropy, response, tokenizer, top_n_entropy_tokens + 2)
        positions = [item["position"] for item in result["top_entropy_tokens"]]
        positions = [position for position in positions if position > 0 and position < len(response)]
        positions = positions[:top_n_entropy_tokens]
        positions = [position - 1 for position in positions]
        positions.sort()
    else:
        positions = find_step_split_token_positions(response, tokenizer, config.trainer.step_split_str)
    sequences = split_list_by_positions(response, positions)
    texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]
    is_correct_response = compute_score(data_source, tokenizer.decode(prompt + response, skip_special_tokens=True), final_answer) == 1
    path = [Node(token_sequence=sequences[i], max_children=config.actor_rollout_ref.rollout.order, text=texts[i], in_correct_path=is_correct_response) for i in range(len(sequences))]
    path[-1].score = 1.0 if is_correct_response else 0.0

    return path


@dataclass
class GenerationRequest:
    """单个生成请求"""
    task_id: str  # 标识这是哪个具体的子任务
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    raw_prompt_ids: np.ndarray

@dataclass 
class GenerationResponse:
    """生成结果"""
    request: GenerationRequest
    responses: torch.Tensor
    attention_mask: torch.Tensor
    raw_responses: np.ndarray
    entropys: torch.Tensor

def find_first_incorrect_step_generator(tree, nodes, l, r, tokenizer, config, task_prefix="") -> Generator[List[GenerationRequest], List[GenerationResponse], None]:
    """
    部分递归版本的find_first_incorrect_step生成器
    - 对当前节点列表的搜索采用迭代
    - 只对新采样的错误样本进行递归
    - 添加递归深度控制
    
    Args:
        tree: SamplingTree
        nodes: list[Node] 
        l, r: 搜索范围
        tokenizer: tokenizer
        config: 配置
        task_prefix: 任务前缀，用于生成唯一的task_id
    
    Yields:
        GenerationRequest: 需要进行生成的请求
        
    Receives:
        GenerationResponse: 生成的结果
    """
    
    # 迭代搜索当前nodes中的first incorrect step
    left, right = l, r
    
    while right >= left:
        m = (left + right) // 2
        mid_node = nodes[m]
        
        N = config.actor_rollout_ref.rollout.order - 1
        
        # 准备生成请求
        input_id = mid_node.state[0]
        raw_prompt_ids = np.array([input_id] * N, dtype='O')
        input_id = to_fixed_length_tensor(input_id, config.data.max_prompt_length, tokenizer.pad_token_id, "left", torch.int32)
        input_ids = input_id.unsqueeze(0).repeat(N, 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int32)
        position_ids = get_position_ids_from_attention_mask(attention_mask)
        
        # 创建生成请求
        task_id = f"{task_prefix}_{left}_{right}_{m}"
        request = GenerationRequest(
            task_id=task_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            raw_prompt_ids=raw_prompt_ids
        )
        
        # yield请求并接收响应
        responses = yield [request]
        response = responses[0]
        
        # 处理响应
        scores = [compute_score(tree.data_source, tokenizer.decode(raw_prompt_ids[i].tolist() + response.raw_responses[i]), tree.final_answer) for i in range(len(response.raw_responses))]
        raw_responses = response.raw_responses
        all_incorrect = all(score == 0 for score in scores)
        entropys = response.entropys
        
        if all_incorrect:
            # 如果所有采样都是错误的，说明当前节点是错误的
            # 继续在左半部分搜索
            right = m - 1
        else:
            # 如果存在正确的采样，说明当前节点是正确的
            paths = generate_paths(tree.data_source, raw_prompt_ids, raw_responses, entropys, tokenizer, config, tree.final_answer, config.trainer.top_n_entropy_tokens - mid_node.depth)
            tree.add_paths(mid_node, paths)
            left = m + 1
    
    nodes[left].is_first_incorrect = True
    for node in nodes[left].get_ancestors()[1:]:
        node.is_first_incorrect = False

    # 判断第一个错误节点的兄弟节点是否正确，即兄弟节点是否也是第一个错误节点
    brothers = []
    for node in nodes[left].parent.children:
        if node is not nodes[left]:
            brothers.append(node)
            
    first_step_generators = []
    for brother in brothers:
        if brother.in_correct_path or brother.is_first_incorrect == False:
            brother.is_first_incorrect = False
        else:
            first_step_generators.append(check_single_step_generator(
                tree, brother, tokenizer, config, f"{task_prefix}_brother_{id(brother)}"
            ))
    
    # 并行处理所有第一步检查
    if first_step_generators:
        yield from run_generators_in_parallel(first_step_generators)

    tree.first_incorrect_nodes.append(nodes[left])


def check_single_step_generator(tree, node, tokenizer, config, task_prefix) -> Generator[List[GenerationRequest], List[GenerationResponse], None]:
    """
    检查单个步骤是否正确的生成器
    
    Args:
        tree: SamplingTree
        node: first_step 对应的 node
        tokenizer: tokenizer
        config: 配置
        task_prefix: 任务前缀
    """
    N = config.actor_rollout_ref.rollout.order - 1
    
    # 准备生成请求
    input_id = node.state[0]
    raw_prompt_ids = np.array([input_id] * N, dtype='O')
    input_id = to_fixed_length_tensor(input_id, config.data.max_prompt_length, tokenizer.pad_token_id, "left", torch.int32)
    input_ids = input_id.unsqueeze(0).repeat(N, 1)
    input_ids = torch.tensor(input_ids, dtype=torch.int32)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.int32)
    position_ids = get_position_ids_from_attention_mask(attention_mask)
    
    # 创建生成请求
    task_id = f"{task_prefix}_check_node_{id(node)}"
    request = GenerationRequest(
        task_id=task_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        raw_prompt_ids=raw_prompt_ids,
    )
    
    # yield请求并接收响应
    responses = yield [request]
    response = responses[0]
    full_texts = [tokenizer.decode(raw_prompt_ids[i].tolist() if not isinstance(raw_prompt_ids[i], list) else  + response.raw_responses[i].tolist() if not isinstance(response.raw_responses[i], list) else response.raw_responses[i]) for i in range(len(response.raw_responses))]
    
    # 处理响应
    scores = [compute_score(tree.data_source, full_text, tree.final_answer) for full_text in full_texts]
    all_incorrect = all(score == 0 for score in scores)
    
    if all_incorrect:
        # 如果所有采样都是错误的，标记为first incorrect
        node.is_first_incorrect = True
        tree.first_incorrect_nodes.append(node)


def run_generators_in_parallel(generators: List[Generator]) -> Generator[List[GenerationRequest], List[GenerationResponse], None]:
    """
    并行运行多个生成器
    """
    if not generators:
        return
    
    active_generators = {i: gen for i, gen in enumerate(generators)}
    pending_requests = {}
    
    # 启动所有生成器
    for i, gen in active_generators.items():
        try:
            requests = next(gen)
            # 处理请求列表
            for req in requests:
                pending_requests[req.task_id] = (i, gen, req)
        except StopIteration:
            pass
    
    while pending_requests:
        # 批量收集所有待处理的请求
        requests = [req for _, _, req in pending_requests.values()]
        
        if requests:
            # yield所有请求，等待批量响应
            responses = yield requests
            
            # 将响应分发回对应的生成器
            response_map = {resp.request.task_id: resp for resp in responses}
            
            for task_id, (gen_id, gen, _) in list(pending_requests.items()):
                if task_id in response_map:
                    response = response_map[task_id]
                    del pending_requests[task_id]
                    
                    try:
                        # 将响应发送给生成器，并获取下一个请求列表
                        next_requests = gen.send([response])
                        for req in next_requests:
                            pending_requests[req.task_id] = (gen_id, gen, req)
                    except StopIteration:
                        # 该生成器已完成
                        pass


class ParallelIncorrectStepFinder:
    """
    并行查找多个采样树的第一个错误步骤
    对于每个采样树的每个错误分支，通过二分法查找第一个错误步骤
    由于每个错误分支的查找次数可能不同，因此使用生成器来处理
    """
    
    def __init__(self, tokenizer, config, actor_rollout_wg, sampling_trees):
        self.tokenizer = tokenizer
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.sampling_trees = sampling_trees
    
    def process_sampling_trees(self):
        # 创建所有任务的生成器
        generators = []
        
        for tree_id, sampling_tree in enumerate(self.sampling_trees):
            leaf_nodes = [node for node in sampling_tree.all_nodes if node.is_leaf]
            uncorrect_leaf_nodes = [
                node for node in leaf_nodes 
                if compute_score(sampling_tree.data_source, node.state[1], sampling_tree.final_answer) == 0
            ]
            
            for leaf_idx, uncorrect_leaf_node in enumerate(uncorrect_leaf_nodes):
                path = uncorrect_leaf_node.get_ancestors()[1:] + [uncorrect_leaf_node]
                gen = find_first_incorrect_step_generator(
                    sampling_tree, path, 0, len(path) - 2, 
                    self.tokenizer, self.config, f"tree_{tree_id}_leaf_{leaf_idx}"
                )
                generators.append(gen)
        
        # 执行并行处理
        self._execute_parallel_generators(generators)
    
    def _execute_parallel_generators(self, generators):
        """执行并行生成器"""

        import re
        def get_next_step_dir(base_dir):
            # 正则匹配 step_数字 的模式
            step_pattern = re.compile(r"^step_(\d+)$")
            max_step = 0

            # If base_dir does not exist, create it
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            # 遍历目录，查找符合模式的子目录
            for name in os.listdir(base_dir):
                full_path = os.path.join(base_dir, name)
                if os.path.isdir(full_path):
                    match = step_pattern.match(name)
                    if match:
                        step_num = int(match.group(1))
                        max_step = max(max_step, step_num)

            # 计算下一个 step 编号
            next_step = max_step + 1
            next_dir_name = f"step_{next_step}"
            next_dir_path = os.path.join(base_dir, next_dir_name)

            return next_dir_path

        next_step_dir = get_next_step_dir(self.config.trainer.sampling_tree_dir)

        if not generators:
            return
        
        active_generators = {i: gen for i, gen in enumerate(generators)}
        pending_requests = {}
        
        # 启动所有生成器
        for i, gen in active_generators.items():
            try:
                requests = next(gen)
                # 处理请求列表
                for req in requests:
                    pending_requests[req.task_id] = (i, gen, req)
            except StopIteration:
                pass
        
        iter = 0
        while pending_requests:
            # 批量处理所有待处理的请求
            requests = [req for _, _, req in pending_requests.values()]
            
            if not requests:
                break
            
            for idx, sampling_tree in enumerate(self.sampling_trees):
                sampling_tree.visualize(output_file=os.path.join(next_step_dir, f"iter_{iter}/tree_{idx}.html"))

            # 批量生成
            responses = self._batch_generate(requests)
            
            # 分发响应
            completed_generators = set()
            new_requests = {}
            
            response_map = {resp.request.task_id: resp for resp in responses}
            
            for task_id, (gen_id, gen, _) in list(pending_requests.items()):
                if task_id in response_map:
                    response = response_map[task_id]
                    try:
                        next_requests = gen.send([response])
                        # 处理下一轮的请求列表
                        for req in next_requests:
                            new_requests[req.task_id] = (gen_id, gen, req)
                    except StopIteration:
                        completed_generators.add(gen_id)
            
            # 更新待处理请求
            pending_requests = {k: v for k, v in new_requests.items()}

            iter += 1
    
    def _batch_generate(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """批量生成处理"""
        if not requests:
            return []
        
        # 合并所有输入
        all_input_ids = []
        all_attention_mask = []
        all_position_ids = []
        all_raw_prompt_ids = []
        request_info = []
        total_samples = 0
        
        for req in requests:
            batch_size = req.input_ids.shape[0]
            all_input_ids.append(req.input_ids)
            all_attention_mask.append(req.attention_mask)
            all_position_ids.append(req.position_ids)
            all_raw_prompt_ids.extend(req.raw_prompt_ids)
            request_info.append((req, batch_size))
            total_samples += batch_size
        
        pprint(f"[BatchGenerate] Processing {len(requests)} requests with {total_samples} total samples")
        
        # 合并为大batch
        batch_input_ids = torch.cat(all_input_ids, dim=0)
        batch_attention_mask = torch.cat(all_attention_mask, dim=0)
        batch_position_ids = torch.cat(all_position_ids, dim=0)
        batch_raw_prompt_ids = np.array(all_raw_prompt_ids, dtype='O')
        
        gen_batch_dict = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "position_ids": batch_position_ids,
            "raw_prompt_ids": batch_raw_prompt_ids
        }
        gen_batch = DataProto.from_single_dict(gen_batch_dict, auto_padding=True)
        
        # 执行批量生成
        pprint(f"[BatchGenerate] Starting generation for batch size: {batch_input_ids.shape[0]}")
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        batch_responses = gen_batch_output.batch["responses"]
        batch_entropys = self.actor_rollout_wg.compute_log_prob(gen_batch_output).batch["entropys"]
        batch_attention_mask_out = gen_batch_output.batch["attention_mask"]
        pprint(f"[BatchGenerate] Generation completed, processing responses")
        
        # 分解结果
        responses = []
        start_idx = 0
        
        for req, batch_size in request_info:
            end_idx = start_idx + batch_size
            
            req_responses = batch_responses[start_idx:end_idx]
            req_attention_mask = batch_attention_mask_out[start_idx:end_idx]
            req_entropys = batch_entropys[start_idx:end_idx]
            response_mask = compute_response_mask(req_responses, req_attention_mask)
            raw_responses = extract_raw_responses(req_responses, response_mask)
            
            response = GenerationResponse(
                request=req,
                responses=req_responses,
                attention_mask=req_attention_mask,
                raw_responses=raw_responses,
                entropys=req_entropys
            )
            responses.append(response)
            start_idx = end_idx
        
        pprint(f"[BatchGenerate] Completed processing {len(responses)} responses")
        return responses


def find_first_incorrect_step_with_python_generator(sampling_trees, tokenizer, config, actor_rollout_wg):
    """
    并行处理多个sampling_trees的入口函数
    """
    finder = ParallelIncorrectStepFinder(tokenizer, config, actor_rollout_wg, sampling_trees)
    finder.process_sampling_trees()