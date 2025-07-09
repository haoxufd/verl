import copy
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import json
from pathlib import Path
from typing import Dict, Any

import os

import torch
import numpy as np

from pprint import pprint

from dataclasses import dataclass
from typing import Generator, List, Tuple

from collections import defaultdict

# from dapo.docs import conf
from functools import wraps

from verl.protocol import pad_dataproto_to_divisor
from verl.utils.torch_functional import pad_sequence_to_length

TOKEN_IDX = 0
TEXT_IDX = 1


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
                 is_correct_path=None,
                 is_original_batch_item=None) -> None:
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
        self.is_correct_path = is_correct_path
        self.batch_item = None
        self.is_original_batch_item = is_original_batch_item
    
    def add_child(self, child_node):
        # Check if the node can accept more children
        if self.max_children is not None and len(self.children) >= self.max_children:
            raise ChildrenFullError(f"Cannot add child to node since children are full.")
        
        # Add the child
        self.children.append(child_node)
        child_node.parent = self
        assert self.state is not None, "State must be set before adding children."
        try:
            child_node.state = (self.state[0] + child_node.token_sequence, self.state[1] + child_node.text)
        except Exception as e:
            breakpoint()

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
    def __init__(self, 
                 root_node, 
                 order, 
                 max_prompt_length, 
                 max_response_length, 
                 tree_dict,
                 tokenizer,
                 rollout_config) -> None:
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
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.all_nodes = [root_node]
        self.tree_dict = tree_dict
        self.first_incorrect_nodes = []
        self.rollout_config = rollout_config
        self.ground_truth = self.tree_dict["reward_model"][0]["ground_truth"]
    
    def get_redundant_batch_and_gen_batch(self, sample_mode="initial"):
        batch_list = []
        gen_batch_list = []
        leaf_nodes = [node for node in self.all_nodes if node.is_leaf]

        structured_tree_data = defaultdict(list)
        for leaf_node in leaf_nodes:
            if leaf_node.is_correct_path:
                continue
            ancestors_wo_root = leaf_node.get_ancestors()[1:]
            for node in ancestors_wo_root:
                batch, gen_batch = self.get_batch_and_gen_batch_of_node(node, sample_mode)
                batch_list.append(batch)
                gen_batch_list.append(gen_batch)
            structured_tree_data[leaf_node] = ancestors_wo_root + [leaf_node]
        
        return DataProto.concat(batch_list), DataProto.concat(gen_batch_list), structured_tree_data
    
    def get_batch_and_gen_batch_of_node(self, node, sample_mode="initial"):
        batch = DataProto.from_single_dict(self.tree_dict)
        gen_batch = batch.pop(
            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            non_tensor_batch_keys=["raw_prompt_ids"]
        )

        gen_batch.non_tensor_batch['raw_question_ids'] = gen_batch.non_tensor_batch.pop('raw_prompt_ids')
        temp_container = np.empty(1, dtype=object)
        temp_container[0] = node.state[TOKEN_IDX]
        gen_batch.non_tensor_batch['real_prompt_ids'] = temp_container
        # gen_batch.non_tensor_batch['real_prompt_ids'] = np.array([node.state[TOKEN_IDX]], dtype='O')
        gen_batch.meta_info.update({"sample_mode": sample_mode})

        return batch, gen_batch
    
    def generate_path(self, node, batch_item, response, entropy, score, num_node):
        if num_node > 1:
            if self.rollout_config.entropy_driven_step_split:
                result = get_top_entropy_tokens(entropy, response, num_node - 1)
                positions = [item["position"] for item in result]
                positions = [position - 1 for position in positions]
                positions.sort()
            else:
                text = self.tokenizer.decode(response)
                encoding = self.tokenizer(text, return_offsets_mapping=True)
                positions = find_step_split_token_positions(text, encoding, self.rollout_config.step_split_str)
            sequences = split_list_by_positions(response, positions)
        else:
            sequences = [response]
        texts = [self.tokenizer.decode(seq, skip_special_tokens=False) for seq in sequences]
        path = [Node(token_sequence=sequences[i], max_children=self.rollout_config.n // 2, text=texts[i]) for i in range(len(sequences))]
        path[-1].is_correct_path = (score == 1.0)
        path[-1].batch_item = batch_item
        path[-1].is_original_batch_item = node.is_root

        self.add_path(node, path)
    
    def generate_paths(self, node, batch, raw_responses, entropys, scores, num_nodes):
        paths = []
        for i in range(len(raw_responses)):
            entropy = None if entropys is None else entropys[i]
            paths.append(self.generate_path(node, batch[i:i+1], raw_responses[i], entropy, scores[i], num_nodes if isinstance(num_nodes, int) else num_nodes[i]))
    
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

    def merge_nodes(self, node: Node):
        new_node = copy.copy(node)
        new_node.children = []
        new_node.children_is_full = False
        new_node.is_leaf = True

        self.all_nodes[self.all_nodes.index(node)] = new_node
        node.parent.children[node.parent.children.index(node)] = new_node
        
        path = []
        iter_node = node
        while iter_node.children:
            assert len(iter_node.children) == 1
            path.append(iter_node.children[0])
            iter_node = iter_node.children[0]

        for _node in path:
            new_node.token_sequence += _node.token_sequence
            new_node.text += _node.text
            assert node.state is not None, "State must be set before merging nodes."
            new_node.state = (new_node.state[0] + _node.token_sequence, new_node.state[1] + _node.text)
            if _node.is_leaf:
                new_node.is_correct_path = _node.is_correct_path
                new_node.batch_item = _node.batch_item
            
            self.all_nodes.remove(_node)
        
        return new_node
    
    def get_all_leaf_nodes_under_a_node(self, node):
        import queue

        res = []
        task_queue = queue.Queue()
        task_queue.put(node)

        while not task_queue.empty():
            _node = task_queue.get()
            if _node.is_leaf:
                res.append(_node)
            else:
                for nd in _node.children:
                    task_queue.put(nd)
        
        return res
    
    def _compute_scores(self, node):
        if node.is_leaf:
            node.score = compute_score(self.data_source, node.state[1], self.ground_truth)
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
        # filtered_first_incorrect_nodes = []
        # parents_of_first_incorrect_nodes = []
        # for first_incorrect_node in self.first_incorrect_nodes:
        #     if first_incorrect_node.parent not in parents_of_first_incorrect_nodes:
        #         filtered_first_incorrect_nodes.append(first_incorrect_node)
        #         parents_of_first_incorrect_nodes.append(first_incorrect_node.parent)
        
        # for first_incorrect_node in filtered_first_incorrect_nodes:
        #     parent = first_incorrect_node.parent
        #     brothers_and_self = parent.children
        #     for node in brothers_and_self:
        #         node.score = 0.0 if node.is_first_incorrect else 0.5 if node.is_first_incorrect is None else 1.0
        self._compute_scores_pruned(self.root)
    
    def compute_advantages_pruned(self):
        # filtered_first_incorrect_nodes = []
        # parents_of_first_incorrect_nodes = []
        # for first_incorrect_node in self.first_incorrect_nodes:
        #     if first_incorrect_node.parent not in parents_of_first_incorrect_nodes:
        #         filtered_first_incorrect_nodes.append(first_incorrect_node)
        #         parents_of_first_incorrect_nodes.append(first_incorrect_node.parent)
        
        # for first_incorrect_node in filtered_first_incorrect_nodes:
        #     parent = first_incorrect_node.parent
        #     brothers_and_self = parent.children

        #     scores = np.array([node.score for node in brothers_and_self])
        #     mean = np.mean(scores)
        #     std = np.std(scores, ddof=1)
        #     # Here std cannot be 0 because self node is a first incorrect node and its score is 0.
        #     # At the same time, there are at leat one correct node in its brother nodes, otherwise its
        #     # parent node would be an incorrect node ahead of it and it couldn't have become a first
        #     # incorrect node. This is why we don't need to add 1e-6 to std to avoid division by zero.
        #     advantages = ((scores - mean) / (std)).tolist()
            
        #     for idx, node in enumerate(brothers_and_self):
        #         node.advantage = advantages[idx]
        self._compute_advantages_pruned(self.root)
    
    def _compute_scores_pruned(self, node):
        if node.is_leaf:
            node.score = 1.0 if node.is_correct_path else 0.0
            return node.score

        scores = []
        for child in node.children:
            score = self._compute_scores_pruned(child)
            scores.append(score)

        if not node.is_root:
            node.score = 1.0 if any(score == 1.0 for score in scores) else 0.0
            return node.score
    
    def _compute_advantages_pruned(self, node):
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
            self._compute_advantages_pruned(child)

    # def collect_batch_data_pruned(self):
    #     filtered_first_incorrect_nodes = []
    #     parents_of_first_incorrect_nodes = []

    #     for first_incorrect_node in self.first_incorrect_nodes:
    #         if first_incorrect_node.parent not in parents_of_first_incorrect_nodes:
    #             filtered_first_incorrect_nodes.append(first_incorrect_node)
    #             parents_of_first_incorrect_nodes.append(first_incorrect_node.parent)

    #     range_start = []
    #     range_end = []
    #     batch_items = []
    #     advs = []
    #     for first_incorrect_node in filtered_first_incorrect_nodes:
    #         parent = first_incorrect_node.parent
    #         brothers_and_self = parent.children
    #         range_start.extend([len(node.state[TOKEN_IDX]) - len(node.token_sequence) - len(self.root.token_sequence) for node in brothers_and_self])
    #         range_end.extend([range_start[i] + len(brothers_and_self[i].token_sequence) for i in range(len(brothers_and_self))])
    #         for node in brothers_and_self:
    #             leaf_nodes = self.get_all_leaf_nodes_under_a_node(node)
    #             find_correct_path = False
    #             for leaf_node in leaf_nodes:
    #                 if leaf_node.is_correct_path:
    #                     batch_items.append(leaf_node.batch_item)
    #                     find_correct_path = True
    #                     break
    #             if not find_correct_path:
    #                 batch_items.append(leaf_nodes[0].batch_item)
    #             advs.append(node.advantage)
        
    #     batch = DataProto.concat(batch_items)
    #     advantages = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    #     for i in range(advantages.size(0)):
    #         advantages[i, range_start[i] : range_end[i]] = advs[i]
        
    #     batch.batch["advantages"] = advantages
    #     batch.batch["returns"] = advantages

    #     return batch

    def collect_batch_data_pruned(self):
        leaf_nodes = [node for node in self.all_nodes if node.is_leaf]
        batch_items = []
        visited_nodes = []

        for node in leaf_nodes:
            if node.advantage == 0:
                continue
            path = node.get_ancestors()[1:] + [node]
            # Find the first unvisited node in the path
            pos = 0
            while pos < len(path) and path[pos] in visited_nodes:
                pos += 1
            assert pos < len(path)

            prefix_len = sum([len(nd.token_sequence) for nd in path[:pos]])
            adv_nodes = path[pos:]
            adv = [0.0] * prefix_len
            for adv_node in adv_nodes:
                adv.extend([adv_node.advantage] * len(adv_node.token_sequence))

            try:
                assert len(adv) <= self.max_response_length, f"Advantage length {len(adv)} exceeds max response length {self.max_response_length}."
            except AssertionError as e:
                breakpoint()

            try:
                adv = torch.tensor([adv], dtype=torch.float32)
            except Exception as e:
                breakpoint()

            adv = pad_sequence_to_length(adv, self.max_response_length, 0.0)

            node.batch_item.batch["advantages"] = adv
            node.batch_item.batch["returns"] = adv
        
            batch_items.append(node.batch_item)
            visited_nodes.extend(adv_nodes)
        try:
            return DataProto.concat(batch_items)
        except Exception as e:
            breakpoint()

    def collect_original_batch_data(self):
        """
        Collect the original batch data from the tree.
        This function collects all the original batch items from the tree, which are stored in the leaf nodes with `is_original_batch_item` set to True.
        """
        original_batch_items = []
        for node in self.all_nodes:
            if node.is_leaf and node.is_original_batch_item:
                original_batch_items.append(node.batch_item)
        
        return DataProto.concat(original_batch_items)

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
                "is_correct_path": node.is_correct_path if node.is_correct_path is not None else "N/A",
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
        
        # 如果模板文件不存在, 提示用户
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

    def compute_total_token_num(self):
        """
        计算树中所有节点的token总数
        """
        total_tokens = 0
        for node in self.all_nodes:
            if not node.is_root:
                total_tokens += len(node.token_sequence)
        return total_tokens

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    score =  _default_compute_score(data_source, solution_str, ground_truth, extra_info)
    
    return score

def generate(actor_rollout_wg, gen_batch, order):
    n = len(gen_batch) * order if gen_batch.meta_info.get("sample_mode", "initial") == "initial" else len(gen_batch) * ((order // 2) - 1)
    gen_batch, _ = pad_dataproto_to_divisor(gen_batch, actor_rollout_wg.world_size)
    gen_batch_output = actor_rollout_wg.generate_sequences_naive(gen_batch)[:n]

    return gen_batch_output


def build_pruned_sampling_trees(batch_dict, actor_rollout_wg, tokenizer, config, reward_fn) -> Tuple[List[SamplingTree], DataProto]:
    start_input_ids = batch_dict["raw_prompt_ids"]
    order = config.actor_rollout_ref.rollout.n
    question_num = len(start_input_ids)
    start_texts = tokenizer.batch_decode(start_input_ids, skip_special_tokens=False)
    root_nodes = [Node(token_sequence=start_input_ids[i], max_children=order, is_root=True, text=start_texts[i]) for i in range(question_num)]

    sampling_trees = []
    for i in range(question_num):
        tree_dict = {}
        for k, v in batch_dict.items():
            tree_dict[k] = v[i : i + 1]

        sampling_tree = SamplingTree(
            root_nodes[i], 
            order,
            config.data.max_prompt_length,
            config.data.max_response_length,
            tree_dict,
            tokenizer,
            config.actor_rollout_ref.rollout
        )
        
        sampling_trees.append(sampling_tree)
    
    batch_list = []
    gen_batch_list = []
    for sampling_tree in sampling_trees:
        tree_batch, tree_gen_batch = sampling_tree.get_batch_and_gen_batch_of_node(sampling_tree.root)
        batch_list.append(tree_batch)
        gen_batch_list.append(tree_gen_batch)

    batch = DataProto.concat(batch_list)
    gen_batch = DataProto.concat(gen_batch_list)

    gen_batch_output = generate(actor_rollout_wg, gen_batch, order)
    raw_responses = gen_batch_output.non_tensor_batch["real_responses"].tolist()
    log_prob = actor_rollout_wg.compute_log_prob(gen_batch_output)
    entropys = log_prob.batch["entropys"]
    raw_entropys = []
    for i in range(entropys.size(0)):
        raw_entropys.append(entropys[i][:len(raw_responses[i])])

    batch = batch.repeat(repeat_times=order, interleave=True)
    batch = batch.union(gen_batch_output)
    reward_result = reward_fn(batch, return_dict=True)
    scores_wo_overlong = reward_result["reward_extra_info"]["score"]
    reward_tensor = reward_result["reward_tensor"]
    batch.batch["token_level_scores"] = reward_tensor
    batch.batch["token_level_rewards"] = reward_tensor

    filtered_sampling_trees = []
    for i in range(question_num):
        s = i * order
        e = (i + 1) * order
        scores = scores_wo_overlong[s : e]
        if np.var(scores) > 0:
            num_nodes = [1 if score == 1.0 else sampling_trees[i].rollout_config.max_tree_depth for score in scores]
            sampling_trees[i].generate_paths(sampling_trees[i].root, batch[s:e], raw_responses[s:e], raw_entropys[s:e], scores_wo_overlong[s:e], num_nodes)
            filtered_sampling_trees.append(sampling_trees[i])
    sampling_trees = filtered_sampling_trees

    finder_stype = config.actor_rollout_ref.rollout.finder_style
    if finder_stype == "redundant":
        find_first_incorrect_step_with_redundant_generation(sampling_trees, tokenizer, config, actor_rollout_wg, reward_fn)
    elif finder_stype == "binary_search":
        find_first_incorrect_step_with_python_generator(sampling_trees, tokenizer, config, actor_rollout_wg)

    return sampling_trees, batch


def find_first_incorrect_step_with_redundant_generation(sampling_trees: List[SamplingTree], tokenizer, config, actor_rollout_wg, reward_fn):
    if not sampling_trees:
        return
    
    order = config.actor_rollout_ref.rollout.n
    batch_list = []
    gen_batch_list = []
    structured_data = defaultdict(lambda: defaultdict(list))
    for sampling_tree in sampling_trees:
        tree_batch, tree_gen_batch, structured_tree_data = sampling_tree.get_redundant_batch_and_gen_batch(sample_mode="intermediate")
        batch_list.append(tree_batch)
        gen_batch_list.append(tree_gen_batch)
        structured_data[sampling_tree] = structured_tree_data
    
    batch = DataProto.concat(batch_list)
    gen_batch = DataProto.concat(gen_batch_list)
    real_prompt_ids = gen_batch.non_tensor_batch["real_prompt_ids"]
    raw_question_ids = gen_batch.non_tensor_batch["raw_question_ids"]

    gen_batch_output = generate(actor_rollout_wg, gen_batch, order)
    real_responses = gen_batch_output.non_tensor_batch["real_responses"].tolist()
    # log_prob = actor_rollout_wg.compute_log_prob(gen_batch_output)
    # entropys = log_prob.batch["entropys"]
    # raw_entropys = []
    # assert entropys.size(0) % (order - 1) == 0, "The number of entropys should be divisible by (order - 1)."
    # for i in range(entropys.size(0)):
    #     s = len(real_prompt_ids[i // (order - 1)]) - len(raw_question_ids[i // (order - 1)])
    #     e = s + len(real_responses[i])
    #     raw_entropys.append(entropys[i][s : e])

    batch = batch.repeat(repeat_times=(order // 2) - 1)
    batch = batch.union(gen_batch_output)
    reward_result = reward_fn(batch, return_dict=True)
    scores_wo_overlong = reward_result["reward_extra_info"]["score"]
    reward_tensor = reward_result["reward_tensor"]
    batch.batch["token_level_scores"] = reward_tensor
    batch.batch["token_level_rewards"] = reward_tensor

    cnt = 0
    unit = (order // 2 - 1)
    for tree, structured_tree_data in structured_data.items():
        for leaf_node, path in structured_tree_data.items():
            find_first_incorrect = False
            for node in path:
                s = cnt * unit
                e = (cnt + 1) * unit
                is_first_incorrect = False
                if not find_first_incorrect and node.is_leaf:
                    is_first_incorrect = True
                elif not find_first_incorrect and not node.is_leaf:
                    is_first_incorrect = all(score == 0 for score in scores_wo_overlong[s : e])

                if is_first_incorrect:
                    # merge the first incorrect node with its following nodes
                    new_node = tree.merge_nodes(node)
                    new_node.is_first_incorrect = True
                    tree.first_incorrect_nodes.append(new_node)
                    find_first_incorrect = True

                if not find_first_incorrect:
                    tree.generate_paths(node, batch[s : e], real_responses[s : e], None, scores_wo_overlong[s : e], 1)
                    for _node in node.children[1:]:
                        _node.is_first_incorrect = not _node.is_correct_path
                        if _node.is_first_incorrect:
                            tree.first_incorrect_nodes.append(_node)
                
                if not node.is_leaf:
                    cnt += 1

    # # after finding the first incorrect steps, we also need to evaluate the correctness of their brother nodes
    # batch_list = []
    # gen_batch_list = []
    # structured_data = defaultdict(list)
    # for sampling_tree in sampling_trees:
    #     for node in sampling_tree.all_nodes:
    #         if node.is_first_incorrect:
    #             for _node in node.parent.children:
    #                 if _node is node:
    #                     continue
    #                 leaf_nodes = sampling_tree.get_all_leaf_nodes_under_a_node(_node)
    #                 are_correct_paths = [leaf_node.is_correct_path for leaf_node in leaf_nodes]
    #                 if any(are_correct_paths):
    #                     # If there is at least one correct path, we do not need to generate this node
    #                     _node.is_first_incorrect = False
                    # if all(not is_correct_path for is_correct_path in are_correct_paths):
                    #     batch, gen_batch = sampling_tree.get_batch_and_gen_batch_of_node(_node, sample_mode="intermediate")
                    #     batch_list.append(batch)
                    #     gen_batch_list.append(gen_batch)
                    #     structured_data[sampling_tree].append(_node)
    
    # batch = DataProto.concat(batch_list)
    # gen_batch = DataProto.concat(gen_batch_list)
    # gen_batch_output = generate(actor_rollout_wg, gen_batch, order)
    # batch = batch.repeat(repeat_times=order - 1)
    # batch = batch.union(gen_batch_output)
    # scores_wo_overlong = scores_wo_overlong = reward_fn(batch, return_dict=True)["reward_extra_info"]["score"]
    # real_responses = gen_batch_output.non_tensor_batch["real_responses"]
    
    # cnt = 0
    # for sampling_tree, nodes in structured_data.items():
    #     for node in nodes:
    #         s = cnt * (order - 1)
    #         e = (cnt + 1) * (order - 1)
    #         if all(score == 0 for score in scores_wo_overlong[s : e]):
    #             node.is_first_incorrect = True
    #         else:
    #             node.is_first_incorrect = False
    #         cnt += 1

def tensor_numpy_to_list(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 将位置参数中的 Tensor 转换为 list
        new_args = [arg.tolist() if isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray) else arg for arg in args]
        # 将关键字参数中的 Tensor 转换为 list
        new_kwargs = {k: v.tolist() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    return wrapper

@tensor_numpy_to_list
def split_list_by_positions(response, positions):
    assert len(set(positions)) == len(positions)
    positions = sorted(positions)
    # 初始化分段起始点
    result = []
    start = 0
    for pos in positions:
        result.append(response[start:pos + 1])
        start = pos + 1
    # 加入最后一段（如果还有剩余）
    if start < len(response):
        result.append(response[start:])
    return result


def find_step_split_token_positions(text, encoding, step_split_str="\n\n"):
    """
    Find positions of the step split token in the input_ids. Note that we cannot directly encode the step_split_str and find the encoding result in input_ids. For example, we seperately encode step_split_str to get step_split_token_ids but this token id sequence may not be in the input_ids, since the input_ids is the result of encoding the whole text. Even if the whole text contains the step_split_str, the tokenization may not match exactly.
    """
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


def get_top_entropy_tokens(entropy: torch.Tensor, response: torch.Tensor, n: int):
    """
    选择 entropy 最高的前 n 个 token, 不包括第一个 token
    
    Args:
        entropy: [response_len] 形状的 tensor, 单个序列的entropy值
        response: 单个序列的 token ids
        tokenizer: 用于解码的 tokenizer
        top_n: 要选择的 top token 数量
    
    Returns:
        result: 包含高entropy token信息的字典
    """
    if n == 0:
        return []
    
    if isinstance(response, list):
        response = torch.tensor(response)
    if isinstance(response, np.ndarray):
        response = torch.tensor(response.tolist())

    response_len = response.shape[0]
    
    # 获取 top_n 个最高 entropy 的索引
    top_indices = torch.topk(entropy, min(n + 1, response_len), dim=0)
    top_entropy_values = top_indices.values  # entropy 值
    top_positions = top_indices.indices      # 位置索引
    
    batch_results = []
    for i in range(len(top_positions)):
        pos = top_positions[i].item()
        entropy_val = top_entropy_values[i].item()
        
        # 获取对应位置的 token id
        if pos > 0:
            try:
                token_id = response[pos].item()
            except IndexError:
                breakpoint()
            
            batch_results.append({
                'position': pos,
                'entropy': entropy_val,
                'token_id': token_id
            })
    
    return batch_results[:n]


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
        task_prefix: 任务前缀, 用于生成唯一的task_id
    
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
        scores = [compute_score(tree.data_source, tokenizer.decode(raw_prompt_ids[i].tolist() + response.raw_responses[i]), tree.ground_truth) for i in range(len(response.raw_responses))]
        raw_responses = response.raw_responses
        all_incorrect = all(score == 0 for score in scores)
        entropys = response.entropys
        
        if all_incorrect:
            # 如果所有采样都是错误的, 说明当前节点是错误的
            # 继续在左半部分搜索
            right = m - 1
        else:
            # 如果存在正确的采样, 说明当前节点是正确的
            paths = generate_paths(tree.data_source, raw_prompt_ids, raw_responses, entropys, tokenizer, config, tree.ground_truth, config.trainer.top_n_entropy_tokens - mid_node.depth)
            tree.add_paths(mid_node, paths)
            left = m + 1
    
    nodes[left].is_first_incorrect = True
    for node in nodes[left].get_ancestors()[1:]:
        node.is_first_incorrect = False

    # 判断第一个错误节点的兄弟节点是否正确, 即兄弟节点是否也是第一个错误节点
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
    scores = [compute_score(tree.data_source, full_text, tree.ground_truth) for full_text in full_texts]
    all_incorrect = all(score == 0 for score in scores)
    
    if all_incorrect:
        # 如果所有采样都是错误的, 标记为first incorrect
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
            # yield所有请求, 等待批量响应
            responses = yield requests
            
            # 将响应分发回对应的生成器
            response_map = {resp.request.task_id: resp for resp in responses}
            
            for task_id, (gen_id, gen, _) in list(pending_requests.items()):
                if task_id in response_map:
                    response = response_map[task_id]
                    del pending_requests[task_id]
                    
                    try:
                        # 将响应发送给生成器, 并获取下一个请求列表
                        next_requests = gen.send([response])
                        for req in next_requests:
                            pending_requests[req.task_id] = (gen_id, gen, req)
                    except StopIteration:
                        # 该生成器已完成
                        pass


class ParallelIncorrectStepFinder:
    """
    并行查找多个采样树的第一个错误步骤
    对于每个采样树的每个错误分支, 通过二分法查找第一个错误步骤
    由于每个错误分支的查找次数可能不同, 因此使用生成器来处理
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
                if compute_score(sampling_tree.data_source, node.state[1], sampling_tree.ground_truth) == 0
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

            # 遍历目录, 查找符合模式的子目录
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
        gen_batch_output = self.actor_rollout_wg.generate_sequences_naive(gen_batch)
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