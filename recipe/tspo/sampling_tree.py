from git import Optional
import torch
import numpy as np

from verl import DataProto

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import os

from verl.protocol import collate_fn
from verl.trainer.ppo.ray_trainer import compute_response_mask

from verl.protocol import DataProtoItem

import uuid
import re


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
                 token_sequence: List[int], 
                 max_children: int) -> None:
        self.token_sequence = token_sequence
        self.max_children = max_children
        self.text : Optional[str] = None
        self.token_state : Optional[List[int]] = None
        self.text_state: Optional[str] = None
        self.depth : Optional[int] = None
        self.parent : Optional[Node] = None
        self.children : List[Node] = []
        self.children_is_full : bool = len(self.children) >= self.max_children
        self.score : Optional[float] = None
        self.advantage : Optional[float] = None
        self.gen_batch_output_item : Optional[DataProtoItem] = None
        self.acc : Optional[bool] = None
        self.is_visited : bool = False
    
    def add_child(self, child_node):
        # Check if the node can accept more children
        if len(self.children) >= self.max_children:
            raise ChildrenFullError(f"Cannot add child to node since children are full.")
        
        # Add the child
        self.children.append(child_node)
        child_node.parent = self
        assert self.token_state is not None, "Token state must be set before adding children."
        child_node.token_state = self.token_state + child_node.token_sequence

        # Check if self's children are full
        if len(self.children) >= self.max_children:
            self.children_is_full = True

        assert self.depth is not None, "Node's depth must be set before adding children"
        child_node.depth = self.depth + 1
    
    def get_ancestors(self, include_root=False, include_self=True):
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
        if not include_root:
            ancestors = ancestors[1:]
        if include_self:
            ancestors.append(self)
        
        return ancestors


class SamplingTree:
    def __init__(self, root_node: Node, order_list: List[int], max_depth: int, align_depth: bool, question_info: DataProto, max_prompt_length: int, max_response_length: int, pad_token_id: int, step_split_mode: str, step_split_pattern: str | None=None) -> None:
        self.root = root_node
        self.order_list = order_list
        self.max_depth = max_depth
        self.align_depth = align_depth
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.question_info = question_info
        self.pad_token_id = pad_token_id
        self.step_split_mode = step_split_mode
        self.step_split_pattern = step_split_pattern
        self.all_nodes = [root_node]
        self.batch : Optional[DataProto] = None
        self.leaf_nodes : Optional[List[Node]] = None
        
        assert len(self.order_list) == self.max_depth
    
    def add_node(self, node1: Node, node2: Node):
        node1.add_child(node2)
        self.all_nodes.append(node2)
    
    def add_path(self, node: Node, path: list[Node]):
        parent_node = node
        for nd in path:
            self.add_node(parent_node, nd)
            parent_node = nd
    
    def add_response(self, node: Node, gen_batch_output_item: DataProtoItem, tokenizer):
        assert node.depth is not None
        max_added_depth = self.max_depth - node.depth

        response_len = compute_response_mask(collate_fn([gen_batch_output_item])).sum(dim=-1).tolist()[0]
        assert node.token_state is not None
        real_response_start = len(node.token_state) - len(self.root.token_sequence)
        response = gen_batch_output_item.batch["responses"][real_response_start : response_len].tolist()
        
        if self.step_split_mode == "average":
            segment_len = len(response) // max_added_depth
            split_positions = [segment_len * i for i in range(1, max_added_depth)]
        elif self.step_split_mode == "semantic":
            split_positions = find_step_start_token_positions(response, tokenizer, self.step_split_pattern)
            if len(split_positions) > 0:
                split_positions = split_positions[1:]
            if len(split_positions) + 1 > max_added_depth:
                num_kept_positions = max_added_depth - 1
                if num_kept_positions > 0:
                    delta = len(split_positions) // num_kept_positions
                    kept_positions = [split_positions[i * delta] for i in range(num_kept_positions)]
                    split_positions = kept_positions
                else:
                    split_positions = []
            if len(split_positions) + 1 < max_added_depth and self.align_depth:
                start_pos = 0 if len(split_positions) == 0 else split_positions[-1] + 1
                delta = (len(response) - start_pos) // max_added_depth
                num_to_be_added = max_added_depth - (len(split_positions) + 1)
                for i in range(1, num_to_be_added + 1):
                    split_positions.append(start_pos + delta * i)
        else:
            raise NotImplementedError("No other step split modes except for average and semantic")
        
        added_depth = len(split_positions) + 1
        segments = split_list_by_positions(response, split_positions)
        path = []
        for i in range(1, added_depth + 1):
            max_children = self.order_list[node.depth + i] if i < added_depth else 0
            path.append(Node(segments[i - 1], max_children))

        gen_batch_output_item.non_tensor_batch["real_response_length"] = len(response)
        
        path[-1].gen_batch_output_item = gen_batch_output_item

        self.add_path(node, path)
    
    def collect_gen_batch(self, depth: int):
        order = self.order_list[depth]
        nodes = [node for node in self.all_nodes if node.depth == depth and (len(node.children) > 0 or depth == 0)]

        if len(nodes) == 0:
            return None

        batch = self.question_info.repeat(len(nodes)) # TODO: Check whether they refer to the same object
        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],)
        gen_batch.non_tensor_batch["sampling_points"] = np.array([(self, node) for node in nodes], dtype=object)
        
        for i in range(len(nodes)):
            gen_batch.non_tensor_batch["raw_prompt_ids"][i] = nodes[i].token_state # TODO: Check data type of non_tensor_batch["raw_prompt_ids"]
        
        repeat_times = order if depth == 0 else order - 1
        gen_batch = gen_batch.repeat(repeat_times, interleave=True)
        
        return gen_batch
    
    def record_leaf_nodes(self):
        self.leaf_nodes = [node for node in self.all_nodes if node.children == []]
    
    def _collect_batch(self):
        assert self.leaf_nodes is not None
        gen_batch_output_items = [node.gen_batch_output_item for node in self.leaf_nodes]
        try:
            gen_batch_output = collate_fn(gen_batch_output_items)
        except AttributeError as e:
            breakpoint()

        batch = self.question_info
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4())])
        batch = batch.repeat(repeat_times=len(self.leaf_nodes))
        batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],)

        batch = batch.union(gen_batch_output)
        batch.batch["response_mask"] = compute_response_mask(batch)

        self.batch = batch
    
    def _compute_scores(self, node: Node):
        if node.children == []:
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
    
    def compute_scores(self, reward_fn):
        """
        Compute scores for all nodes in the tree.
        """
        self._collect_batch()
        assert self.batch is not None
        assert self.leaf_nodes is not None

        reward_result = reward_fn(self.batch, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        self.batch.batch["token_level_scores"] = reward_tensor
        self.batch.batch["token_level_rewards"] = reward_tensor
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        if reward_extra_infos_dict:
            self.batch.non_tensor_batch.update(
                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
            )
        
        leaf_scores = reward_tensor.sum(dim=-1).tolist()
        for i, node in enumerate(self.leaf_nodes):
            node.score = leaf_scores[i]
            node.acc = bool(self.batch.non_tensor_batch["acc"][i])

        self._compute_scores(self.root)
    
    def _compute_advantages(self, node: Node):
        if node.children == []:
            return
        
        assert node.children_is_full

        if len(node.children) == 1:
            node.children[0].advantage = 0
        else:
            scores = np.array([_node.score for _node in node.children])
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            advantages = ((scores - mean) / (std + 1e-6)).tolist()

            for i, child in enumerate(node.children):
                child.advantage = advantages[i]

        for i, child in enumerate(node.children):
            child.advantage = advantages[i]
        
        for child in node.children:
            self._compute_advantages(child)
        
    def compute_advantages(self):
        """
        Compute advantages for all nodes in the tree.
        """
        self._compute_advantages(self.root)

        assert self.batch is not None
        assert self.leaf_nodes is not None

        advantages = torch.zeros_like(self.batch.batch["responses"], dtype=self.batch.batch["token_level_scores"].dtype)
        response_len = self.batch.batch["response_mask"].sum(dim=-1).tolist()
        for i, leaf_node in enumerate(self.leaf_nodes):
            path = leaf_node.get_ancestors()

            advantage = []
            for node in path:
                adv = 0 if node.is_visited else node.advantage
                advantage.extend([adv for _ in range(len(node.token_sequence))])
                node.is_visited = True
            
            assert len(advantage) == response_len[i]
            
            advantages[i][:len(advantage)] = torch.tensor(advantage)
        
        self.batch.batch["advantages"] = advantages
        self.batch.batch["returns"] = advantages

    def collect_batch(self):
        if self.batch is None:
            self._collect_batch()

        return self.batch
    
    def store_plain_text_content(self, tokenizer):
        for node in self.all_nodes:
            node.text = tokenizer.decode(node.token_sequence)
            node.text_state = tokenizer.decode(node.token_state)

    def visualize(self, output_file):
        tree_data = self._tree_to_json()
        
        html_content = self._generate_html_from_template(tree_data)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _tree_to_json(self) -> Dict[str, Any]:
        def node_to_dict(node):
            node_data = {
                "score": round(node.score, 4) if node.score is not None else None,
                "advantage": round(node.advantage, 4) if node.advantage is not None else None,
                "full_text": node.text,
                "token_sequence": node.token_sequence,
                "depth": node.depth,
                "is_root": node.parent is None,
                "is_leaf": node.children == [],
                "acc": node.acc,
                "children": []
            }
            
            # 递归处理子节点
            for child in node.children:
                child_data = node_to_dict(child)
                node_data["children"].append(child_data)
            
            return node_data
        
        root_data = node_to_dict(self.root)
        
        # 添加树的元信息
        tree_info = {
            "tree_data": root_data,
            "meta_info": {
                "ground_truth": self.question_info.non_tensor_batch["reward_model"][0]["ground_truth"],
            }
        }
        
        return tree_info
    
    def _generate_html_from_template(self, tree_data: Dict[str, Any]) -> str:
        template_dir = Path(__file__).parent / "templates"
        template_path = template_dir / "tree_template.html"
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template file not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        try:
            tree_json = json.dumps(tree_data, ensure_ascii=False, indent=2)
        except TypeError as e:
            breakpoint()
        html_content = template.replace('{tree_json}', tree_json)
        
        return html_content


def build_sampling_trees(batch_dict, actor_rollout_wg, tokenizer, config):
    question_token_ids = batch_dict["raw_prompt_ids"]
    order_list = config.actor_rollout_ref.sampling_tree.tree_order_list
    max_depth = config.actor_rollout_ref.sampling_tree.tree_max_depth
    batch_size = len(question_token_ids)

    root_nodes = [Node(token_sequence=question_token_ids[i], max_children=order_list[0]) for i in range(batch_size)]
    for node in root_nodes:
        node.token_state = node.token_sequence
        node.depth = 0

    batch = DataProto.from_single_dict(batch_dict)
    assert isinstance(batch, DataProto)

    sampling_trees = [
        SamplingTree(
            root_nodes[i],
            order_list,
            max_depth,
            config.actor_rollout_ref.sampling_tree.align_depth,
            batch[i:i+1],
            config.data.max_prompt_length,
            config.data.max_response_length,
            tokenizer.pad_token_id,
            config.algorithm.step_split_mode,
            config.actor_rollout_ref.sampling_tree.step_split_pattern) for i in range(batch_size)
    ]

    for d in range(max_depth):
        gen_batch_list = []
        for i in range(batch_size):
            gen_batch_of_tree = sampling_trees[i].collect_gen_batch(depth=d)
            if gen_batch_of_tree is not None:
                gen_batch_list.append(gen_batch_of_tree)
        
        if len(gen_batch_list) == 0:
            break
        
        try:
            gen_batch = DataProto.concat(gen_batch_list)
        except ValueError as w:
            breakpoint()
        sampling_points = gen_batch.pop(non_tensor_batch_keys=["sampling_points"]).non_tensor_batch["sampling_points"].tolist()

        world_size = actor_rollout_wg.world_size
        padding_size = world_size - len(gen_batch) % world_size
        gen_batch.padding(padding_size, "last")
        gen_batch_output = actor_rollout_wg.generate_sequences_tspo(gen_batch)
        gen_batch_output = gen_batch_output[: len(gen_batch_output) - padding_size]

        for i in range(len(gen_batch_output)):
            tree, node = sampling_points[i]
            tree.add_response(node, gen_batch_output[i], tokenizer)
    
    for tree in sampling_trees:
        tree.record_leaf_nodes()

    return sampling_trees


def split_list_by_positions(lst, positions):
    result = []
    start = 0
    for pos in positions:
        result.append(lst[start:pos])
        start = pos
    result.append(lst[start:])  # 最后一段
    return result


def find_step_start_token_positions(token_ids, tokenizer, step_start_pattern):
    text = tokenizer.decode(token_ids)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    matches = [(m.group(), m.start(), m.end()) for m in re.finditer(step_start_pattern, text)]
    
    step_start_token_positions = []
    start_idx = 0
    for _, start, end in matches:
        for i in range(start_idx, len(encoding.offset_mapping)):
            if encoding.offset_mapping[i][0] <= start and encoding.offset_mapping[i][1] > start:
                step_start_token_positions.append(i)
                start_idx = i + 1
                break
    
    assert len(step_start_token_positions) == len(matches)
    
    return step_start_token_positions