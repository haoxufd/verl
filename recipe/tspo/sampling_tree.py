from git import Optional
import torch
import numpy as np

from verl import DataProto

import json
from pathlib import Path
from typing import Dict, Any, List

import os

from verl.protocol import collate_fn
from verl.trainer.ppo.ray_trainer import compute_response_mask

from verl.protocol import DataProtoItem

import uuid
import re

import math

def split_integer(a: int, x: int) -> list[int]:
    base = a // x
    remainder = a % x
    result = [base] * x
    for i in range(remainder):
        result[i] += 1
    return result

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
                 max_children: int,
                 log_prob: List[float] | None=None,
                 entropy: List[float] | None=None) -> None:
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
        self.leaf_node_num : Optional[int] = None
        self.log_prob = log_prob
        if self.log_prob is not None:
            self.perplexity = math.exp(-sum(self.log_prob) / len(self.log_prob))
        self.entropy = entropy
        if self.entropy is not None:
            self.avg_entropy = sum(self.entropy) / len(self.entropy)
    
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
        
        if self.order_list is not None and self.max_depth is not None:
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
        log_probs = gen_batch_output_item.batch["rollout_log_probs"][real_response_start : response_len].tolist()
        entropys = gen_batch_output_item.batch["entropys"][real_response_start : response_len].tolist()
        
        if self.step_split_mode == "average":
            segment_len = len(response) // max_added_depth
            split_positions = [segment_len * i for i in range(1, max_added_depth)]
        elif self.step_split_mode == "semantic":
            step_start_positions = find_step_start_token_positions(response, tokenizer, self.step_split_pattern)
            num_steps = len(step_start_positions) if len(step_start_positions) > 0 else 1
            if num_steps > max_added_depth:
                num_steps_per_depth = split_integer(num_steps, max_added_depth)
                split_positions = [sum(num_steps_per_depth[:i]) for i in range(1, max_added_depth)]
                split_positions = [step_start_positions[idx] for idx in split_positions]
            else:
                split_positions = step_start_positions[1:] if len(step_start_positions) > 1 else []
        else:
            raise NotImplementedError("No other step split modes except for average and semantic")
        
        added_depth = len(split_positions) + 1
        segments = split_list_by_positions(response, split_positions)
        log_prob_segments = split_list_by_positions(log_probs, split_positions)
        entropys_segments = split_list_by_positions(entropys, split_positions)
        path = []
        for i in range(1, added_depth + 1):
            max_children = self.order_list[node.depth + i] if i < added_depth else 0
            path.append(Node(segments[i - 1], max_children, log_prob_segments[i - 1], entropys_segments[i - 1]))

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
        gen_batch_output = collate_fn(gen_batch_output_items)

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
            node.leaf_node_num = 1
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
        node.leaf_node_num = sum(leaf_node_nums)

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
        if node.parent is not None:
            assert node.score is not None and node.parent.score is not None and self.root.score is not None
            assert node.leaf_node_num is not None
            node.advantage = (node.score - node.parent.score + node.score - self.root.score) / math.sqrt(node.leaf_node_num)
        
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
                advantage.extend([node.advantage for _ in range(len(node.token_sequence))])
            
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
    
    def dump_json(self, output_file):
        tree_data = self._tree_to_json()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls,
                  json_file: str) -> "SamplingTree":
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        tree_data = data["tree_data"]

        def build_node(node_dict, parent=None):
            node = Node(
                token_sequence=node_dict["token_sequence"],
                max_children=len(node_dict["children"]) if not node_dict["is_leaf"] else 0,
                log_prob=node_dict.get("log_prob"),
                entropy=node_dict.get("entropy")
            )
            node.text = node_dict.get("full_text")
            node.depth = node_dict.get("depth")
            node.parent = parent
            node.score = node_dict.get("score")
            node.advantage = node_dict.get("advantage")
            node.acc = node_dict.get("acc")
            node.perplexity = node_dict.get("perplexity")
            node.avg_entropy = node_dict.get("avg_entropy")
            node.is_root = node_dict.get("is_root", False)
            node.is_leaf = node_dict.get("is_leaf", False)

            # 递归处理子节点
            for child_dict in node_dict["children"]:
                child_node = build_node(child_dict, parent=node)
                node.children.append(child_node)

            node.children_is_full = (len(node.children) >= node.max_children)
            return node

        root_node = build_node(tree_data)

        tree = cls(
            root_node=root_node,
            order_list=None,
            max_depth=None,
            align_depth=None,
            question_info=None,
            max_prompt_length=None,
            max_response_length=None,
            pad_token_id=None,
            step_split_mode=None,
            step_split_pattern=None,
        )

        # 补全 all_nodes
        def collect_all_nodes(node):
            nodes = [node]
            for child in node.children:
                nodes.extend(collect_all_nodes(child))
            return nodes

        tree.all_nodes = collect_all_nodes(root_node)
        tree.leaf_nodes = [node for node in tree.all_nodes if node.children == []]

        return tree

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
                "perplexity": node.perplexity if hasattr(node, 'perplexity') else None,
                "avg_entropy": node.avg_entropy if hasattr(node, 'avg_entropy') else None,
                "log_prob": node.log_prob if node.log_prob is not None else None,
                "entropy": node.entropy if node.entropy is not None else None,
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
        
        tree_json = json.dumps(tree_data, ensure_ascii=False, indent=2)
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

        gen_batch = DataProto.concat(gen_batch_list)
        sampling_points = gen_batch.pop(non_tensor_batch_keys=["sampling_points"]).non_tensor_batch["sampling_points"].tolist()

        world_size = actor_rollout_wg.world_size
        padding_size = world_size - len(gen_batch) % world_size if len(gen_batch) % world_size != 0 else 0
        gen_batch.padding(padding_size, "last")
        gen_batch_output = actor_rollout_wg.generate_sequences_tspo(gen_batch)

        log_probs = actor_rollout_wg.compute_log_prob(gen_batch_output)
        entropys = log_probs.batch["entropys"]
        gen_batch_output.batch["entropys"] = entropys

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