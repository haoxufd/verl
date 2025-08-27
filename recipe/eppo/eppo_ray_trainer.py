# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
from pydoc import text
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

import os


class RayEPPOTrainer(RayDAPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.gen_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        first_gen_batch_output = self.actor_rollout_wg.generate_sequences_eppo(gen_batch)
                        timing_raw.update(first_gen_batch_output.meta_info["timing"])
                    
                    if self.config.actor_rollout_ref.rollout.calculate_log_probs:
                        new_batch.batch["rollout_log_probs"] = first_gen_batch_output.batch["rollout_log_probs"]
                    
                    with marked_timer("entropy", timing_raw, "red"):
                        log_prob = self.actor_rollout_wg.compute_log_prob(first_gen_batch_output)
                        entropys = log_prob.batch["entropys"]
                    
                    rollout_n = self.config.actor_rollout_ref.rollout.n
                    num_entropy_points = self.config.actor_rollout_ref.rollout.top_entropy
                    rollout_N = rollout_n * num_entropy_points

                    # find top entropy points as sampling positions
                    response_len = compute_response_mask(first_gen_batch_output).sum(dim=-1).tolist()
                    if not self.config.actor_rollout_ref.rollout.get("group_entropy", True):
                        indices = torch.topk(entropys, k=num_entropy_points, dim=-1).indices
                    else:
                        window = self.config.actor_rollout_ref.rollout.get("entropy_window_size", 100)
                        shift_size = self.config.actor_rollout_ref.rollout.get("window_shift_size", 50)
                        indices = []

                        for i in range(entropys.shape[0]):
                            candidate_sampling_points = [0]
                            last_possible_sampling_point = response_len[i] - window
                            for j in range(shift_size, last_possible_sampling_point + 1, shift_size):
                                if j + window <= response_len[i]:
                                    candidate_sampling_points.append(j)
                                else:
                                    break

                            if len(candidate_sampling_points) < num_entropy_points:
                                index = candidate_sampling_points + [candidate_sampling_points[-1]] * (num_entropy_points - len(candidate_sampling_points))
                                indices.append(index)
                                continue
                            
                            last_point = None
                            for point in candidate_sampling_points:
                                entropys[i, point] = torch.mean(entropys[i, point : min(point + window, response_len[i])])
                                if last_point is not None:
                                    entropys[i, last_point + 1 : point] = 0
                                last_point = point
                            assert last_point is not None
                            entropys[i, min(last_point + 1, response_len[i]) :] = 0

                            index = torch.topk(entropys[i], k=num_entropy_points, dim=-1).indices.tolist()
                            indices.append(index)
                        
                        indices = torch.tensor(indices, device=entropys.device) # TODO: device?

                    # record high-entropy tokens
                    all_info = []
                    for i in range(indices.shape[0]):
                        info = {
                            "position": indices[i].tolist(),
                            "relative_pos": (indices[i] / (response_len[i] - 1)).tolist(),
                        }

                        entropy_list = []
                        text_list = []
                        for j in range(num_entropy_points):
                            point = indices[i][j].item()
                            entropy = entropys[i][point].item()
                            entropy_list.append(entropy)
                            token = first_gen_batch_output.batch["responses"][i][point].item()
                            tokens_ahead = first_gen_batch_output.batch["responses"][i][point + 1 : min(point + 50, response_len[i])].tolist()
                            tokens_back = first_gen_batch_output.batch["responses"][i][max(0, point - 50) : point].tolist()
                            text_ahead = self.tokenizer.decode(tokens_ahead, skip_special_tokens=True)
                            text_back = self.tokenizer.decode(tokens_back, skip_special_tokens=True)
                            text = self.tokenizer.decode([token], skip_special_tokens=True)
                            concat_text = text_back + "<<<<<<" + text + ">>>>>>" + text_ahead
                            text_list.append(concat_text)
                        
                        info["entropy"] = entropy_list
                        info["text"] = text_list

                        all_info.append(info)
                    
                    high_entropy_token_dir = self.config.trainer.get("high_entropy_token_dir", None)
                    if high_entropy_token_dir is not None:
                        self._dump_high_entropy_tokens(all_info, high_entropy_token_dir)

                    response_prefix = []
                    for i in range(indices.shape[0]):
                        for j in range(num_entropy_points):
                            num_samples = rollout_n - 1 if j == 0 else rollout_n
                            response_prefix.extend([first_gen_batch_output.batch["responses"][i][:indices[i][j]].tolist()] * num_samples)
                    gen_batch = gen_batch.repeat_with_delta_raw_prompt_ids(response_prefix, repeat_times=rollout_N - 1, interleave=True)

                    with marked_timer("gen", timing_raw, "red"):
                        second_gen_batch_output = self.actor_rollout_wg.generate_sequences_eppo(gen_batch)
                        timing_raw.update(second_gen_batch_output.meta_info["timing"])
                        second_gen_batch_output.meta_info.pop("timing", None)

                    gen_batch_output = DataProto.concat([first_gen_batch_output[:1], second_gen_batch_output[:rollout_N - 1]])
                    for i in range(1, len(first_gen_batch_output)):
                        delta_gen_batch_output = DataProto.concat([first_gen_batch_output[i:i+1], second_gen_batch_output[i * (rollout_N - 1) : (i + 1) * (rollout_N - 1)]])
                        gen_batch_output = DataProto.concat([gen_batch_output, delta_gen_batch_output])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences_eppo(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # first repeat to align with the number of entropy points
                    new_batch = new_batch.repeat(repeat_times=num_entropy_points, interleave=True)
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # record the real start positions of the responses in new_batch
                    new_batch.non_tensor_batch["response_start_positions"] = np.array(indices[:, :num_entropy_points].reshape(-1).tolist())

                    # second repeat to align with the number of samples per point
                    new_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        if self.config.trainer.align_batch:
                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz * num_entropy_points:
                                print(f"{num_prompt_in_batch=} < {prompt_bsz * num_entropy_points=}")
                                max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"{num_gen_batches=}. Keep generating...")
                                    progress_bar.update(1)
                                    self.gen_steps += 1
                                    continue
                                else:
                                    raise ValueError(
                                        f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                        + " Generated too many. Please check if your data are too difficult."
                                        + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                    )
                            else:
                                # Align the batch
                                traj_bsz = self.config.data.train_batch_size * rollout_N
                                original_traj_bsz = len(batch)
                                batch = batch[:traj_bsz]
                                aligned_batch_size = len(batch)
                                print(f"Aligned batch size: {aligned_batch_size} from original {original_traj_bsz}")
                        else:
                            rest = len(batch) % self.actor_rollout_wg.world_size
                            if rest != 0:
                                batch = batch[: len(batch) - rest]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    n, m = batch.batch["response_mask"].shape
                    cols = torch.arange(m).expand(n, m)
                    mask = cols < torch.from_numpy(batch.non_tensor_batch["response_start_positions"]).unsqueeze(1)
                    batch.batch["process_response_mask"] = batch.batch["response_mask"].clone()
                    batch.batch["process_response_mask"][mask] = 0

                    # Log rollout generations if enabled
                    # This should be before the balancing step, or the orders of batch and reward_extra_info_dict would dismatch.
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                metrics["train/batch_size"] = len(batch)
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

    def _dump_high_entropy_tokens(self, high_entropy_token_info, dump_path):
        """Dump high-entropy tokens as json."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.gen_steps}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(high_entropy_token_info, f, ensure_ascii=False, indent=2)

        print(f"Dumped high-entropy tokens to {filename}")