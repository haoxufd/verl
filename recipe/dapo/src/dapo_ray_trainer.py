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

from collections import defaultdict
from pprint import pprint
import time

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
)

from verl.protocol import pad_dataproto_to_divisor

# from dapo.recipe.dapo.sampling_tree import build_pruned_sampling_trees
from ..sampling_tree import build_pruned_sampling_trees

import os

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        self.data_load_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.data_load_steps += 1
        last_val_metrics = None
        
        timing_raw = defaultdict(float)
        batch = None
        batch_size = 0
        # needed_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.trainer.update_times_per_call
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        sampling_trees, original_batch = build_pruned_sampling_trees(batch_dict, self.actor_rollout_wg, self.tokenizer, self.config, self.reward_fn)
                    
                    with _timer('adv', timing_raw):
                        for tree in sampling_trees:
                            tree.compute_scores_pruned()
                            tree.compute_advantages_pruned()
                    
                    with _timer('print_tree', timing_raw):
                        if self.config.trainer.output_sampling_tree:
                            for i, sampling_tree in enumerate(sampling_trees):
                                step_dir = os.path.join(self.config.trainer.sampling_tree_dir, f"step_{self.global_steps}")
                                sampling_tree.visualize(output_file=os.path.join(step_dir, f"tree_{i}.html"))

                    with _timer('collect', timing_raw):
                        batch_list = []
                        # original_batch_list = []
                        for tree in sampling_trees:
                            #original_batch_list.append(tree.collect_original_batch_data())
                            batch_list.append(tree.collect_batch_data_pruned())
                        batch = DataProto.concat(batch_list)
                        # original_batch = DataProto.concat(original_batch_list)

                    # if batch is None:
                    #     # first batch, initialize the batch
                    #     batch = new_batch
                    # else:
                    #     if new_batch is not None:
                    #         batch = DataProto.concat([batch, new_batch])
                    
                    batch_size = len(batch)
                    rest = batch_size % self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    update_times = batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    batch = batch[:batch_size - rest] if rest > 0 else batch
                    metrics['update/collected_batch_size'] = batch_size
                    metrics['update/mini_batch_size'] = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    metrics['update/update_times'] = update_times
                    # batch_size = len(batch) if batch is not None else 0
                    # if batch_size < needed_batch_size:
                    #     if is_last_step:
                    #         self._save_checkpoint()
                    #     # if the batch size is not enough, continue to the next batch
                    #     print(f"Not enough batch size {batch_size} (need {needed_batch_size}), entering next global step for more data...")
                    #     progress_bar.update(1)
                    #     self.data_load_steps += 1
                    #     continue
                    # else:
                    #     # needed_batch = batch[:needed_batch_size]
                    #     # left_batch = batch[needed_batch_size:]
                    #     # batch = needed_batch
                    #     print(f"Batch size {batch_size} is enough. Keeping the first {needed_batch_size} pieces and discarding the rest {batch_size - needed_batch_size}.")
                    #     batch = batch[:needed_batch_size]

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    global_token_num = 0
                    for tree in sampling_trees:
                        global_token_num += tree.compute_total_token_num()
                    batch.meta_info['global_token_num'] = global_token_num

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=original_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=original_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                # n_gpus = self.resource_pool_manager.get_n_gpus()
                # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # batch = left_batch
                batch = None

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                # self.data_load_steps += 1
