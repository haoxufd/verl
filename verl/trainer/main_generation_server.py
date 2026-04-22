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
Generate responses given a dataset of prompts
"""

import os
import json
from glob import glob

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int_list(value: object) -> list[int]:
    if value is None:
        return []
    from omegaconf import ListConfig

    if isinstance(value, ListConfig):
        return [int(x) for x in value]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        if "," in stripped:
            return [int(x.strip()) for x in stripped.split(",") if x.strip()]
        return [int(stripped)]
    return [int(value)]


def _get_stop_token_ids(model_path: str, extra_ids: list[int] | None = None) -> list[int]:
    """Return tokenizer EOS token IDs plus optional extra stop token IDs."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, list):
            ids.update(int(x) for x in eos)
        else:
            ids.add(int(eos))
    if extra_ids:
        ids.update(int(x) for x in extra_ids)
    result = sorted(ids)
    print(f"[stop_token_ids] model={model_path!r} -> {result}")
    return result


async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    try:
        extra_headers = chat_complete_request.pop("extra_headers", {})
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


async def generate_per_replica(server_address, model_path: str, n_samples: int, sampling_params: dict, chat_lst: list):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    tasks = [submit_request(server_address, **req) for req in chat_complete_request]
    results = await asyncio.gather(*tasks)
    return results


async def generate(
    server_addresses: list, model_path: str, n_samples: int, sampling_params: dict, chat_numpy: np.ndarray
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    results = await asyncio.gather(
        *[
            generate_per_replica(server_addresses[i], model_path, n_samples, sampling_params, chat_sub_array[i])
            for i in range(num_replicas)
        ]
    )
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}}, address="auto")

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    extra_stop_ids = _as_int_list(config.data.get("extra_stop_token_ids", []))
    stop_token_ids = _get_stop_token_ids(config.actor_rollout_ref.model.path, extra_stop_ids or None)
    skip_special_tokens = _as_bool(config.data.get("skip_special_tokens", True), default=True)

    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        # "top_k": config.actor_rollout_ref.rollout.top_k,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
        "skip_special_tokens": skip_special_tokens,
        "stop_token_ids": stop_token_ids,
    }

    from omegaconf import ListConfig

    train_files = config.data.train_files
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

    datasets = []
    for train_file in train_files:
        dataset = pd.read_parquet(train_file)
        datasets.append(dataset)

    # concat dataset
    dataset = pd.concat(datasets, axis=0, ignore_index=True)
    prompt_series = dataset[config.data.prompt_key]
    total_rows = len(dataset)

    # start native server
    server_handles, server_addresses = asyncio.run(start_server(config))

    gen_chunk_size = int(config.data.get("gen_chunk_size", 0))
    if gen_chunk_size <= 0:
        # backward-compatible mode: one-shot generate + one-shot write
        chat_lst = prompt_series.tolist()
        chat_lst = [chat.tolist() for chat in chat_lst]
        chat_numpy = np.array(chat_lst, dtype=object)

        gen_results = asyncio.run(
            generate(server_addresses, config.actor_rollout_ref.model.path, n_samples, sampling_params, chat_numpy)
        )

        import itertools

        results = list(itertools.chain.from_iterable(gen_results))
        responses = np.array([result.choices[0].message.content for result in results], dtype=object)
        finish_reasons = np.array([result.choices[0].finish_reason for result in results], dtype=object)

        responses = np.reshape(responses, (-1, n_samples))
        finish_reasons = np.reshape(finish_reasons, (-1, n_samples))
        assert responses.shape == (len(chat_lst), n_samples)
        assert finish_reasons.shape == (len(chat_lst), n_samples)

        dataset["responses"] = responses.tolist()
        dataset["finish_reasons"] = finish_reasons.tolist()

        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        print(f"Saving results to {config.data.output_path}")
        dataset.to_parquet(config.data.output_path)
        return

    # chunked mode: generate and write incrementally
    resume = bool(config.data.get("resume_from_progress", True))
    merge_output = bool(config.data.get("merge_chunked_output", False))
    output_path = config.data.output_path
    chunk_dir = output_path if not output_path.endswith(".parquet") else f"{output_path}.chunks"
    makedirs(chunk_dir, exist_ok=True)

    output_parent = os.path.dirname(output_path)
    if output_parent:
        makedirs(output_parent, exist_ok=True)

    progress_path = os.path.join(chunk_dir, "_progress.json")
    meta_path = os.path.join(chunk_dir, "_meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_files": [str(x) for x in train_files],
                    "output_path": output_path,
                    "chunk_dir": chunk_dir,
                    "total_rows": total_rows,
                    "chunk_size": gen_chunk_size,
                    "n_samples": int(n_samples),
                    "prompt_key": str(config.data.prompt_key),
                },
                f,
                ensure_ascii=True,
                indent=2,
            )

    done_until = 0
    if resume and os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                done_until = int(json.load(f).get("done_until", 0))
        except Exception:
            done_until = 0

    import itertools

    print(f"Chunked generation enabled: total_rows={total_rows}, chunk_size={gen_chunk_size}, chunk_dir={chunk_dir}")
    for start in range(done_until, total_rows, gen_chunk_size):
        end = min(start + gen_chunk_size, total_rows)
        part_path = os.path.join(chunk_dir, f"part-{start:012d}-{end:012d}.parquet")

        if resume and os.path.exists(part_path):
            done_until = end
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"done_until": done_until}, f, ensure_ascii=True)
            print(f"Skip existing chunk [{start}, {end})")
            continue

        chunk_df = dataset.iloc[start:end].copy()
        chat_lst = [chat.tolist() for chat in prompt_series.iloc[start:end].tolist()]
        chat_numpy = np.array(chat_lst, dtype=object)

        gen_results = asyncio.run(
            generate(server_addresses, config.actor_rollout_ref.model.path, n_samples, sampling_params, chat_numpy)
        )
        results = list(itertools.chain.from_iterable(gen_results))

        responses = np.array([result.choices[0].message.content for result in results], dtype=object)
        finish_reasons = np.array([result.choices[0].finish_reason for result in results], dtype=object)
        responses = np.reshape(responses, (-1, n_samples))
        finish_reasons = np.reshape(finish_reasons, (-1, n_samples))
        assert responses.shape == (len(chunk_df), n_samples)
        assert finish_reasons.shape == (len(chunk_df), n_samples)

        chunk_df["responses"] = responses.tolist()
        chunk_df["finish_reasons"] = finish_reasons.tolist()
        chunk_df["_row_idx"] = list(range(start, end))
        chunk_df.to_parquet(part_path, index=False)

        done_until = end
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"done_until": done_until}, f, ensure_ascii=True)
        print(f"Saved chunk [{start}, {end}) to {part_path}")

    if merge_output:
        part_files = sorted(glob(os.path.join(chunk_dir, "part-*.parquet")))
        if len(part_files) == 0:
            raise RuntimeError(f"No part files found in {chunk_dir}")
        merged = pd.concat([pd.read_parquet(f) for f in part_files], axis=0, ignore_index=True)
        merged = merged.sort_values("_row_idx").drop(columns=["_row_idx"])
        merged.to_parquet(output_path, index=False)
        print(f"Merged {len(part_files)} part files to {output_path}")
    else:
        print(f"Chunked results are written incrementally under {chunk_dir}")


if __name__ == "__main__":
    main()
