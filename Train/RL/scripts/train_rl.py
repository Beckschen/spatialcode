#!/usr/bin/env python3
"""
Train LLM with GRPO + Spatial Rubric Reward.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 scripts/train_rl.py \
        --config config/yaml/rl_grpo.yaml

    # Custom model
    torchrun --nproc_per_node=4 scripts/train_rl.py \
        --config config/yaml/rl_grpo.yaml \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --lr 1e-6
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from spatialcode.code.spatial_code import SceneCode
from spatialcode.reason.prompt.builder import build_prompt
from spatialcode_train.trainer.grpo_trainer import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_main_process() -> bool:
    rank, _, _ = get_dist_info()
    return rank == 0


def setup_distributed() -> tuple[int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_distributed_mode(strategy: str | None, world_size: int) -> str:
    """Resolve configured distributed strategy into an executable mode."""
    raw = (strategy or "ddp").strip().lower()

    if raw in {"none", "single", "disabled"}:
        if world_size > 1:
            raise ValueError(
                "infrastructure.distributed_strategy is set to 'none/single' "
                f"but WORLD_SIZE={world_size}. Launch with a single process or use 'ddp'."
            )
        return "single"

    if raw in {"ddp", "auto"}:
        return "ddp" if world_size > 1 else "single"

    if raw in {"deepspeed_zero2", "deepspeed_stage2", "deepspeed"}:
        return "deepspeed_zero2"

    if is_main_process():
        logger.warning(
            "Unknown distributed_strategy=%s. Falling back to %s.",
            raw,
            "ddp" if world_size > 1 else "single",
        )
    return "ddp" if world_size > 1 else "single"


def ddp_debug_check(
    *,
    rank: int,
    world_size: int,
    step: int,
    sample_idx: int,
    policy,
):
    if world_size <= 1 or not dist.is_initialized():
        return

    # Check data sharding: gather one sample index per rank at this step.
    idx_tensor = torch.tensor([sample_idx], device=next(policy.parameters()).device)
    idx_gathered = [torch.zeros_like(idx_tensor) for _ in range(world_size)]
    dist.all_gather(idx_gathered, idx_tensor)

    # Check gradient synchronization: all ranks should see identical post-allreduce grad norms.
    grad_norm_value = None
    for p in policy.parameters():
        if p.grad is not None:
            grad_norm_value = p.grad.detach().norm()
            break
    if grad_norm_value is None:
        raise RuntimeError("DDP debug check failed: no gradient found after backward().")

    grad_tensor = grad_norm_value.unsqueeze(0)
    grad_gathered = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
    dist.all_gather(grad_gathered, grad_tensor)

    if rank == 0:
        idx_list = [int(x.item()) for x in idx_gathered]
        grad_list = [float(x.item()) for x in grad_gathered]
        grad_min, grad_max = min(grad_list), max(grad_list)
        tol = 1e-6 * max(1.0, abs(grad_max))
        grads_synced = abs(grad_max - grad_min) <= tol
        logger.info(
            "[DDP-DEBUG] step=%d sample_idx_per_rank=%s grad_norm_per_rank=%s grads_synced=%s",
            step,
            idx_list,
            [round(g, 6) for g in grad_list],
            grads_synced,
        )
        if not grads_synced:
            raise RuntimeError(
                "DDP debug check failed at step="
                f"{step}: grad norms differ across ranks: {grad_list}"
            )


class SpatialRLDataset(Dataset):
    """Dataset for spatial reasoning RL training."""

    def __init__(self, data_file: str, task_types: list[str] | None = None):
        with open(data_file) as f:
            self.data = json.load(f)
        self.task_types = tuple(task_types or ())
        if self.task_types:
            self.data = [
                item
                for item in self.data
                if self._configured_task_type(item) is not None
            ]
        if is_main_process():
            logger.info(f"Loaded {len(self.data)} training examples from {data_file}")

    @staticmethod
    def _data_task_type(item: dict) -> str | None:
        """Read task labels from either the training or evaluation schema."""
        return item.get("task_type") or item.get("qa_type")

    def _configured_task_type(self, item: dict) -> str | None:
        """Return the configured task matched by an item's possibly refined label."""
        data_task_type = self._data_task_type(item)
        if data_task_type is None:
            return None
        for task_type in self.task_types:
            if data_task_type == task_type or data_task_type.startswith(f"{task_type}_"):
                return task_type
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task_type = self._configured_task_type(item) or self._data_task_type(item) or "unknown"
        scene_code = SceneCode.from_dict(item)
        prompt = build_prompt(
            scene_code,
            item["questions"],
            task_type=task_type,
            options=item.get("options"),
        )
        return {
            "prompt": prompt,
            "ground_truth": item["solution"],
            "task_type": task_type,
            "sample_idx": idx,
        }


def main():
    parser = argparse.ArgumentParser(description="GRPO RL Training for Spatial Reasoning")
    parser.add_argument("--config", type=str, default=str(RL_ROOT / "config/yaml/rl_grpo.yaml"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/rl_spatial")
    parser.add_argument(
        "--ddp_debug_steps",
        type=int,
        default=0,
        help="Run DDP debug assertions for the first N steps (0 disables).",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    try:
        # Load config
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        seed = int(cfg.get("infrastructure", {}).get("seed", 42))
        set_global_seed(seed)
        if is_main_process():
            logger.info("Using global seed=%s", seed)

        batch_size_per_device = int(cfg["optimization"].get("batch_size_per_device", 1))
        if batch_size_per_device < 1:
            raise ValueError(
                f"optimization.batch_size_per_device must be >= 1, got {batch_size_per_device}"
            )

        gradient_accumulation_steps = int(cfg["optimization"]["gradient_accumulation_steps"])
        if gradient_accumulation_steps < 1:
            raise ValueError(
                "optimization.gradient_accumulation_steps must be >= 1, "
                f"got {gradient_accumulation_steps}"
            )
        max_grad_norm = float(cfg["optimization"].get("max_grad_norm", 1.0))
        if max_grad_norm <= 0:
            raise ValueError(f"optimization.max_grad_norm must be > 0, got {max_grad_norm}")
        log_every_n_steps = int(cfg.get("logging", {}).get("log_every_n_steps", 10))
        if log_every_n_steps < 1:
            raise ValueError(f"logging.log_every_n_steps must be >= 1, got {log_every_n_steps}")
        save_every_n_steps = int(cfg.get("logging", {}).get("save_every_n_steps", 500))
        if save_every_n_steps < 1:
            raise ValueError(f"logging.save_every_n_steps must be >= 1, got {save_every_n_steps}")

        computed_effective_bs = batch_size_per_device * world_size * gradient_accumulation_steps
        if is_main_process():
            logger.info(
                "Effective batch size = %s "
                "(batch_size_per_device=%s, world_size=%s, grad_accum=%s).",
                computed_effective_bs,
                batch_size_per_device,
                world_size,
                gradient_accumulation_steps,
            )

        distributed_mode = resolve_distributed_mode(
            cfg.get("infrastructure", {}).get("distributed_strategy"),
            world_size,
        )
        if is_main_process():
            logger.info(
                "Runtime setup rank/world_size/local_rank=%s/%s/%s distributed_mode=%s device=%s",
                rank,
                world_size,
                local_rank,
                distributed_mode,
                "cuda" if torch.cuda.is_available() else "cpu",
            )

        # Build GRPO config
        grpo_config = GRPOConfig(
            base_model=args.model or cfg["model"]["base_model"],
            group_size=cfg["grpo"]["group_size"],
            kl_coeff=cfg["grpo"]["kl_coeff"],
            temperature=cfg["grpo"]["temperature"],
            reward_clip=tuple(cfg["grpo"].get("reward_clip", (-0.5, 1.8))),
            learning_rate=args.lr or cfg["optimization"]["learning_rate"],
            batch_size_per_device=batch_size_per_device,
            gradient_accumulation=gradient_accumulation_steps,
            num_epochs=cfg["optimization"]["num_epochs"],
            max_prompt_length=cfg["sequence"]["max_prompt_length"],
            max_completion_length=cfg["sequence"]["max_completion_length"],
            precision=cfg["model"]["precision"],
        )

        # Setup trainer
        trainer = GRPOTrainer(grpo_config)
        device = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        use_grad_ckpt = bool(cfg.get("infrastructure", {}).get("gradient_checkpointing", False))
        if use_grad_ckpt:
            policy_for_ckpt = trainer.policy
            policy_for_ckpt.gradient_checkpointing_enable()
            # Required by HF when gradient checkpointing is enabled for decoder-only models.
            policy_for_ckpt.config.use_cache = False
            if is_main_process():
                logger.info("Enabled gradient checkpointing (policy.config.use_cache=False).")
        trainer.policy.to(device)
        trainer.ref_policy.to(device)
        if distributed_mode == "ddp":
            trainer.policy = DDP(
                trainer.policy,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )
            if is_main_process():
                logger.info("Using distributed mode: DDP")
        elif distributed_mode == "deepspeed_zero2":
            try:
                import deepspeed
            except ImportError as exc:
                raise ImportError(
                    "distributed_strategy=deepspeed_zero2 requires DeepSpeed. "
                    "Install with `pip install deepspeed`."
                ) from exc

            ds_config = {
                "train_micro_batch_size_per_gpu": grpo_config.batch_size_per_device,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": max_grad_norm,
                "bf16": {"enabled": grpo_config.precision == "bf16"},
                "fp16": {"enabled": grpo_config.precision == "fp16"},
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 5e8,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e8,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": float(grpo_config.learning_rate),
                        "weight_decay": float(cfg["optimization"].get("weight_decay", 0.0)),
                    },
                },
            }
            trainer.policy, _, _, _ = deepspeed.initialize(
                model=trainer.policy,
                model_parameters=trainer.policy.parameters(),
                config=ds_config,
                dist_init_required=False,
            )
            if is_main_process():
                logger.info("Using distributed mode: DeepSpeed ZeRO-2")
        elif is_main_process():
            logger.info("Using distributed mode: single-process")

        # Setup data
        dataset = SpatialRLDataset(
            cfg["data"]["train_file"],
            task_types=cfg.get("data", {}).get("task_types"),
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        ) if distributed_mode in {"ddp", "deepspeed_zero2"} and world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=grpo_config.batch_size_per_device,
            shuffle=(sampler is None),
            sampler=sampler,
        )
        if is_main_process():
            logger.info(
                "Data ready: samples=%s, batches_per_epoch=%s, batch_size_per_device=%s.",
                len(dataset),
                len(dataloader),
                grpo_config.batch_size_per_device,
            )

        # Setup optimizer
        optimizer = None
        if distributed_mode != "deepspeed_zero2":
            optimizer = torch.optim.AdamW(
                trainer.policy.parameters(),
                lr=grpo_config.learning_rate,
                weight_decay=cfg["optimization"].get("weight_decay", 0.0),
            )
            optimizer.zero_grad()

        # Training loop
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        accum_steps_since_update = 0
        for epoch in range(grpo_config.num_epochs):
            if is_main_process():
                logger.info("Starting epoch %s/%s", epoch + 1, grpo_config.num_epochs)
            if sampler is not None:
                sampler.set_epoch(epoch)
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for batch in pbar:
                sample_idx = int(batch["sample_idx"][0])
                losses = []
                step_stats_list = []
                batch_items = len(batch["prompt"])
                for i in range(batch_items):
                    sample_loss, sample_stats = trainer.train_step(
                        prompt=batch["prompt"][i],
                        ground_truth=batch["ground_truth"][i],
                        task_type=batch["task_type"][i],
                    )
                    losses.append(sample_loss)
                    step_stats_list.append(sample_stats)
                loss = torch.stack(losses).mean()
                mean_step_stats = {
                    "loss": sum(s["loss"] for s in step_stats_list) / len(step_stats_list),
                    "mean_reward": (
                        sum(s["mean_reward"] for s in step_stats_list) / len(step_stats_list)
                    ),
                    "std_reward": (
                        sum(s["std_reward"] for s in step_stats_list) / len(step_stats_list)
                    ),
                    "min_reward": (
                        sum(s["min_reward"] for s in step_stats_list) / len(step_stats_list)
                    ),
                    "max_reward": (
                        sum(s["max_reward"] for s in step_stats_list) / len(step_stats_list)
                    ),
                }

                # Backward
                if distributed_mode == "deepspeed_zero2":
                    trainer.policy.backward(loss)
                else:
                    loss.backward()

                if (
                    distributed_mode == "ddp"
                    and args.ddp_debug_steps > 0
                    and global_step < args.ddp_debug_steps
                ):
                    policy_for_debug = (
                        trainer.policy.module
                        if hasattr(trainer.policy, "module")
                        else trainer.policy
                    )
                    ddp_debug_check(
                        rank=rank,
                        world_size=world_size,
                        step=global_step,
                        sample_idx=sample_idx,
                        policy=policy_for_debug,
                    )

                accum_steps_since_update += 1
                if accum_steps_since_update >= grpo_config.gradient_accumulation:
                    grad_norm = None
                    if distributed_mode == "deepspeed_zero2":
                        trainer.policy.step()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainer.policy.parameters(),
                            max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                    accum_steps_since_update = 0
                    if is_main_process() and grad_norm is not None:
                        logger.debug(
                            "Optimizer step at global_step=%s grad_norm=%.4f",
                            global_step,
                            float(grad_norm),
                        )

                if is_main_process():
                    pbar.set_postfix({
                        "loss": f"{mean_step_stats['loss']:.3f}",
                        "reward": f"{mean_step_stats['mean_reward']:.3f}",
                    })
                    if global_step % log_every_n_steps == 0:
                        logger.info(
                            "step=%s loss=%.4f reward_mean=%.4f reward_std=%.4f "
                            "reward_min=%.4f reward_max=%.4f accum=%s/%s",
                            global_step,
                            mean_step_stats["loss"],
                            mean_step_stats["mean_reward"],
                            mean_step_stats["std_reward"],
                            mean_step_stats["min_reward"],
                            mean_step_stats["max_reward"],
                            accum_steps_since_update,
                            grpo_config.gradient_accumulation,
                        )
                global_step += 1

                # Save checkpoint
                if (
                    is_main_process()
                    and global_step % save_every_n_steps == 0
                ):
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    policy_to_save = (
                        trainer.policy.module
                        if hasattr(trainer.policy, "module")
                        else trainer.policy
                    )
                    policy_to_save.save_pretrained(ckpt_dir)
                    trainer.tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")
            if is_main_process():
                logger.info("Finished epoch %s/%s", epoch + 1, grpo_config.num_epochs)

        # Flush remainder gradients if the total steps are not divisible by accumulation.
        if accum_steps_since_update > 0:
            if distributed_mode == "deepspeed_zero2":
                trainer.policy.step()
            else:
                torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if is_main_process():
                logger.info(
                    "Flushed final optimizer step with remainder accumulation (%s/%s).",
                    accum_steps_since_update,
                    grpo_config.gradient_accumulation,
                )

        if dist.is_initialized():
            dist.barrier()

        # Save final model
        if is_main_process():
            policy_to_save = (
                trainer.policy.module
                if hasattr(trainer.policy, "module")
                else trainer.policy
            )
            policy_to_save.save_pretrained(output_dir / "final")
            trainer.tokenizer.save_pretrained(output_dir / "final")
            logger.info(f"Training complete. Model saved to {output_dir / 'final'}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
