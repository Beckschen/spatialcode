"""
GRPO Trainer for spatial reasoning RL finetuning.

Implements Group Relative Policy Optimization (Shao et al., 2024)
with spatial rubric rewards for training LLMs on spatial code reasoning.

Training loop:
    1. Sample G rollouts per prompt
    2. Score each rollout with spatial rubric reward
    3. Compute group-normalized advantages
    4. Update policy with PPO-style objective + KL penalty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO training hyperparameters (Table 6 from paper)."""
    # Model
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    # GRPO
    group_size: int = 16          # G: samples per prompt
    kl_coeff: float = 0.01        # β: KL penalty coefficient
    temperature: float = 0.7      # τ: sampling temperature
    reward_clip: tuple = (-0.5, 1.8)

    # Optimization
    learning_rate: float = 1e-6
    batch_size_per_device: int = 4
    gradient_accumulation: int = 2
    num_epochs: int = 1

    # Sequence
    max_prompt_length: int = 4096
    max_completion_length: int = 2048

    # Infrastructure
    precision: str = "bf16"


class GRPOTrainer:
    """GRPO trainer for spatial reasoning.

    Implements Equation (13):
        L_GRPO(θ) = E[Σ_i A^(i) log π_θ(y^(i)|x)] - β · KL[π_θ || π_ref]

    Args:
        config: GRPOConfig with training hyperparameters.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        if self.config.group_size < 2:
            raise ValueError(
                "GRPO requires group_size >= 2 for stable normalization, "
                f"got {self.config.group_size}."
            )
        self._setup_models()

    def _setup_models(self):
        """Initialize policy and reference models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading base model: {self.config.base_model}")

        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float32
        logger.info("Model precision=%s (torch dtype=%s)", self.config.precision, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            # Prevent ambiguous padding behavior during generation.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer had no pad_token; set pad_token to eos_token.")

        # Policy model (trainable)
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # Reference model (frozen)
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        logger.info("Policy/reference models initialized. Reference model frozen.")

    def _policy_unwrapped(self):
        return self.policy.module if hasattr(self.policy, "module") else self.policy

    def _policy_device(self) -> torch.device:
        return next(self._policy_unwrapped().parameters()).device

    def _ref_device(self) -> torch.device:
        return next(self.ref_policy.parameters()).device

    def compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute group-normalized advantages (Equation 12).

        A^(i) = (r(y^(i)|x) - μ_G) / (σ_G + ε)

        Args:
            rewards: (G,) reward scores for each rollout.

        Returns:
            (G,) normalized advantages.
        """
        mu = rewards.mean()
        # Use population std to avoid NaN when rewards has a single element.
        sigma = rewards.std(correction=0) + 1e-8
        return (rewards - mu) / sigma

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GRPO policy loss (Equation 13).

        Args:
            log_probs:     (G, L) log probs from policy.
            ref_log_probs: (G, L) log probs from reference.
            advantages:    (G,) normalized advantages.

        Returns:
            Scalar loss.
        """
        # Policy gradient term: Σ A^(i) * log π_θ(y^(i)|x)
        per_token_advantage = advantages[:, None] * log_probs  # (G, L)
        pg_loss = -per_token_advantage.sum(dim=-1).mean()

        # KL penalty: stable non-negative approximation used in RLHF/GRPO practice.
        # Equivalent to: exp(logp_ref - logp) - (logp_ref - logp) - 1, per token.
        log_ratio_ref_to_policy = ref_log_probs - log_probs
        per_token_kl = torch.exp(log_ratio_ref_to_policy) - log_ratio_ref_to_policy - 1.0
        kl_loss = self.config.kl_coeff * per_token_kl.sum(dim=-1).mean()

        return pg_loss + kl_loss

    def train_step(
        self,
        prompt: str,
        ground_truth: str,
        task_type: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single GRPO training step for one prompt.

        1. Sample G rollouts
        2. Score with spatial rubric reward
        3. Compute advantages
        4. Compute loss

        Args:
            prompt:       Input prompt with spatial codes.
            ground_truth: Ground truth answer.
            task_type:    Task type for reward computation.

        Returns:
            (loss, stats) where loss is a scalar tensor with gradients attached.
        """
        from spatialcode_train.reward.spatial_rubric import compute_reward

        G = self.config.group_size

        # Tokenize prompt
        policy_device = self._policy_device()
        policy_for_generate = self._policy_unwrapped()
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        prompt_ids = prompt_inputs["input_ids"].to(policy_device)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(policy_device)
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len >= self.config.max_prompt_length:
            logger.debug(
                "Prompt reached max_prompt_length=%s and may be truncated.",
                self.config.max_prompt_length,
            )

        # Sample G rollouts
        rollouts = []
        rollout_token_ids = []
        for _ in range(G):
            with torch.no_grad():
                output = policy_for_generate.generate(
                    prompt_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completion = self.tokenizer.decode(
                output[0][prompt_ids.shape[1]:],
                skip_special_tokens=True,
            )
            completion_ids = output[0][prompt_ids.shape[1]:].detach().to(policy_device)
            rollouts.append(completion)
            rollout_token_ids.append(completion_ids)
        if rollouts:
            rollout_lengths = [len(r) for r in rollouts]
            logger.debug(
                "Generated %s rollouts (completion chars min/mean/max=%s/%.1f/%s).",
                G,
                min(rollout_lengths),
                sum(rollout_lengths) / len(rollout_lengths),
                max(rollout_lengths),
            )

        # Score rollouts
        rewards = []
        for rollout in rollouts:
            breakdown = compute_reward(
                rollout,
                ground_truth,
                task_type,
                reward_clip=self.config.reward_clip,
            )
            rewards.append(breakdown.total)
        rewards_tensor = torch.tensor(rewards, device=policy_device)
        if not torch.isfinite(rewards_tensor).all():
            logger.warning("Non-finite rewards detected in train_step: %s", rewards)
        logger.debug(
            "Reward stats mean=%.4f std=%.4f min=%.4f max=%.4f",
            rewards_tensor.mean().item(),
            rewards_tensor.std().item(),
            rewards_tensor.min().item(),
            rewards_tensor.max().item(),
        )

        # Compute advantages
        advantages = self.compute_advantages(rewards_tensor)
        if not torch.isfinite(advantages).all():
            logger.warning("Non-finite advantages detected.")

        # Compute log probs for each rollout
        all_log_probs = []
        all_ref_log_probs = []

        for completion_ids in rollout_token_ids:
            if completion_ids.numel() == 0:
                # Keep tensor shape consistent when generation returns no completion token.
                all_log_probs.append(torch.zeros(0, device=policy_device))
                all_ref_log_probs.append(torch.zeros(0, device=policy_device))
                continue
            full_ids = torch.cat([prompt_ids[0], completion_ids], dim=0).unsqueeze(0)
            attention_mask = torch.ones_like(full_ids, device=policy_device)
            prompt_len = prompt_ids.shape[1]

            # Policy log probs
            with torch.set_grad_enabled(True):
                logits = self.policy(full_ids, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits[:, prompt_len-1:-1], dim=-1)
            target_ids = full_ids[:, prompt_len:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            all_log_probs.append(token_log_probs.squeeze(0))

            # Reference log probs
            with torch.no_grad():
                ref_device = self._ref_device()
                ref_input_ids = full_ids.to(ref_device)
                ref_attention_mask = attention_mask.to(ref_device)
                ref_logits = self.ref_policy(
                    ref_input_ids,
                    attention_mask=ref_attention_mask,
                ).logits
            ref_log_probs = F.log_softmax(ref_logits[:, prompt_len-1:-1], dim=-1)
            ref_token_log_probs = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            all_ref_log_probs.append(ref_token_log_probs.squeeze(0))

        if not all(lp.numel() > 0 for lp in all_log_probs):
            non_empty_pairs = [
                (lp, rlp)
                for lp, rlp in zip(all_log_probs, all_ref_log_probs, strict=True)
                if lp.numel() > 0
            ]
            if len(non_empty_pairs) == 0:
                # No completion token at all; return a zero loss that still supports backward.
                loss = torch.zeros((), device=policy_device, requires_grad=True)
                logger.warning(
                    "All rollouts produced empty completions; skipping update for this sample."
                )
                return loss, {
                    "loss": 0.0,
                    "mean_reward": rewards_tensor.mean().item(),
                    "std_reward": rewards_tensor.std().item(),
                    "min_reward": rewards_tensor.min().item(),
                    "max_reward": rewards_tensor.max().item(),
                }
            all_log_probs = [p[0] for p in non_empty_pairs]
            all_ref_log_probs = [p[1] for p in non_empty_pairs]
            valid_idx = [i for i, lp in enumerate(rollout_token_ids) if lp.numel() > 0]
            advantages = advantages[valid_idx]

        # Pad and stack
        max_len = max(lp.shape[0] for lp in all_log_probs)
        log_probs_padded = torch.stack([
            F.pad(lp, (0, max_len - lp.shape[0]), value=0) for lp in all_log_probs
        ])
        ref_log_probs_padded = torch.stack([
            F.pad(lp, (0, max_len - lp.shape[0]), value=0) for lp in all_ref_log_probs
        ])

        # Compute loss
        loss = self.compute_grpo_loss(log_probs_padded, ref_log_probs_padded, advantages)
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected in train_step.")

        return loss, {
            "loss": loss.item(),
            "mean_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std().item(),
            "min_reward": rewards_tensor.min().item(),
            "max_reward": rewards_tensor.max().item(),
        }
