"""
LLM inference backends: HuggingFace, vLLM, API.

All backends accept a config and are registered in LLM_BACKENDS.
Build via:
    backend = LLM_BACKENDS.build(HFBackendConfig(model_name="Qwen/Qwen3-4B"))
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from spatialcode.config import (
    LLMBackendConfig,
    HFBackendConfig,
    VLLMBackendConfig,
    APIBackendConfig,
)
from spatialcode.utils.registry import LLM_BACKENDS

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0, **kw) -> str: ...

    @abstractmethod
    def batch_generate(self, prompts: List[str], max_tokens: int = 2048, temperature: float = 0.0, **kw) -> List[str]: ...


# ── HuggingFace ───────────────────────────────────────────────────────

@LLM_BACKENDS.register("hf")
class HFBackend(LLMBackend):
    def __init__(self, cfg: HFBackendConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype, device_map=cfg.device, trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"Loaded HF model: {cfg.model_name}")

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0, **kw) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-4), do_sample=temperature > 0, **kw,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kw) -> List[str]:
        return [self.generate(p, **kw) for p in prompts]


# ── vLLM ──────────────────────────────────────────────────────────────

@LLM_BACKENDS.register("vllm")
class VLLMBackend(LLMBackend):
    def __init__(self, cfg: VLLMBackendConfig):
        from vllm import LLM, SamplingParams
        self.cfg = cfg
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=cfg.model_name,
            tensor_parallel_size=cfg.tensor_parallel,
            max_model_len=cfg.max_model_len,
            trust_remote_code=True,
        )
        logger.info(f"Loaded vLLM: {cfg.model_name} (tp={cfg.tensor_parallel})")

    def generate(self, prompt: str, **kw) -> str:
        return self.batch_generate([prompt], **kw)[0]

    def batch_generate(self, prompts: List[str], max_tokens: int = 2048, temperature: float = 0.0, **kw) -> List[str]:
        params = self._SamplingParams(max_tokens=max_tokens, temperature=max(temperature, 1e-4), **kw)
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


# ── API (OpenAI-compatible) ───────────────────────────────────────────

@LLM_BACKENDS.register("api")
class APIBackend(LLMBackend):
    def __init__(self, cfg: APIBackendConfig):
        from openai import OpenAI
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0, **kw) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content

    def batch_generate(self, prompts: List[str], max_workers: int = 16, **kw) -> List[str]:
        """Run generate in parallel (concurrent API requests). More workers = higher throughput but
        the server (e.g. SGLang) may use more VRAM if it keeps multiple requests in flight; reduce
        max_workers if the server runs OOM."""
        if len(prompts) <= 1:
            return [self.generate(p, **kw) for p in prompts]
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = min(max_workers, len(prompts))
        out = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(self.generate, p, **kw): i for i, p in enumerate(prompts)}
            for f in as_completed(fut):
                out[fut[f]] = f.result()
        return out
