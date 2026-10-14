# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code borrowed from TLC - https://www.internalfb.com/code/fbsource/fbcode/pytorch/tlc/torchtlc/loggers/tensorboard.py
import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union
from omegaconf import OmegaConf
import dataclasses
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from training.utils.train_utils import get_machine_local_and_dist_rank, makedir

# ---- NEW: wandb integration -------------------------------------------------
import os
import atexit

try:
    import wandb
except Exception:
    wandb = None


Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)

def _to_py_scalar(x: Scalar):
    # 保障 wandb.log 能吃到标量
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, ndarray) and x.size == 1:
        return float(x.reshape(-1)[0])
    return float(x) if isinstance(x, (int, float)) else x  # 最后兜底

def _make_json_safe_config(config):
    if config is None:
        return {}
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, dict):
        out = {}
        for k, v in config.items():
            if isinstance(v, (int,float,bool,str)):
                out[k] = v
            elif isinstance(v, (list,dict)):
                out[k] = v
            else:
                out[k] = str(v)
        return out
    if hasattr(config, "to_container"):  # DictConfig
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


class WandbLogger:
    """
    A simple logger for Weights & Biases (rank0-only).
    """
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        dir: Optional[str] = None,
        mode: str = "online",   # "online" | "offline" | "disabled"
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        id: Optional[str] = None,
        resume: Optional[str] = None,  # "allow"/"must"/None
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._run = None
        _, self._rank = get_machine_local_and_dist_rank()
        if self._rank != 0:
            logging.debug(f"[wandb] skip init on rank {self._rank}")
            return
        if wandb is None:
            logging.warning("[wandb] not installed. `pip install wandb` to enable.")
            return

        # 确保目录存在（wandb 会在该目录下写入 run）
        if dir:
            makedir(dir)

        # 允许通过 mode 控制 offline/disabled
        os.environ.setdefault("WANDB_DIR", dir or os.getcwd())
        os.environ.setdefault("WANDB_MODE", mode)

        self._run = wandb.init(
            project=project,
            name=name,
            dir=dir,
            entity=entity,
            tags=tags,
            group=group,
            id=id,
            resume=resume,   # "allow" 可断点续写
            notes=notes,
            config=_make_json_safe_config(config) or {},
            **kwargs,
        )
        if self._run:
            logging.info(f"[wandb] init: project={project}, name={name}, dir={dir}")
            wandb.run.log_code(
            ".",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".sh") or path.endswith(".yaml")
        )
        atexit.register(self.close)

    @property
    def run(self):
        return self._run

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._run:
            return
        # 将可能的 Tensor/ndarray 转成标量
        data = {k: _to_py_scalar(v) for k, v in payload.items()}
        wandb.log(data, step=step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._run:
            return
        wandb.log({name: _to_py_scalar(data)}, step=step)

    def log_hparams(self, hparams: Dict[str, Any], meters: Dict[str, Scalar]) -> None:
        if not self._run:
            return
        # hparams 写入到 config，最终指标写入到 summary
        self._run.config.update(hparams, allow_val_change=True)
        summary = {k: _to_py_scalar(v) for k, v in meters.items()}
        self._run.summary.update(summary)

    def flush(self) -> None:
        # wandb 没有显式 flush；调用 run.save/keep 不是必须
        pass

    def close(self) -> None:
        if not self._run:
            return
        try:
            self._run.finish()
        finally:
            self._run = None



def make_wandb_logger(
    project: str,
    name: Optional[str] = None,
    dir: Optional[str] = None,
    mode: str = "online",
    **kwargs: Any,
):
    """
    Hydra-friendly 工厂函数，保持与 make_tensorboard_logger 风格一致。
    用法参考配置样例。
    """
    print(f"[wandb] make_wandb_logger: project={project}, name={name}, dir={dir}, mode={mode}")
    return WandbLogger(project=project, name=name, dir=dir, mode=mode, **kwargs)
# -----------------------------------------------------------------------------

class Logger:
    """
    A logger class that can interface with multiple loggers.
    Supports TensorBoard and Weights & Biases.
    """

    def __init__(self, logging_conf):
        # TensorBoard
        tb_config = getattr(logging_conf, "tensorboard_writer", None)
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

        # Weights & Biases
        wb_config = getattr(logging_conf, "wandb_writter", None)
        wb_should_log = wb_config and wb_config.pop("should_log", True)
        self.wb_logger = instantiate(wb_config) if wb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if self.wb_logger:
            self.wb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if self.wb_logger:
            self.wb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        if self.wb_logger:
            self.wb_logger.log_hparams(hparams, meters)



# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
