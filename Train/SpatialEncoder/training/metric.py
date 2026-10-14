"""
Metric classes for training evaluation.

This module provides metric implementations that are used during training
and validation to evaluate model performance. Each metric class follows
a consistent interface with update() and compute_synced() methods.
"""

import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Union
from collections import defaultdict


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_dist_device():
    """Get the appropriate device for distributed operations.

    Returns:
        torch.device: The device to use for distributed tensors.
        - In DDP: returns the current CUDA device for this process
        - Single GPU: returns cuda:0 if available
        - CPU only: returns cpu
    """
    if is_dist_avail_and_initialized():
        # In DDP, use the current device for this process
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            return torch.device(f"cuda:{current_device}")
        else:
            return torch.device("cpu")
    else:
        # Single process training
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")


def gather_values(values: List | np.ndarray | torch.Tensor) -> List:
    """Gather values with detailed diagnostics"""
    if not is_dist_avail_and_initialized():
        return (
            values
            if isinstance(values, list)
            else values.tolist()
            if isinstance(values, np.ndarray)
            else values.cpu().tolist()
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert to list
    values = (
        values
        if isinstance(values, list)
        else values.tolist()
        if isinstance(values, np.ndarray)
        else values.cpu().tolist()
    )

    # ========== 详细诊断 ==========
    import sys
    import gc

    data_size_mb = sys.getsizeof(values) / (1024 * 1024)

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        available = total - allocated

        # 估算 gather 需要的内存
        estimated_need = data_size_mb * 5 / 1024  # 序列化开销约 5x

        print(f"[Rank {rank}] Gather diagnostics:", flush=True)
        print(f"  - Data: {len(values)} values, {data_size_mb:.2f} MB", flush=True)
        print(f"  - GPU allocated: {allocated:.2f} GB", flush=True)
        print(f"  - GPU reserved: {reserved:.2f} GB", flush=True)
        print(f"  - GPU available: {available:.2f} GB", flush=True)
        print(f"  - Estimated need: {estimated_need:.2f} GB", flush=True)
        print(f"  - Safety margin: {available - estimated_need:.2f} GB", flush=True)

        if estimated_need > available * 0.8:
            print(
                f"[Rank {rank}] WARNING: Might OOM! "
                f"Need {estimated_need:.2f} GB but only {available:.2f} GB available",
                flush=True,
            )

    # Gather
    try:
        if rank == 0:
            gathered_data = [None for _ in range(world_size)]

            print(f"[Rank 0] Starting gather from {world_size} ranks...", flush=True)
            dist.gather_object(values, gathered_data, dst=0)

            # 打印每个 rank 的数据量
            for i, data in enumerate(gathered_data):
                if data is not None:
                    data_mb = sys.getsizeof(data) / (1024 * 1024)
                    print(
                        f"[Rank 0] Received from Rank {i}: {len(data)} values, "
                        f"{data_mb:.2f} MB",
                        flush=True,
                    )

            combined_data = []
            for data in gathered_data:
                if data is not None:
                    combined_data.extend(data)

            print(f"[Rank 0] Total gathered: {len(combined_data)} values", flush=True)
            return combined_data
        else:
            print(
                f"[Rank {rank}] Sending {len(values)} values to Rank 0...", flush=True
            )
            dist.gather_object(values, dst=0)
            print(f"[Rank {rank}] Send completed", flush=True)
            return []

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print(f"[Rank {rank}] OOM ERROR during gather!", flush=True)
            print(f"[Rank {rank}] Data size: {len(values)} values", flush=True)
            if torch.cuda.is_available():
                print(f"[Rank {rank}] GPU allocated: {allocated:.2f} GB", flush=True)
        raise
    finally:
        # 清理
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

            

class BaseMetric:
    """Base class for all metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric state."""
        pass

    def update(self, find_stages: Dict, find_metadatas: List[Dict], **kwargs):
        """Update metric with new data.

        Args:
            find_stages: Dictionary containing model outputs
            find_metadatas: List of metadata dictionaries for each sample
        """
        pass

    def compute_synced(self) -> Dict[str, float]:
        """Compute metric values synchronized across processes.

        Returns:
            Dictionary with metric values
        """
        return {}

    def compute_local(self) -> Dict[str, float]:
        """Compute metric values locally without distributed synchronization.
        Fast implementation for inference/logging.

        Returns:
            Dictionary with metric values
        """
        return {}

    def compute(self) -> Dict[str, float]:
        """Compute metric values. Uses fast local computation for compatibility.

        Returns:
            Dictionary with metric values
        """
        return self.compute_local()

    def is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best value.

        Args:
            current: Current metric value
            best: Best metric value so far

        Returns:
            True if current is better than best
        """
        return current > best  # Default: higher is better


class MeanIoU(BaseMetric):
    """Mean Intersection over Union metric for 3D bounding boxes."""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """Reset IoU accumulator."""
        self.ious = []
        self.count = 0

    def update(
        self, find_stages: Dict[str, List[float]], find_metadatas: List[Dict], **kwargs
    ):
        """Update with IoU values from model predictions."""
        assert "ious" in find_stages, "IoUs not found in find_stages"
        ious = find_stages["ious"]
        assert ious is not None, "IoUs should not be None"
        if torch.is_tensor(ious):
            ious = ious.detach().cpu().numpy()
        elif not isinstance(ious, np.ndarray):
            ious = np.array([ious] if isinstance(ious, (int, float)) else ious)

        # Remove invalid IoUs
        valid_mask = ~(np.isnan(ious) | np.isinf(ious))
        ious = ious[valid_mask]

        self.ious.extend(ious.tolist())

        # If find_stages["gt_existence"] is a list, use sum() built-in function.
        gt_existence = find_stages.get("gt_existence")
        assert gt_existence is not None, "Ground truth existence should not be None"
        assert isinstance(gt_existence, list), "Ground truth existence should be a list"
        self.count += sum(gt_existence)

    def compute_synced(self) -> Dict[str, float]:
        """Compute mean IoU using gather (only valid on rank 0)."""
        if not is_dist_avail_and_initialized():
            return self.compute_local()

        # Gather all IoU values to rank 0
        all_ious = gather_values(self.ious)
        all_counts = gather_values([self.count])

        if dist.get_rank() == 0:
            if not all_ious:
                return {"miou": 0.0}
            total_sum = sum(all_ious)
            total_count = sum(all_counts) if all_counts else 0
            mean_iou = total_sum / total_count if total_count > 0 else 0.0
            ## DEBUG
            print(f"Global mean IoU computed: {mean_iou}, count: {total_count}")
            return {"miou": mean_iou, "global_count": total_count}
        else:
            # Non-rank-0 processes return empty dict
            return {}

    def compute_local(self) -> Dict[str, float]:
        """Compute local mean IoU without distributed synchronization."""
        if not self.ious:
            return {"miou": 0.0}

        local_sum = sum(self.ious)
        local_count = self.count
        mean_iou = local_sum / local_count if local_count > 0 else 0.0

        return {"miou": mean_iou}


class PreRecF1Score(BaseMetric):
    """Precision, Recall, and F1 Score metric with configurable IoU threshold(s)."""

    def __init__(self, threshold: Union[float, List[float]] = 0.5):
        # if isinstance(threshold, list):
        #     threshold = [_ for _ in np.arange(threshold[0], threshold[1]+ 0.05, 0.05)]
        self.threshold = threshold if isinstance(threshold, list) else [threshold]
        super().__init__()

    def reset(self):
        """Reset F1 score counters."""
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def update(
        self, find_stages: Dict[str, List[float]], find_metadatas: List[Dict], **kwargs
    ):
        """Update F1 score counters based on IoU predictions."""
        assert "ious" in find_stages, "IoUs not found in find_stages"
        ious = find_stages["ious"]
        assert ious is not None, "IoUs should not be None"
        if torch.is_tensor(ious):
            ious = ious.detach().cpu().numpy()
        elif not isinstance(ious, np.ndarray):
            ious = np.array([ious] if isinstance(ious, (int, float)) else ious)
        # Remove invalid IoUs
        valid_mask = ~(np.isnan(ious) | np.isinf(ious))
        ious = ious[valid_mask]

        for thresh in self.threshold:
            # Count predictions above threshold as positive
            predictions_positive = int(np.sum(ious >= thresh))
            predictions_negative = len(ious) - predictions_positive

            # True positives: predictions above threshold
            self.tp[thresh] += predictions_positive
            # False positives: none in this simplified case
            # False negatives: predictions below threshold
            self.fn[thresh] += predictions_negative

            # Assuming find_stages contains 'pred_existence' indicating predicted positives
            assert "pred_existence" in find_stages, (
                "Predicted existence not found in find_stages"
            )
            assert isinstance(find_stages["pred_existence"], list), (
                "Predicted existence should be a list"
            )
            pred_positive_count = sum(find_stages["pred_existence"])
            self.fp[thresh] += max(pred_positive_count - int(self.tp[thresh]), 0)

    def compute_synced(self) -> Dict[str, float]:
        """Compute synchronized F1 scores."""
        results = {}

        # Check if we're rank 0 once, outside the loop
        is_rank_0 = not is_dist_avail_and_initialized() or dist.get_rank() == 0

        for thresh in self.threshold:
            local_tp = int(self.tp[thresh])  # Ensure it's an int
            local_fp = int(self.fp[thresh])  # Ensure it's an int
            local_fn = int(self.fn[thresh])  # Ensure it's an int

            # Gather values to rank 0 - all ranks participate
            if is_dist_avail_and_initialized():
                global_tp_list = gather_values([local_tp])
                global_fp_list = gather_values([local_fp])
                global_fn_list = gather_values([local_fn])

                if is_rank_0:
                    global_tp = sum(global_tp_list) if global_tp_list else 0
                    global_fp = sum(global_fp_list) if global_fp_list else 0
                    global_fn = sum(global_fn_list) if global_fn_list else 0
                else:
                    # Non-rank-0 processes continue the loop but don't compute results
                    continue
            else:
                global_tp, global_fp, global_fn = local_tp, local_fp, local_fn

            # Compute F1 score (only on rank 0 in distributed mode, or always if not distributed)
            if is_rank_0:
                precision = (
                    global_tp / (global_tp + global_fp)
                    if (global_tp + global_fp) > 0
                    else 0.0
                )
                recall = (
                    global_tp / (global_tp + global_fn)
                    if (global_tp + global_fn) > 0
                    else 0.0
                )
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                results[f"f1@{thresh}"] = f1
                results[f"precision@{thresh}"] = precision
                results[f"recall@{thresh}"] = recall

        # Only compute summary stats on rank 0
        if is_rank_0 and results:
            results["mf1"] = np.mean(list(results.values()))
            results["mprecision"] = np.mean(
                [results[f"precision@{thresh}"] for thresh in self.threshold]
            )
            results["mrecall"] = np.mean(
                [results[f"recall@{thresh}"] for thresh in self.threshold]
            )

        return results

    def compute_local(self) -> Dict[str, float]:
        """Compute local F1 scores without distributed synchronization."""
        results = {}

        for thresh in self.threshold:
            local_tp = self.tp[thresh]
            local_fp = self.fp[thresh]
            local_fn = self.fn[thresh]

            # Compute F1 score locally
            precision = (
                local_tp / (local_tp + local_fp) if (local_tp + local_fp) > 0 else 0.0
            )
            recall = (
                local_tp / (local_tp + local_fn) if (local_tp + local_fn) > 0 else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            results[f"f1@{thresh}"] = f1
            results[f"precision@{thresh}"] = precision
            results[f"recall@{thresh}"] = recall

        if results:
            results["mf1"] = np.mean(list(results.values()))
            results["mprecision"] = np.mean(
                [results[f"precision@{thresh}"] for thresh in self.threshold]
            )
            results["mrecall"] = np.mean(
                [results[f"recall@{thresh}"] for thresh in self.threshold]
            )

        return results



class TranslationError(BaseMetric):
    """translation error metric."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        super().__init__()

    def reset(self):
        """Reset translation error accumulator."""
        self.errors = []
        self.counts = 0

    def update(
        self, find_stages: Dict[str, List[float]], find_metadatas: List[Dict], **kwargs
    ):
        """Update with translation errors."""
        if "translation_errors" in find_stages:
            errors = find_stages["translation_errors"]
            if errors is not None:
                if torch.is_tensor(errors):
                    errors = errors.detach().cpu().numpy()
                elif not isinstance(errors, np.ndarray):
                    errors = np.array(
                        [errors] if isinstance(errors, (int, float)) else errors
                    )

                # Scale errors and remove invalid values
                errors = errors * self.scale
                valid_mask = ~(np.isnan(errors) | np.isinf(errors))
                errors = errors[valid_mask]

                self.errors.extend(errors.tolist())
        self.counts += find_stages["gt_existence"].count(1)

    def compute_synced(self) -> Dict[str, float]:
        """Compute synchronized translation error."""
        if not self.errors:
            return {"translation_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts

        # Gather values to rank 0 - all ranks participate
        if is_dist_avail_and_initialized():
            global_sum_list = gather_values([local_sum])
            global_count_list = gather_values([local_count])

            if dist.get_rank() == 0:
                global_sum = sum(global_sum_list) if global_sum_list else 0.0
                global_count = sum(global_count_list) if global_count_list else 0
                mean_error = global_sum / global_count if global_count > 0 else 0.0
                ## DEBUG
                print(
                    f"Global translation error computed: {mean_error}, count: {global_count}"
                )
                return {"translation_error": mean_error, "global_count": global_count}
            else:
                # Non-rank-0 processes return empty dict after participating in gather
                return {}
        else:
            # Single process
            mean_error = local_sum / local_count if local_count > 0 else 0.0
            return {"translation_error": mean_error}

    def compute_local(self) -> Dict[str, float]:
        """Compute local translation error without distributed synchronization."""
        if not self.errors:
            return {"translation_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts
        mean_error = local_sum / local_count if local_count > 0 else 0.0

        return {"translation_error": mean_error}

    def is_better(self, current: float, best: float) -> bool:
        """Lower translation error is better."""
        return current < best


class DepthError(BaseMetric):
    """Depth error metric."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        super().__init__()

    def reset(self):
        """Reset depth error accumulator."""
        self.errors = []
        self.counts = 0

    def update(
        self, find_stages: Dict[str, List[float]], find_metadatas: List[Dict], **kwargs
    ):
        """Update with depth errors."""
        if "depth_errors" in find_stages:
            errors = find_stages["depth_errors"]
            if errors is not None:
                if torch.is_tensor(errors):
                    errors = errors.detach().cpu().numpy()
                elif not isinstance(errors, np.ndarray):
                    errors = np.array(
                        [errors] if isinstance(errors, (int, float)) else errors
                    )

                # Scale errors and remove invalid values
                errors = errors * self.scale
                valid_mask = ~(np.isnan(errors) | np.isinf(errors))
                errors = errors[valid_mask]

                self.errors.extend(errors.tolist())
        self.counts += find_stages["gt_existence"].count(1)

    def compute_synced(self) -> Dict[str, float]:
        """Compute synchronized depth error."""
        if not self.errors:
            return {"depth_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts

        # Gather values to rank 0 - all ranks participate
        if is_dist_avail_and_initialized():
            global_sum_list = gather_values([local_sum])
            global_count_list = gather_values([local_count])

            if dist.get_rank() == 0:
                global_sum = sum(global_sum_list) if global_sum_list else 0.0
                global_count = sum(global_count_list) if global_count_list else 0
                mean_error = global_sum / global_count if global_count > 0 else 0.0
                ## DEBUG
                print(
                    f"Global depth error computed: {mean_error}, count: {global_count}"
                )
                return {"depth_error": mean_error, "global_count": global_count}
            else:
                # Non-rank-0 processes return empty dict after participating in gather
                return {}
        else:
            # Single process
            mean_error = local_sum / local_count if local_count > 0 else 0.0
            return {"depth_error": mean_error}

    def compute_local(self) -> Dict[str, float]:
        """Compute local depth error without distributed synchronization."""
        if not self.errors:
            return {"depth_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts
        mean_error = local_sum / local_count if local_count > 0 else 0.0

        return {"depth_error": mean_error}

    def is_better(self, current: float, best: float) -> bool:
        """Lower depth error is better."""
        return current < best


class ScaleError(BaseMetric):
    """Scale error metric."""

    def __init__(self):
        super().__init__()

    def reset(self):
        """Reset scale error accumulator."""
        self.errors = []
        self.counts = 0

    def update(
        self, find_stages: Dict[str, List[float]], find_metadatas: List[Dict], **kwargs
    ):
        """Update with scale errors."""
        if "scale_errors" in find_stages:
            errors = find_stages["scale_errors"]
            if errors is not None:
                if torch.is_tensor(errors):
                    errors = errors.detach().cpu().numpy()
                elif not isinstance(errors, np.ndarray):
                    errors = np.array(
                        [errors] if isinstance(errors, (int, float)) else errors
                    )

                # Remove invalid values
                valid_mask = ~(np.isnan(errors) | np.isinf(errors))
                errors = errors[valid_mask]

                self.errors.extend(errors.tolist())
        self.counts += find_stages["gt_existence"].count(1)

    def compute_synced(self) -> Dict[str, float]:
        """Compute synchronized scale error."""
        if not self.errors:
            return {"scale_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts

        # Gather values to rank 0 - all ranks participate
        if is_dist_avail_and_initialized():
            global_sum_list = gather_values([local_sum])
            global_count_list = gather_values([local_count])

            if dist.get_rank() == 0:
                global_sum = sum(global_sum_list) if global_sum_list else 0.0
                global_count = sum(global_count_list) if global_count_list else 0
                mean_error = global_sum / global_count if global_count > 0 else 0.0
                ## DEBUG
                print(
                    f"Global scale error computed: {mean_error}, count: {global_count}"
                )
                return {"scale_error": mean_error, "global_count": global_count}
            else:
                # Non-rank-0 processes return empty dict after participating in gather
                return {}
        else:
            # Single process
            mean_error = local_sum / local_count if local_count > 0 else 0.0
            return {"scale_error": mean_error}

    def compute_local(self) -> Dict[str, float]:
        """Compute local scale error without distributed synchronization."""
        if not self.errors:
            return {"scale_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts
        mean_error = local_sum / local_count if local_count > 0 else 0.0

        return {"scale_error": mean_error}


class YawError(BaseMetric):
    """Yaw angle error metric."""

    def __init__(self, unit: str = "rad"):
        self.unit = unit
        super().__init__()

    def reset(self):
        """Reset yaw error accumulator."""
        self.errors = []
        self.counts = 0

    def update(
        self, find_stages: Dict[str, List[float]], find_metadatas: List[Dict], **kwargs
    ):
        """Update with yaw angle errors."""
        if "yaw_errors" in find_stages or "orientation_errors" in find_stages:
            errors = find_stages.get(
                "yaw_errors", find_stages.get("orientation_errors")
            )
            if errors is not None:
                if torch.is_tensor(errors):
                    errors = errors.detach().cpu().numpy()
                elif not isinstance(errors, np.ndarray):
                    errors = np.array(
                        [errors] if isinstance(errors, (int, float)) else errors
                    )

                # Convert to specified unit if needed
                if self.unit == "deg" and errors.size > 0:
                    errors = np.degrees(errors)

                # Remove invalid values
                valid_mask = ~(np.isnan(errors) | np.isinf(errors))
                errors = errors[valid_mask]

                self.errors.extend(errors.tolist())
        self.counts += find_stages["gt_existence"].count(1)

    def compute_synced(self) -> Dict[str, float]:
        """Compute synchronized yaw error."""
        if not self.errors:
            return {"yaw_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts

        # Gather values to rank 0 - all ranks participate
        if is_dist_avail_and_initialized():
            global_sum_list = gather_values([local_sum])
            global_count_list = gather_values([local_count])

            if dist.get_rank() == 0:
                global_sum = sum(global_sum_list) if global_sum_list else 0.0
                global_count = sum(global_count_list) if global_count_list else 0
                mean_error = global_sum / global_count if global_count > 0 else 0.0
                ## DEBUG
                print(f"Global yaw error computed: {mean_error}, count: {global_count}")
                return {"yaw_error": mean_error, "global_count": global_count}
            else:
                # Non-rank-0 processes return empty dict after participating in gather
                return {}
        else:
            # Single process
            mean_error = local_sum / local_count if local_count > 0 else 0.0
            return {"yaw_error": mean_error}

    def compute_local(self) -> Dict[str, float]:
        """Compute local yaw error without distributed synchronization."""
        if not self.errors:
            return {"yaw_error": 0.0}

        local_sum = sum(self.errors)
        local_count = self.counts
        mean_error = local_sum / local_count if local_count > 0 else 0.0

        return {"yaw_error": mean_error}

    def is_better(self, current: float, best: float) -> bool:
        """Lower yaw error is better."""
        return current < best