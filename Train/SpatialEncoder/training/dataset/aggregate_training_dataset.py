import torch
import random
from torch.utils.data import DataLoader, IterableDataset
from typing import List, Dict


class AggregateDataset(IterableDataset):
    def __init__(
        self,
        meta_dir_dict: Dict,
        default_scales: Dict[str, float] = [1.0],
        base_dir: Dict[str, str] = ".",
        training: bool = True,
        resolution_aug_scale: List[float] = [0.8, 1.2],
        predict_metric: bool = False,
        use_extrinsic: bool = False,
        norm_scale: float = 1.0,
        resolution: int = 1024,
        spatial_resolution: int = 518,
        train_transforms=None,
        val_transforms=None,
        spatial_transforms=None,
        spatial_type: str = "vggt",
        video_split_per_frame: int = 20,
        frame_num_range: List[int] = [20, 64],
        frame_sample_gap_range: List[int] = [1, 3],
        max_num_objects: int = 20,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn=None,
        weights: List[float] = None,
    ):
        super().__init__()
        self.weights = weights
        # ============================
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # ============================
        dataset_cfg = dict(
            training=training,
            resolution_aug_scale=resolution_aug_scale,
            predict_metric=predict_metric,
            use_extrinsic=use_extrinsic,
            norm_scale=norm_scale,
            resolution=resolution,
            spatial_resolution=spatial_resolution,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            spatial_transforms=spatial_transforms,
            spatial_type=spatial_type,
            video_split_per_frame=video_split_per_frame,
            frame_num_range=frame_num_range,
            frame_sample_gap_range=frame_sample_gap_range,
            max_num_objects=max_num_objects,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers//len(meta_dir_dict),
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        dataset_names = list(meta_dir_dict.keys())
        if "CA-1M" in dataset_names:
            from .ca1m_iterable_dataset import CA1MTrainDataset

            self.CA1M = CA1MTrainDataset(
                meta_dir=meta_dir_dict["CA-1M"],
                dataset_name="CA-1M",
                default_scale=default_scales["CA-1M"],
                base_dir=base_dir["CA-1M"],
                **dataset_cfg,
            )
        if "hyperism" in dataset_names:
            from .base_dataset import ObjectDetectionDataset

            self.hyperism = ObjectDetectionDataset(
                meta_dir=meta_dir_dict["hyperism"],
                dataset_name="hyperism",
                default_scale=default_scales["hyperism"],
                base_dir=base_dir["hyperism"],
                **dataset_cfg,
            )
        if "ADT" in dataset_names:
            from .base_dataset import ObjectDetectionDataset

            self.ADT = ObjectDetectionDataset(
                meta_dir=meta_dir_dict["ADT"],
                dataset_name="ADT",
                default_scale=default_scales["ADT"],
                base_dir=base_dir["ADT"],
                **dataset_cfg,
            )
        self.datasets = [_.replace("-", "") for _ in dataset_names]
        self.dataset_cls = {_: getattr(self, _) for _ in self.datasets}
        self.length = 100 * num_workers

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 为每个数据集创建迭代器
        dataset_iters = {name: iter(self.dataset_cls[name]) for name in self.datasets}
        target_samples = max(len(self) // worker_info.num_workers, 1)
        
        count = 0
        while count < target_samples:
            # 随机选择数据集
            dataset = random.choices(self.datasets, weights=self.weights)[0]
            
            try:
                sample = next(dataset_iters[dataset])
                yield sample
                count += 1
            except StopIteration:
                # 重新创建该数据集的迭代器
                dataset_iters[dataset] = iter(self.dataset_cls[dataset])
                sample = next(dataset_iters[dataset])
                yield sample
                count += 1
            except Exception as e:
                print(f"Error occurred while fetching data from {dataset}: {e}")
                continue

    def get_dataloader(self, **kwargs):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=False,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = kwargs.get("prefetch_factor", 2)
        return DataLoader(self, **loader_kwargs)
