import torch
import numpy as np

import argparse
import os, cv2, json
from torch.utils.data import DataLoader
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SPATIAL_ENCODER_ROOT = os.path.join(REPO_ROOT, "Train", "SpatialEncoder")
if SPATIAL_ENCODER_ROOT not in sys.path:
    sys.path.insert(0, SPATIAL_ENCODER_ROOT)

from modeling.build_sam import build_sam2_video_predictor
from training.dataset.demo_dataset import ExampleDataset
from training.dataset.path_utils import resolve_existing_path
from training.dataset.transforms import ComposeAPI, ToTensor, NormalizeTensor


def collect_fn(batch):
    return batch[0]


def get_pred_root():
    output_root = os.environ.get("BOX_OUTPUT_PATH", os.getcwd())
    return os.path.join(output_root, "BoxDet", "pred")


def make_model():
    box_weights_path = os.environ["BOX_WEIGHTS_PATH"]
    if not os.path.isfile(os.path.join(box_weights_path, "sam2.1_hiera_large.pt")):
        nested_weights_path = os.path.join(box_weights_path, "weights")
        if os.path.isfile(os.path.join(nested_weights_path, "sam2.1_hiera_large.pt")):
            box_weights_path = nested_weights_path

    checkpoint = os.environ.get(
        "BOX_EVAL_CKPT",
        os.path.join(
            box_weights_path, "Exps/BoxDet/CA1M_Hyperism/checkpoints/checkpoint.pt"
        ),
    )
    sam_checkpoint = os.path.join(box_weights_path, "sam2.1_hiera_large.pt")
    for path_name, path in (
        ("BOX_EVAL_CKPT", checkpoint),
        ("SAM checkpoint", sam_checkpoint),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path_name} not found: {path}. "
                "Set BOX_WEIGHTS_PATH to the directory containing "
                "sam2.1_hiera_large.pt, or set BOX_EVAL_CKPT to an "
                "explicit checkpoint.pt."
            )
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(
        model_cfg,
        checkpoint,
        spatial_pretrained_path=None,
        sam_checkpoint=sam_checkpoint,
    )
    return predictor


DATASET_CONFIGS = {
    "arkitscenes": {
        "sample_meta_path": "arkitscenes/val-json/{scene_id}.json",
        "name": "vsibench-arkitscenes",
        "base_dir": "vsibench-arkitscenes/",
        "clip_length": 16,
        "slide_stride": 8,
    },
    "scannet": {
        "sample_meta_path": "vsibench-scannet/val-json/{scene_id}.json",
        "name": "vsibench-scannet",
        "base_dir": "vsibench-scannet",
        "clip_length": 16,
        "slide_stride": 8,
    },
    "scannetpp": {
        "sample_meta_path": "scannetpp_data/processing/val_frame_meta/{scene_id}.json",
        "name": "vsibench-scannetpp",
        "base_dir": "scannetpp_data/processed_data",
        "clip_length": 8,
        "slide_stride": 8,
    },
}


def inference(
    scene_id=42446540, save_dir_name="eval_sample", predictor=None, dataset_name="scannet"
):
    depthanything_config = {
        "predict_metric": True,
        "norm_scale": 2.5,
        "spatial_type": "depthanythingv3",
        "spatial_resolution": 504,
    }
    assert predictor is not None, "Please provide a valid predictor model."

    box_data_path = os.environ["BOX_DATA_PATH"]
    pred_root = get_pred_root()
    scene_save_dir = os.path.join(pred_root, save_dir_name, str(scene_id))
    done_flag = os.path.join(scene_save_dir, "done.flag")
    if os.path.exists(done_flag):
        print(f"[skip] {scene_id} already done (found {done_flag})")
        return
    # dataset = ExampleDataset(
    #     sample_meta_path=f"{box_data_path}/ARKitScenes_data/val_frame_meta2/{scene_id}.json",
    #     name="vsibench-arkitscenes",
    #     default_scale=1000.0,
    #     base_dir=f"{box_data_path}/ARKitScenes_data/",
    #     resolution=1024,
    #     eval_transforms=ComposeAPI(
    #         [
    #             ToTensor(),
    #             NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     ),
    #     clip_length=8,
    #     slide_stride=8,
    #     max_objects_per_frame=30,
    #     **depthanything_config,
    # )
    # dataset = ExampleDataset(
    #     sample_meta_path=f"{box_data_path}/scannetpp_data/processing/val_frame_meta/{scene_id}.json",
    #     name="vsibench-scannetpp",
    #     default_scale=1000.0,
    #     base_dir=f"{box_data_path}/scannetpp_data/processed_data",
    #     resolution=1024,
    #     eval_transforms=ComposeAPI(
    #         [
    #             ToTensor(),
    #             NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     ),
    #     clip_length=8,
    #     slide_stride=8,
    #     max_objects_per_frame=30,
    #     **depthanything_config,
    # )
    # dataset = ExampleDataset(
    #     sample_meta_path=f"{box_data_path}/arkitscenes/val-json/{scene_id}.json",
    #     name="vsibench-arkitscenes",
    #     default_scale=1000.0,
    #     base_dir=f"{box_data_path}/vsibench-arkitscenes/",
    #     resolution=1024,
    #     eval_transforms=ComposeAPI(
    #         [
    #             ToTensor(),
    #             NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     ),
    #     clip_length=16,
    #     slide_stride=8,
    #     max_objects_per_frame=30,
    #     **depthanything_config,
    # )
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Expected one of: {', '.join(sorted(DATASET_CONFIGS))}"
        )

    dataset_config = DATASET_CONFIGS[dataset_name]
    sample_meta_rel = dataset_config["sample_meta_path"].format(scene_id=scene_id)
    sample_meta_path = resolve_existing_path(
        os.path.join(box_data_path, sample_meta_rel),
        [
            os.path.join(
                box_data_path,
                "vsi-scannet",
                "vsibench-scannet",
                "val-json",
                "val_frame_meta",
                f"{scene_id}.json",
            ),
            os.path.join(
                box_data_path,
                "vsibench-scannet",
                "val-json",
                "val_frame_meta",
                f"{scene_id}.json",
            ),
        ],
    )
    base_dir = resolve_existing_path(
        os.path.join(box_data_path, dataset_config["base_dir"]),
        [
            os.path.join(box_data_path, "vsi-scannet", dataset_config["base_dir"]),
        ],
    )
    if not os.path.isfile(sample_meta_path):
        raise FileNotFoundError(f"Sample metadata not found: {sample_meta_path}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dataset base directory not found: {base_dir}")

    dataset = ExampleDataset(
        sample_meta_path=sample_meta_path,
        name=dataset_config["name"],
        default_scale=1000.0,
        base_dir=base_dir,
        resolution=1024,
        eval_transforms=ComposeAPI(
            [
                ToTensor(),
                NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        clip_length=dataset_config["clip_length"],
        slide_stride=dataset_config["slide_stride"],
        max_objects_per_frame=30,
        **depthanything_config,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        prefetch_factor=2,
        collate_fn=collect_fn,
    )

    # =========================================
    # dataset = ExampleDataset(
    #     sample_meta_path=f"{box_data_path}/aria_digital_twin/train-json/{scene_id}.json",
    #     name="ADT",
    #     default_scale=1000.0,
    #     base_dir=f"{box_data_path}/aria_digital_twin/ADT",
    #     resolution=1024,
    #     eval_transforms=ComposeAPI(
    #         [
    #             ToTensor(),
    #             NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     ),
    #     clip_length=8,
    #     **depthanything_config,
    # )
    # =========================================
    # dataset = ExampleDataset(
    #     sample_meta_path=f"{box_data_path}/CA-1M/val-json/{scene_id}.json",
    #     name="CA-1M",
    #     default_scale=1000.0,
    #     base_dir=f"{box_data_path}/CA-1M/val-unzip/",
    #     resolution=1024,
    #     eval_transforms=ComposeAPI(
    #         [
    #             ToTensor(),
    #             NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ]
    #     ),
    #     clip_length=8,
    #     **depthanything_config,
    # )
    # =========================================

    ious_2d = []
    ious_3d = []
    for iter_idx, sample in enumerate(dataloader):
        print("================================")
        print(f"propogating sample {iter_idx}/{len(dataloader)}")

        h, w = sample["original_size"].tolist()

        os.makedirs(os.path.join(pred_root, save_dir_name), exist_ok=True)
        scene_id = sample["scene_id"]
        save_path = os.path.join(pred_root, save_dir_name, str(scene_id))
        os.makedirs(save_path, exist_ok=True)

        image_paths = sample["image_paths"]
        image_name_0 = os.path.basename(image_paths[0]).split(".")[0]
        if image_name_0 == "image":
            image_name_0 = f"image_{iter_idx:03d}"
        if sample["valid"] is False:
            continue

        # Check if object_ids is available
        if sample["object_ids"] is None or sample["bboxes_2d"] is None:
            print(f"Skipping clip {iter_idx}: object_ids or bboxes are None")
            continue

        # Check the first frame data
        if sample["object_ids"][0] is None or sample["bboxes_2d"][0] is None:
            print(f"Skipping clip {iter_idx}: object_ids[0] or bboxes_2d[0] is None")
            continue

        extrinsics = sample["original_extrinsics"].float().cpu().numpy()
        image_paths = sample["image_paths"]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            # Get all boxes for this clip
            boxes_all = sample["bboxes_2d"][0]  # [N,4], xyxy format
            object_ids_all = sample["object_ids"][0]  # [N,]

            # Process in batches of 8 objects
            for box_iter in range(0, boxes_all.shape[0], 8):
                # Initialize state for each batch
                state = predictor.init_state(video_frames=sample)

                boxes = boxes_all[box_iter : box_iter + 8]
                batch_object_ids = (
                    object_ids_all[box_iter : box_iter + 8]
                    if object_ids_all is not None
                    else None
                )

                if batch_object_ids is None:
                    print(
                        f"Warning: No object IDs for batch starting at {box_iter}, skipping"
                    )
                    continue

                # Track mapping from out_obj_id to global index
                obj_id_to_global_idx = {}

                for i, box in enumerate(boxes):
                    if box.mean() <= 0:
                        print(f"Skipping object {i} with box {box}")
                        continue

                    obj_idx = box_iter + i
                    if obj_idx >= len(object_ids_all):
                        print(
                            f"Warning: object_ids index {obj_idx} out of range (length {len(object_ids_all)}), skipping"
                        )
                        continue

                    center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    points = torch.tensor([center])
                    print(f"Adding object {obj_idx} with box {box}")

                    obj_id = object_ids_all[obj_idx]
                    obj_id_to_global_idx[obj_id] = obj_idx  # Store mapping

                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                        state,
                        frame_idx=0,
                        points=points,
                        labels=torch.tensor([1]),
                        box=box,
                        obj_id=obj_id,
                    )

                video_bboxes_3d = {}

                for out_dict in predictor.propagate_in_video(state):
                    out_frame_idx = out_dict["frame_idx"]
                    out_obj_ids = out_dict["obj_ids"]
                    has_object = out_dict["has_object"]

                    print(
                        "Objects present in this frame:",
                        [
                            out_obj_id
                            for i, out_obj_id in enumerate(out_obj_ids)
                            if has_object[i]
                        ],
                    )
                    print(
                        f"Processing frame {out_frame_idx} with objects {out_obj_ids}"
                    )
                    image_name = os.path.basename(image_paths[out_frame_idx])[
                        : -len(".png")
                    ]
                    if image_name == "image":
                        image_name = f"frame{out_frame_idx:03d}"
                    image_name = f"{image_name_0}_{image_name}"

                    video_bbox_3d = out_dict["all_pred_bbox_3d"]
                    intrinsics = out_dict["intrinsics"].cpu().numpy()[0]
                    extrinsics_homo = extrinsics[out_frame_idx]
                    current_pose = np.linalg.inv(extrinsics_homo)
                    quaternions = out_dict["all_pred_bbox_quat"].cpu().numpy()

                    video_bboxes_3d[out_frame_idx] = {
                        out_obj_id: (video_bbox_3d[i], quaternions[i])
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    # ## save dict as json
                    os.makedirs(os.path.join(save_path, "json"), exist_ok=True)
                    os.makedirs(
                        os.path.join(save_path, "json", f"{iter_idx:03d}"),
                        exist_ok=True,
                    )
                    with open(
                        os.path.join(
                            save_path, f"json/{iter_idx:03d}/{image_name}.jsonl"
                        ),
                        "a",
                    ) as f:
                        for i, out_obj_id in enumerate(out_obj_ids):
                            if not has_object[i]:
                                continue

                            # Get global index for this object
                            global_idx = obj_id_to_global_idx.get(out_obj_id)
                            if global_idx is None:
                                print(
                                    f"Warning: Cannot find global index for {out_obj_id}, skipping"
                                )
                                continue

                            if (
                                sample["bboxes_center_3d"][out_frame_idx][global_idx]
                                .float()
                                .cpu()
                                .numpy()
                                .mean()
                                < 0
                            ):
                                continue
                            bbox3d = (
                                video_bboxes_3d[out_frame_idx][out_obj_id][0]
                                .cpu()
                                .numpy()
                            )
                            quaternions = video_bboxes_3d[out_frame_idx][out_obj_id][1]

                            x, y, z, w, h, l = bbox3d[:6].tolist()
                            conf = bbox3d[-1]
                            wq, xq, yq, zq = quaternions.tolist()

                            # Object of type float32 is not JSON serializable

                            save_dict = {
                                "conf2d": float(
                                    sample["bboxes_trunc_2d"][out_frame_idx][global_idx]
                                    .cpu()
                                    .numpy()
                                ),
                                "conf3d": float(conf),
                                "label": str(out_obj_id),
                                "bbox_3d": [x, y, z, w, h, l],
                                "quaternion": [wq, xq, yq, zq],
                                "intrinsics": intrinsics.tolist(),
                                "camera_pose": current_pose.tolist(),
                            }
                            json.dump(save_dict, f)
                            f.write("\n")

                # Clean up memory after each batch
                del state
                del video_bboxes_3d
                if "out_dict" in locals():
                    del out_dict
                torch.cuda.empty_cache()
                print(f"Batch {box_iter // 8 + 1} completed, memory cleared")

    print("Mean 2D IoU:", np.array(ious_2d).mean())
    print("Mean 3D IoU:", np.array(ious_3d).mean())

    os.makedirs(scene_save_dir, exist_ok=True)
    with open(done_flag, "w") as f:
        f.write("done\n")
    print(f"[done] {scene_id}")


def main(shard_path, save_dir_name, dataset_name="scannet"):
    predictor = make_model()
    scene_ids = []
    with open(shard_path, "r") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id:
                scene_ids.append(scene_id)
    for scene_id in scene_ids:
        inference(
            scene_id=scene_id,
            save_dir_name=save_dir_name,
            predictor=predictor,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_path")
    parser.add_argument("save_dir_name")
    parser.add_argument("dataset_name", nargs="?", default="scannet")
    args = parser.parse_args()
    main(args.shard_path, args.save_dir_name, args.dataset_name)
