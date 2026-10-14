import os
from glob import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


def get_pred_root():
    output_root = os.environ.get("BOX_OUTPUT_PATH", os.getcwd())
    return os.path.join(output_root, "BoxDet", "pred")


def load_json(scene_path):
    json_files = sorted(glob(os.path.join(scene_path, "json/*/*.jsonl")))
    data = []
    for json_file in json_files:
        timestamp = json_file.split("/")[-1].split(".jsonl")[0].split("_")[-1]
        timestamp = float(timestamp) * 1e3
        added_obj = []
        with open(json_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if sample["label"] in added_obj:
                    continue
                added_obj.append(sample["label"])
                sample["timestamp"] = timestamp
                data.append(sample)

    scene_bbox = {}
    for sample in data:
        cam_pose = np.array(sample["camera_pose"])
        label = sample["label"]
        timestamp = sample["timestamp"]
        x, y, z, w, l, h = sample["bbox_3d"]
        cx, cy, cz = cam_pose[:3, :3] @ np.array([x, y, z]) * 2.5 + cam_pose[:3, 3]
        wq, xq, yq, zq = sample["quaternion"]
        rotmat = R.from_quat([xq, yq, zq, wq]).as_matrix()
        rotmat = cam_pose[:3, :3] @ rotmat
        conf = sample["conf3d"] / (sample["conf3d"] + 1)
        if label not in scene_bbox:
            scene_bbox[label] = []
        scene_bbox[label].append((cx, cy, cz, w * 2.5, l * 2.5, h * 2.5, rotmat, conf, timestamp))
    # sort by confidence
    appear_timestamp = {}
    for label in scene_bbox:
        scene_bbox[label] = sorted(scene_bbox[label], key=lambda x: x[-2], reverse=True)
        appear_timestamp[label] = min(scene_bbox[label], key=lambda x: x[-1])[-1]
    # sort by first appear timestamp
    sorted_scene_bbox = dict(
        sorted(scene_bbox.items(), key=lambda item: appear_timestamp[item[0]])
    )
    return sorted_scene_bbox


def _compute_robust_merged_bbox(bbox_params_list):
    """
    Merges multiple 3D bounding boxes by:
    1. Excluding center outliers (if > 3 boxes).
    2. Averaging the center (x, y, z) and rotation matrix (R) of inliers.
    3. Taking the maximum dimensions (w, h, l) of inliers.

    Args:
        bbox_params_list (list): List of parameter lists/arrays. Each list must be
                                 [x, y, z, w, h, l, [R00, R01, ..., R22]] (7 elements).

    Returns:
        tuple: (x, y, z, l, w, h, R) parameters for the robustly merged box.
    """
    # 1. Prepare data structures (Centers: [x, y, z], Dims: [w, h, l])
    centers = np.array([p[0:3] for p in bbox_params_list])
    dims = np.array([p[3:6] for p in bbox_params_list])
    # w_merged, h_merged, l_merged = bbox_params_list[0][3:6]
    # R_matrices = [p[6] for p in bbox_params_list[: max(len(bbox_params_list), 5)]]
    R_matrices = bbox_params_list[0][6]

    inlier_indices = np.arange(len(centers))

    if len(centers) > 5:
        # 2. Outlier Exclusion using Center Distance (MAD-based for robustness)
        median_center = np.median(centers, axis=0)
        distances = np.linalg.norm(centers - median_center, axis=1)
        # Calculate Median Absolute Deviation (MAD) for distances
        mad = np.median(np.abs(distances - np.median(distances)))
        # Threshold: 3 * MAD (robust estimation of 3-sigma)
        threshold = 3.0 * mad if mad > 1e-6 else np.max(distances) * 0.1
        inlier_indices = np.where(distances < threshold)[0]
        # Fallback: If outlier removal leaves too few boxes, use all original boxes
        if len(inlier_indices) < 2:
            inlier_indices = np.arange(len(centers))

    # Select inlier data
    inlier_centers = centers[inlier_indices]
    inlier_dims = dims[inlier_indices]

    # 3. Average Centers (x, y, z)
    x_merged, y_merged, z_merged = np.mean(inlier_centers, axis=0)

    # 4. Max Dimensions (w, h, l)

    # inlier_dims[:, 0] = W, inlier_dims[:, 1] = H, inlier_dims[:, 2] = L
    w_merged = np.mean(inlier_dims[:, 0])
    h_merged = np.mean(inlier_dims[:, 1])
    l_merged = np.mean(inlier_dims[:, 2])

    # 5. Average Rotation Matrix (R) using SVD for re-orthogonalization (Robust Averaging)
    # inlier_R_matrices = [R_matrices[i] for i in inlier_indices]
    # R_mean = np.mean(inlier_R_matrices, axis=0)

    # Final return order: x, y, z, L, W, H, R
    return x_merged, y_merged, z_merged, w_merged, h_merged, l_merged, R_matrices


def merge_boxes_in_3d(bboxes3d_dict, topk=20):
    merged_bbox3d_dict = {}
    for obj_id, bbox3d in bboxes3d_dict.items():
        merged_boxed = _compute_robust_merged_bbox(bbox3d[:topk])
        x, y, z, w, h, l, R_merged = merged_boxed

        qx, qy, qz, qw = R.from_matrix(R_merged).as_quat()
        merged_bbox3d_dict[obj_id] = (x, y, z, w, h, l, qx, qy, qz, qw)
    return merged_bbox3d_dict


def save_merged_bbox(scene_bbox, output_path):
    # save_pred_boxes as {"label": <category>, "bbox_3d": [x,y,z,w,h,l,rx,ry,rz]}
    with open(output_path, "w") as f:
        for k, v in scene_bbox.items():
            bbox_3d = list(v)
            f.write(f'{{"label": "{k}", "bbox_3d": {bbox_3d}}}\n')


def main(scene_id, pred_dir, exp_name):
    pred_root = get_pred_root()
    scene_path = os.path.join(pred_root, pred_dir, scene_id)
    scene_json = load_json(scene_path)
    merged_box = merge_boxes_in_3d(scene_json)
    save_merged_bbox(
        merged_box,
        os.path.join(pred_root, pred_dir, scene_id, f"{exp_name}.json"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--exp_name", required=True)
    args = parser.parse_args()
    main(args.scene_id, args.pred_dir, args.exp_name)
