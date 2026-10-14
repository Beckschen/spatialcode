import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append(".")


def get_pred_root():
    output_root = os.environ.get("BOX_OUTPUT_PATH", os.getcwd())
    return os.path.join(output_root, "BoxDet", "pred")


def first_existing_path(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(p) for p in paths if p))


def normalize_label(label: str) -> str:
    """Normalize predicted label to match GT category names."""
    label = label.lower().strip()

    MULTI_WORD_MAP = {
        "monitor tv":        "tv",
        "bookshelf closet":  "bookshelf",
        "table chair":       "table",
        "table counter":     "table",
        "chair sofa":        "sofa",
        "computer mouse":    "computer mouse",
        # ARKitScenes
        "cabinet shelf":     "cabinet",
        "tv _ monitor":      "tv_monitor",
        "tv monitor":        "tv_monitor",
        "bed chair":         "bed",
        "bed table":         "bed",
        "sofa bed":          "sofa",
        "sofa chair":        "sofa",
        "stool chair":       "stool",
        "table stool":       "table",
        "dishwasher washer": "dishwasher",
        "oven refrigerator": "oven",
        "oven stove":        "stove",
    }
    if label in MULTI_WORD_MAP:
        return MULTI_WORD_MAP[label]

    if "tv_monitor" in label:
        return "tv_monitor"
    if "tv" in label or "television" in label:
        return "tv"
    if "monitor" in label:
        return "monitor"

    return label.split("_")[0]


def get_valid_categories(all_gt_boxes_per_cat, min_box_side):
    """Return categories whose average min-side >= min_box_side.

    Args:
        all_gt_boxes_per_cat: dict {cat: [{"scale": (l,w,h), ...}, ...]}
        min_box_side: float threshold (0 = no filtering).
    """
    if min_box_side <= 0:
        return set(all_gt_boxes_per_cat.keys())
    valid = set()
    for cat, boxes in all_gt_boxes_per_cat.items():
        if not boxes:
            continue
        avg_min_side = np.mean([min(b["scale"]) for b in boxes])
        if avg_min_side >= min_box_side:
            valid.add(cat)
    return valid


def load_merged_predictions(file_path, dataset="arkitscenes", has_obj_id=None):
    """
    Load merged predictions from a JSONL file.

    Args:
        file_path:   JSONL file with one prediction per line.
        dataset:     "arkitscenes" or "scannet" — controls rotation post-processing
                     (arkitscenes zeroes out x/y tilt).
        has_obj_id:  Whether the `label` field is a per-object id ("chair_0") or
                     a category name ("chair").
                       True  → returns {obj_id: single dict}   (1-to-1 lookup)
                       False → returns {category: [list]}      (Hungarian matching)
                       None  → infer: arkitscenes=False, scannet=True
    """
    if has_obj_id is None:
        has_obj_id = (dataset != "arkitscenes")

    bboxes = [json.loads(s) for s in open(file_path).readlines()]
    pred_dict = {}
    for bbox in bboxes:
        center = np.array(bbox["bbox_3d"][:3])
        # reorder [w,h,l] → [w,l,h] to match GT axesLengths convention (both datasets)
        scale = np.array(
            [bbox["bbox_3d"][3]] + [bbox["bbox_3d"][5]] + [bbox["bbox_3d"][4]]
        )
        if dataset == "arkitscenes":
            euler = R.from_quat(bbox["bbox_3d"][6:]).as_euler("xyz")
            euler[0] = 0  # zero out x/y tilt — ARKitScenes is gravity-aligned
            euler[1] = 0
            rot_mat = R.from_euler("xyz", euler).as_matrix()
        else:
            rot_mat = R.from_quat(bbox["bbox_3d"][6:]).as_matrix()

        entry = {"center": center, "scale": scale, "rotation": rot_mat}
        if has_obj_id:
            pred_dict[bbox["label"]] = entry
        else:
            label = normalize_label(bbox["label"])
            pred_dict.setdefault(label, []).append(entry)
    return pred_dict


def load_ground_truths(file_path):
    """Load ARKitScenes ground truth from a JSON file."""
    rt_gt = {}
    with open(file_path, "r") as f:
        gt_data = json.load(f)
    for obj in gt_data["data"]:
        cat = obj["label"]
        x, y, z = obj["segments"]["obbAligned"]["centroid"]
        l, w, h = obj["segments"]["obbAligned"]["axesLengths"]
        rot = obj["segments"]["obbAligned"]["normalizedAxes"]
        rot = np.array(rot).reshape(3, 3).T
        rt_gt.setdefault(cat, [])
        obj_dict = {"center": (x, y, z), "scale": (l, w, h), "rotation": rot}
        rt_gt[cat].append(obj_dict)
    return rt_gt


def compute_3d_bbox_vertices(
    x, y, z, w, h, l, yaw=0.0, rotation_matrix=None, device=None
):
    # 8 corners relative to center
    corners = np.array(
        [
            [w / 2, h / 2, l / 2],
            [w / 2, -h / 2, l / 2],
            [-w / 2, -h / 2, l / 2],
            [-w / 2, h / 2, l / 2],
            [w / 2, h / 2, -l / 2],
            [w / 2, -h / 2, -l / 2],
            [-w / 2, -h / 2, -l / 2],
            [-w / 2, h / 2, -l / 2],
        ],
        dtype=np.float32,
    )

    if rotation_matrix is not None and (isinstance(yaw, float) and yaw == 0):
        Rot = rotation_matrix
    elif rotation_matrix is None and (not isinstance(yaw, float) or yaw != 0):
        Rot = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
    elif rotation_matrix is not None and (not isinstance(yaw, float) or yaw != 0):
        R_yaw = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        Rot = rotation_matrix @ R_yaw
    else:
        Rot = np.eye(3, dtype=np.float32)

    corners_rotated = corners @ Rot.T
    translation = np.array([x, y, z], dtype=np.float32)
    corners_translated = corners_rotated + translation
    return corners_translated


def convex_hull_intersection(p1, p2):
    poly1 = Polygon(p1).convex_hull
    poly2 = Polygon(p2).convex_hull
    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return inter_area


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def box3d_iou(corners1, corners2, rot_to_z_up=False):
    if rot_to_z_up:
        rect1 = np.array([(corners1[i, 0], corners1[i, 1]) for i in [0, 3, 7, 4]])
        rect2 = np.array([(corners2[i, 0], corners2[i, 1]) for i in [0, 3, 7, 4]])
        inter_area = convex_hull_intersection(rect1, rect2)
        z1_min, z1_max = corners1[:, 2].min(), corners1[:, 2].max()
        z2_min, z2_max = corners2[:, 2].min(), corners2[:, 2].max()
        z_max = min(z1_max, z2_max)
        z_min = max(z1_min, z2_min)
        inter_h = max(0.0, z_max - z_min)
        inter_vol = inter_area * inter_h
        vol1 = box3d_vol(corners1)
        vol2 = box3d_vol(corners2)
        iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-6)
        return iou

    rect1 = np.array([(corners1[i, 0], corners1[i, 2]) for i in [0, 3, 4, 7]])
    rect2 = np.array([(corners2[i, 0], corners2[i, 2]) for i in [0, 3, 4, 7]])
    inter_area = convex_hull_intersection(rect1, rect2)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[1, 1], corners2[1, 1])
    inter_vol = inter_area * max(max(0.0, ymax - ymin), ymin - ymax)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou


def iou_aabb(corners1, corners2):
    x_min1, y_min1, z_min1 = np.min(corners1, axis=0)
    x_max1, y_max1, z_max1 = np.max(corners1, axis=0)
    x_min2, y_min2, z_min2 = np.min(corners2, axis=0)
    x_max2, y_max2, z_max2 = np.max(corners2, axis=0)

    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_zmin = max(z_min1, z_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)
    inter_zmax = min(z_max1, z_max2)

    inter_dx = max(0.0, inter_xmax - inter_xmin)
    inter_dy = max(0.0, inter_ymax - inter_ymin)
    inter_dz = max(0.0, inter_zmax - inter_zmin)
    inter_vol = inter_dx * inter_dy * inter_dz
    vol1 = (x_max1 - x_min1) * (y_max1 - y_min1) * (z_max1 - z_min1)
    vol2 = (x_max2 - x_min2) * (y_max2 - y_min2) * (z_max2 - z_min2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou


def scale_error(gt_scale, pred_scale):
    min_scale = np.minimum(gt_scale, pred_scale)
    volume_intersection = np.prod(min_scale)
    volume_pred = np.prod(pred_scale)
    volume_gt = np.prod(gt_scale)
    scale_iou = volume_intersection / (volume_gt + volume_pred - volume_intersection)
    return 1 - scale_iou


def get_f1(tp, fp, fn):
    """Return (f1, precision, recall)."""
    if tp + fp == 0 or tp + fn == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


# Keep old name for backward compat
def get_f1_score(tp, fp, fn):
    f1, _, _ = get_f1(tp, fp, fn)
    return f1


def _init_cat(stats, cat):
    if cat not in stats:
        stats[cat] = {"tp10": 0, "tp25": 0, "tp50": 0, "gt": 0, "pred": 0, "matched": 0}


def _print_per_cat_and_macro(category_stats, matched_only):
    cat_f1_10, cat_f1_25, cat_f1_50 = [], [], []
    cat_p10,   cat_p25,   cat_p50   = [], [], []
    cat_r10,   cat_r25,   cat_r50   = [], [], []

    gt_col = "Matched" if matched_only else "GT"
    mode_str = "matched-pairs only" if matched_only else "all GT vs all Pred"
    print(f"\n--- Per-category breakdown [{mode_str}] ---")
    print(
        f"{'Category':20s} | {gt_col:>7} {'TP@10':>6} {'TP@25':>6} {'TP@50':>6} | "
        f"{'P@10':>6} {'R@10':>6} {'F1@10':>6} | "
        f"{'P@25':>6} {'R@25':>6} {'F1@25':>6} | "
        f"{'P@50':>6} {'R@50':>6} {'F1@50':>6}"
    )
    print("-" * 118)

    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        cf1_10, cp10, cr10 = get_f1(s["tp10"], s["pred"] - s["tp10"], s["gt"] - s["tp10"])
        cf1_25, cp25, cr25 = get_f1(s["tp25"], s["pred"] - s["tp25"], s["gt"] - s["tp25"])
        cf1_50, cp50, cr50 = get_f1(s["tp50"], s["pred"] - s["tp50"], s["gt"] - s["tp50"])
        cat_f1_10.append(cf1_10); cat_p10.append(cp10); cat_r10.append(cr10)
        cat_f1_25.append(cf1_25); cat_p25.append(cp25); cat_r25.append(cr25)
        cat_f1_50.append(cf1_50); cat_p50.append(cp50); cat_r50.append(cr50)
        count = s["matched"] if matched_only else s["gt"]
        print(
            f"{cat:20s} | {count:>7} {s['tp10']:>6} {s['tp25']:>6} {s['tp50']:>6} | "
            f"{cp10:>6.3f} {cr10:>6.3f} {cf1_10:>6.3f} | "
            f"{cp25:>6.3f} {cr25:>6.3f} {cf1_25:>6.3f} | "
            f"{cp50:>6.3f} {cr50:>6.3f} {cf1_50:>6.3f}"
        )

    n = len(cat_f1_25)
    print(f"\n--- Macro F1 (mean over {n} categories) ---")
    print(f"F1@0.1:  {np.mean(cat_f1_10):.4f}  (P: {np.mean(cat_p10):.4f}, R: {np.mean(cat_r10):.4f})")
    print(f"F1@0.25: {np.mean(cat_f1_25):.4f}  (P: {np.mean(cat_p25):.4f}, R: {np.mean(cat_r25):.4f})")
    print(f"F1@0.5:  {np.mean(cat_f1_50):.4f}  (P: {np.mean(cat_p50):.4f}, R: {np.mean(cat_r50):.4f})")


def arkitscenes(
    pred_dir="arkitscenes",
    exp_name="merged_bbox_3dconf_top5meanwhl1r_timestamp",
    min_box_side=0.0,
    matched_only=False,
):
    """Evaluate on ARKitScenes.

    Args:
        pred_dir: Subdirectory under $BOX_OUTPUT_PATH/BoxDet/pred/ containing predictions.
        min_box_side: Exclude categories whose average GT min-side < threshold (m).
        matched_only: If True, compute F1 only over Hungarian-matched pairs.
    """
    box_data_path = os.environ["BOX_DATA_PATH"]
    gt_files = glob(f"{box_data_path}/arkitscenes/ground_truth/*.json")
    pred_base = os.path.join(get_pred_root(), pred_dir)

    # Pre-pass: collect all GT boxes per category for min_box_side filtering
    all_gt_per_cat = {}
    for gt_file in gt_files:
        gt_dict = load_ground_truths(gt_file)
        for cat, boxes in gt_dict.items():
            all_gt_per_cat.setdefault(cat, []).extend(boxes)
    valid_cats = get_valid_categories(all_gt_per_cat, min_box_side)
    excluded = sorted(set(all_gt_per_cat.keys()) - valid_cats)
    if excluded:
        print(f"Excluded categories (avg min-side < {min_box_side}m): {excluded}")

    metrics = {
        "iou_3d": [],
        "depth_error": [],
        "scale_error": [],
        "rotation_error": [],
        "translation_error": [],
        "gt_num": 0,
        "pred_num": 0,
        "tp10": 0,
        "tp25": 0,
        "tp50": 0,
    }
    category_stats = {}

    for gt_file in tqdm(gt_files):
        scene_name = os.path.basename(gt_file).replace("_3dod_annotation.json", "")
        pred_scene_dir = os.path.join(pred_base, scene_name, f"{exp_name}.json")
        gt_dict = load_ground_truths(gt_file)

        # Filter to valid categories
        gt_dict = {k: v for k, v in gt_dict.items() if k in valid_cats}

        if not os.path.exists(pred_scene_dir):
            if not matched_only:
                for cat, boxes in gt_dict.items():
                    _init_cat(category_stats, cat)
                    category_stats[cat]["gt"] += len(boxes)
                    metrics["gt_num"] += len(boxes)
            continue

        pred_dict = load_merged_predictions(pred_scene_dir, dataset="arkitscenes")

        if not matched_only:
            metrics["pred_num"] += sum(len(v) for v in pred_dict.values())

        for cls_name, gt_boxes in gt_dict.items():
            pred_boxes = pred_dict.get(cls_name, [])
            _init_cat(category_stats, cls_name)

            if not matched_only:
                category_stats[cls_name]["gt"] += len(gt_boxes)
                category_stats[cls_name]["pred"] += len(pred_boxes)
                metrics["gt_num"] += len(gt_boxes)

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    gt_corners = compute_3d_bbox_vertices(
                        gt_box["center"][0], gt_box["center"][1], gt_box["center"][2],
                        gt_box["scale"][0], gt_box["scale"][1], gt_box["scale"][2],
                        rotation_matrix=gt_box["rotation"],
                    )
                    pred_corners = compute_3d_bbox_vertices(
                        pred_box["center"][0], pred_box["center"][1], pred_box["center"][2],
                        pred_box["scale"][0], pred_box["scale"][1], pred_box["scale"][2],
                        rotation_matrix=pred_box["rotation"],
                    )
                    iou_matrix[i, j] = box3d_iou(gt_corners, pred_corners)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            n_matched = len(row_ind)

            if matched_only:
                category_stats[cls_name]["matched"] += n_matched
                category_stats[cls_name]["gt"] += n_matched
                category_stats[cls_name]["pred"] += n_matched
                metrics["gt_num"] += n_matched
                metrics["pred_num"] += n_matched

            for r, c in zip(row_ind, col_ind):
                iou_3d = iou_matrix[r, c]
                metrics["iou_3d"].append(iou_3d)
                if iou_3d >= 0.1:
                    metrics["tp10"] += 1
                    category_stats[cls_name]["tp10"] += 1
                if iou_3d >= 0.25:
                    metrics["tp25"] += 1
                    category_stats[cls_name]["tp25"] += 1
                if iou_3d >= 0.5:
                    metrics["tp50"] += 1
                    category_stats[cls_name]["tp50"] += 1

                gt_box = gt_boxes[c]
                pred_box = pred_boxes[r]
                depth_error = np.abs(gt_box["center"][2] - pred_box["center"][2])
                metrics["depth_error"].append(depth_error)
                scale_err = scale_error(
                    np.array(gt_box["scale"]), np.array(pred_box["scale"])
                )
                metrics["scale_error"].append(scale_err)
                pred_yaw = R.from_matrix(pred_box["rotation"]).as_euler("xyz")[2]
                gt_yaw = R.from_matrix(gt_box["rotation"]).as_euler("xyz")[2]
                rotation_error = np.abs(pred_yaw - gt_yaw)
                rotation_error = min(rotation_error, 2 * np.pi - rotation_error)
                metrics["rotation_error"].append(rotation_error)
                translation_error = np.linalg.norm(
                    np.array(gt_box["center"]) - np.array(pred_box["center"])
                )
                metrics["translation_error"].append(translation_error)

    gt_num  = metrics["gt_num"]
    pred_num = metrics["pred_num"]
    tp10, tp25, tp50 = metrics["tp10"], metrics["tp25"], metrics["tp50"]
    fp10, fp25, fp50 = pred_num - tp10, pred_num - tp25, pred_num - tp50
    fn10, fn25, fn50 = gt_num - tp10,  gt_num - tp25,  gt_num - tp50

    f1_10, prec_10, rec_10 = get_f1(tp10, fp10, fn10)
    f1_25, prec_25, rec_25 = get_f1(tp25, fp25, fn25)
    f1_50, prec_50, rec_50 = get_f1(tp50, fp50, fn50)

    aggregated_metrics = {
        "iou_3d":            np.mean(metrics["iou_3d"])            if metrics["iou_3d"]            else 0,
        "depth_error":       np.mean(metrics["depth_error"])       if metrics["depth_error"]       else 0,
        "scale_error":       np.mean(metrics["scale_error"])       if metrics["scale_error"]       else 0,
        "rotation_error":    np.mean(metrics["rotation_error"])    if metrics["rotation_error"]    else 0,
        "translation_error": np.mean(metrics["translation_error"]) if metrics["translation_error"] else 0,
        "gt_num":   gt_num,
        "pred_num": pred_num,
        "tp10": tp10, "tp25": tp25, "tp50": tp50,
        "fp10": fp10, "fp25": fp25, "fp50": fp50,
        "fn10": fn10, "fn25": fn25, "fn50": fn50,
        "f1@0.1":  f1_10,  "precision@0.1":  prec_10,  "recall@0.1":  rec_10,
        "f1@0.25": f1_25,  "precision@0.25": prec_25,  "recall@0.25": rec_25,
        "f1@0.5":  f1_50,  "precision@0.5":  prec_50,  "recall@0.5":  rec_50,
    }

    print(f"\n{'='*60}")
    print("ARKitScenes Results:")
    print(f"{'='*60}")
    print(f"Total GT:   {gt_num}")
    print(f"Total Pred: {pred_num}")
    print(f"TP@0.1:     {tp10}")
    print(f"TP@0.25:    {tp25}")
    print(f"TP@0.5:     {tp50}")
    print(f"\n--- Micro F1 (global) ---")
    print(f"F1@0.1:  {f1_10:.4f}  (P: {prec_10:.4f}, R: {rec_10:.4f})")
    print(f"F1@0.25: {f1_25:.4f}  (P: {prec_25:.4f}, R: {rec_25:.4f})")
    print(f"F1@0.5:  {f1_50:.4f}  (P: {prec_50:.4f}, R: {rec_50:.4f})")

    _print_per_cat_and_macro(category_stats, matched_only)
    print(f"{'='*60}\n")

    return aggregated_metrics


def get_scannet_gt(gt_data):
    gt_dict = {}
    for cat, obj_list in gt_data["object_bbox"].items():
        for idx, obj in enumerate(obj_list):
            obj_id = f"{cat}_{idx}"
            x, y, z = obj["centroid"]
            l, w, h = obj["axesLengths"]
            rot = obj["normalizedAxes"]
            rot = np.array(rot).reshape(3, 3)
            rot = rot.T
            gt_dict[obj_id] = {}
            gt_dict[obj_id]["center"] = (x, y, z)
            gt_dict[obj_id]["scale"] = (l, w, h)
            gt_dict[obj_id]["rotation"] = rot
    return gt_dict


def _record_scannet_metrics(gt_box, pred_box, iou_3d, cat, metrics, category_stats):
    """Append per-pair errors and bump TP counters for a matched (gt, pred)."""
    metrics["iou_3d"].append(iou_3d)
    if iou_3d >= 0.1:
        metrics["tp10"] += 1
        category_stats[cat]["tp10"] += 1
    if iou_3d > 0.25:
        metrics["tp25"] += 1
        category_stats[cat]["tp25"] += 1
    if iou_3d > 0.5:
        metrics["tp50"] += 1
        category_stats[cat]["tp50"] += 1

    metrics["depth_error"].append(np.abs(gt_box["center"][2] - pred_box["center"][2]))

    gt_s = sorted(gt_box["scale"])
    pred_s = sorted(pred_box["scale"])
    metrics["scale_error"].append(scale_error(np.array(gt_s), np.array(pred_s)))
    metrics["translation_error"].append(
        np.linalg.norm(np.array(gt_box["center"]) - np.array(pred_box["center"]))
    )


def _scannet_eval_obj_id(
    gt_dict, pred_dict, scene_name, valid_cats,
    metrics, category_stats, matched_only,
):
    """1-to-1 matching by obj_id (predictions keyed by 'chair_0' etc.)."""
    for obj_id, gt_box in gt_dict.items():
        cat = "_".join(obj_id.split("_")[:-1])
        if cat not in valid_cats:
            continue
        _init_cat(category_stats, cat)

        if obj_id not in pred_dict:
            print("Missing object:", obj_id, "in scene:", scene_name)
            if not matched_only:
                category_stats[cat]["gt"] += 1
                metrics["gt_num"] += 1
            continue

        pred_box = pred_dict[obj_id]

        if not matched_only:
            category_stats[cat]["gt"] += 1
            category_stats[cat]["pred"] += 1
            metrics["gt_num"] += 1

        gt_corners = compute_3d_bbox_vertices(
            gt_box["center"][0], gt_box["center"][1], gt_box["center"][2],
            gt_box["scale"][0],  gt_box["scale"][1],  gt_box["scale"][2],
            rotation_matrix=gt_box["rotation"],
        )
        pred_corners = compute_3d_bbox_vertices(
            pred_box["center"][0], pred_box["center"][1], pred_box["center"][2],
            pred_box["scale"][0],  pred_box["scale"][1],  pred_box["scale"][2],
            rotation_matrix=pred_box["rotation"],
        )
        iou_3d = iou_aabb(gt_corners, pred_corners)

        if matched_only:
            category_stats[cat]["matched"] += 1
            category_stats[cat]["gt"] += 1
            category_stats[cat]["pred"] += 1
            metrics["gt_num"] += 1
            metrics["pred_num"] += 1

        _record_scannet_metrics(gt_box, pred_box, iou_3d, cat, metrics, category_stats)


def _scannet_eval_hungarian(
    gt_dict, pred_dict, valid_cats,
    metrics, category_stats, matched_only,
):
    """Per-category Hungarian matching (predictions keyed by category name)."""
    # Group GT by category
    gt_by_cat = {}
    for obj_id, gt_box in gt_dict.items():
        cat = "_".join(obj_id.split("_")[:-1])
        if cat not in valid_cats:
            continue
        gt_by_cat.setdefault(cat, []).append(gt_box)

    for cat, gt_boxes in gt_by_cat.items():
        _init_cat(category_stats, cat)
        pred_boxes = pred_dict.get(cat, [])

        if not matched_only:
            category_stats[cat]["gt"] += len(gt_boxes)
            category_stats[cat]["pred"] += len(pred_boxes)
            metrics["gt_num"] += len(gt_boxes)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                gt_corners = compute_3d_bbox_vertices(
                    gt_box["center"][0], gt_box["center"][1], gt_box["center"][2],
                    gt_box["scale"][0],  gt_box["scale"][1],  gt_box["scale"][2],
                    rotation_matrix=gt_box["rotation"],
                )
                pred_corners = compute_3d_bbox_vertices(
                    pred_box["center"][0], pred_box["center"][1], pred_box["center"][2],
                    pred_box["scale"][0],  pred_box["scale"][1],  pred_box["scale"][2],
                    rotation_matrix=pred_box["rotation"],
                )
                iou_matrix[i, j] = iou_aabb(gt_corners, pred_corners)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        n_matched = len(row_ind)

        if matched_only:
            category_stats[cat]["matched"] += n_matched
            category_stats[cat]["gt"] += n_matched
            category_stats[cat]["pred"] += n_matched
            metrics["gt_num"] += n_matched
            metrics["pred_num"] += n_matched

        for r, c in zip(row_ind, col_ind):
            _record_scannet_metrics(
                gt_boxes[c], pred_boxes[r], iou_matrix[r, c],
                cat, metrics, category_stats,
            )


def scannet(
    pred_dir="scannet",
    exp_name="merged_bbox_3dconf_top10meanwhl1r_timestamp",
    min_box_side=0.0,
    matched_only=False,
    has_obj_id=True,
):
    """Evaluate on ScanNet.

    Args:
        pred_dir: Subdirectory under $BOX_OUTPUT_PATH/BoxDet/pred/ containing predictions.
        min_box_side: Exclude categories whose average GT min-side < threshold (m).
        matched_only: If True, skip unmatched GT objects in F1 counting.
        has_obj_id: If True, predictions are keyed by obj_id (e.g. "chair_0") and
            matched 1-to-1 with GT. If False, predictions are category-only and
            matched per category via Hungarian assignment (like ARKitScenes).
    """
    box_data_path = os.environ["BOX_DATA_PATH"]
    gt_path = first_existing_path(
        os.environ.get("BOX_SCANNET_META_PATH"),
        "meta/vsi_scannet_meta.json",
        os.path.join(
            box_data_path,
            "box_code/BoxFromMotion/training/offline_evals/vsi_scannet_meta.json",
        ),
        os.path.join(box_data_path, "box_code/SpatialEncoder/inference/meta/vsi_scannet_meta.json"),
    )
    pred_base = os.path.join(get_pred_root(), pred_dir)
    translation_meta = first_existing_path(
        os.environ.get("BOX_SCANNET_TRANSLATION_PATH"),
        "meta/vsi_scannet_translation.json",
        os.path.join(
            box_data_path,
            "box_code/BoxFromMotion/training/offline_evals/vsi_scannet_translation.json",
        ),
        os.path.join(
            box_data_path,
            "box_code/SpatialEncoder/inference/meta/vsi_scannet_translation.json",
        ),
    )
    with open(translation_meta, "r") as f:
        translation_data = json.load(f)
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    # Pre-pass: collect all GT boxes per category for min_box_side filtering
    all_gt_per_cat = {}
    for scene_name in os.listdir(pred_base):
        gt_dict = get_scannet_gt(gt_data[scene_name])
        for obj_id, obj in gt_dict.items():
            cat = "_".join(obj_id.split("_")[:-1])
            all_gt_per_cat.setdefault(cat, []).append(obj)
    valid_cats = get_valid_categories(all_gt_per_cat, min_box_side)
    excluded = sorted(set(all_gt_per_cat.keys()) - valid_cats)
    if excluded:
        print(f"Excluded categories (avg min-side < {min_box_side}m): {excluded}")

    metrics = {
        "iou_3d": [],
        "depth_error": [],
        "scale_error": [],
        "translation_error": [],
        "gt_num": 0,
        "pred_num": 0,
        "tp10": 0,
        "tp25": 0,
        "tp50": 0,
    }
    category_stats = {}

    for scene_name in tqdm(os.listdir(pred_base)):
        gt_dict = get_scannet_gt(gt_data[scene_name])
        translation = np.array(translation_data[scene_name])
        pred_scene_dir = os.path.join(pred_base, scene_name, f"{exp_name}.json")

        if not os.path.exists(pred_scene_dir):
            if not matched_only:
                for obj_id in gt_dict:
                    cat = "_".join(obj_id.split("_")[:-1])
                    if cat not in valid_cats:
                        continue
                    _init_cat(category_stats, cat)
                    category_stats[cat]["gt"] += 1
                    metrics["gt_num"] += 1
            continue

        pred_dict = load_merged_predictions(
            pred_scene_dir, dataset="scannet", has_obj_id=has_obj_id,
        )

        # Apply scene translation to every GT center (shared by both branches)
        for obj_id in gt_dict:
            gt_dict[obj_id]["center"] = np.array(gt_dict[obj_id]["center"]) + translation

        if not matched_only:
            if has_obj_id:
                metrics["pred_num"] += len(pred_dict)
            else:
                metrics["pred_num"] += sum(len(v) for v in pred_dict.values())

        if has_obj_id:
            _scannet_eval_obj_id(
                gt_dict, pred_dict, scene_name, valid_cats,
                metrics, category_stats, matched_only,
            )
        else:
            _scannet_eval_hungarian(
                gt_dict, pred_dict, valid_cats,
                metrics, category_stats, matched_only,
            )

    gt_num   = metrics["gt_num"]
    pred_num = metrics["pred_num"]
    tp10, tp25, tp50 = metrics["tp10"], metrics["tp25"], metrics["tp50"]
    fp10, fp25, fp50 = pred_num - tp10, pred_num - tp25, pred_num - tp50
    fn10, fn25, fn50 = gt_num - tp10,  gt_num - tp25,  gt_num - tp50

    f1_10, prec_10, rec_10 = get_f1(tp10, fp10, fn10)
    f1_25, prec_25, rec_25 = get_f1(tp25, fp25, fn25)
    f1_50, prec_50, rec_50 = get_f1(tp50, fp50, fn50)

    aggregated_metrics = {
        "iou_3d":            np.mean(metrics["iou_3d"])            if metrics["iou_3d"]            else 0,
        "depth_error":       np.mean(metrics["depth_error"])       if metrics["depth_error"]       else 0,
        "scale_error":       np.mean(metrics["scale_error"])       if metrics["scale_error"]       else 0,
        "translation_error": np.mean(metrics["translation_error"]) if metrics["translation_error"] else 0,
        "gt_num":   gt_num,
        "pred_num": pred_num,
        "tp10": tp10, "tp25": tp25, "tp50": tp50,
        "fp10": fp10, "fp25": fp25, "fp50": fp50,
        "fn10": fn10, "fn25": fn25, "fn50": fn50,
        "f1@0.1":  f1_10,  "precision@0.1":  prec_10,  "recall@0.1":  rec_10,
        "f1@0.25": f1_25,  "precision@0.25": prec_25,  "recall@0.25": rec_25,
        "f1@0.5":  f1_50,  "precision@0.5":  prec_50,  "recall@0.5":  rec_50,
    }

    print(f"\n{'='*60}")
    print("ScanNet Results:")
    print(f"{'='*60}")
    print(f"Total GT:   {gt_num}")
    print(f"Total Pred: {pred_num}")
    print(f"TP@0.1:     {tp10}")
    print(f"TP@0.25:    {tp25}")
    print(f"TP@0.5:     {tp50}")
    print(f"\n--- Micro F1 (global) ---")
    print(f"F1@0.1:  {f1_10:.4f}  (P: {prec_10:.4f}, R: {rec_10:.4f})")
    print(f"F1@0.25: {f1_25:.4f}  (P: {prec_25:.4f}, R: {rec_25:.4f})")
    print(f"F1@0.5:  {f1_50:.4f}  (P: {prec_50:.4f}, R: {rec_50:.4f})")

    _print_per_cat_and_macro(category_stats, matched_only)
    print(f"{'='*60}\n")

    return aggregated_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute 3D detection metrics")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="arkitscenes",
        choices=["scannet", "arkitscenes"],
        help="Dataset to evaluate (default: arkitscenes)",
    )
    parser.add_argument(
        "--pred-dir", type=str, default=None,
        help="Subdirectory under $BOX_DATA_PATH/pred/ (default: same as dataset)",
    )
    parser.add_argument(
        "--exp-name", type=str, default="merged_bbox_3dconf_top5meanwhl1r_timestamp",
        help="Experiment name / prediction file stem",
    )
    parser.add_argument(
        "--min-box-side", type=float, default=0.0,
        help="Exclude categories whose avg GT min-side < threshold (m). Default: 0 (no filtering)",
    )
    parser.add_argument(
        "--matched-only", action="store_true",
        help="Compute F1 only on matched pairs; unmatched GT/Pred are ignored",
    )
    parser.add_argument(
        "--has-obj-id", dest="has_obj_id", action="store_true", default=None,
        help="Predictions' 'label' field is a per-object id like 'chair_0' "
             "(scannet 1-to-1 matching). Default: True for scannet, False for arkitscenes.",
    )
    parser.add_argument(
        "--no-obj-id", dest="has_obj_id", action="store_false",
        help="Predictions' 'label' field is a category name "
             "(forces per-category Hungarian matching).",
    )
    args = parser.parse_args()
    pred_dir = args.pred_dir or args.dataset

    if args.dataset == "arkitscenes":
        results = arkitscenes(
            pred_dir=pred_dir,
            exp_name=args.exp_name,
            min_box_side=args.min_box_side,
            matched_only=args.matched_only,
        )
    else:
        # scannet defaults to has_obj_id=True; --no-obj-id flips to Hungarian path
        scannet_kwargs = dict(
            pred_dir=pred_dir,
            exp_name=args.exp_name,
            min_box_side=args.min_box_side,
            matched_only=args.matched_only,
        )
        if args.has_obj_id is not None:
            scannet_kwargs["has_obj_id"] = args.has_obj_id
        results = scannet(**scannet_kwargs)

    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
