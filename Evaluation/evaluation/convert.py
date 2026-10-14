import argparse
import json
import math
import re
from pathlib import Path


DEFAULT_MERGED_NAME = "merged_bbox_3dconf_top5meanwhl1r_timestamp.json"


def quaternion_wxyz_to_yaw(qw, qx, qy, qz):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        return 0.0

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def strip_instance_suffix(label):
    label = str(label).strip()
    return re.sub(r"[_\s-]+\d+$", "", label)


def round_number(value, decimals):
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0 else rounded


def convert_bbox_10_to_eval_bbox(bbox_3d, decimals):
    if not isinstance(bbox_3d, list):
        raise ValueError(f"bbox_3d must be a list, got {type(bbox_3d).__name__}")
    if len(bbox_3d) != 10:
        raise ValueError(f"expected bbox_3d length 10, got {len(bbox_3d)}")

    # Inference/merge_json.py uses scipy Rotation.as_quat(), whose serialized
    # order is qx, qy, qz, qw.
    x, y, z, w, h, length, qx, qy, qz, qw = [float(value) for value in bbox_3d]
    yaw = quaternion_wxyz_to_yaw(qw, qx, qy, qz)
    converted = [x, y, z, w, h, length, 0.0, 0.0, yaw]
    return [round_number(value, decimals) for value in converted]


def read_jsonl_records(path):
    records = []
    with path.open("r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def convert_scene_file(path, decimals, keep_instance_suffix):
    converted = []
    for idx, record in enumerate(read_jsonl_records(path)):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {idx} is not a dict")
        if "label" not in record:
            raise ValueError(f"{path}: record {idx} missing label")
        if "bbox_3d" not in record:
            raise ValueError(f"{path}: record {idx} missing bbox_3d")

        label = str(record["label"]).strip()
        if not keep_instance_suffix:
            label = strip_instance_suffix(label)

        converted.append(
            {
                "bbox_3d": convert_bbox_10_to_eval_bbox(record["bbox_3d"], decimals),
                "label": label,
            }
        )
    return converted


def build_bbox_lookup(input_dir, merged_name, decimals, keep_instance_suffix, require_done):
    lookup = {}
    skipped = []

    for scene_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        scene_name = scene_dir.name
        if require_done and not (scene_dir / "done.flag").exists():
            skipped.append((scene_name, "missing done.flag"))
            continue

        merged_path = scene_dir / merged_name
        if not merged_path.exists():
            skipped.append((scene_name, f"missing {merged_name}"))
            continue

        lookup[scene_name] = convert_scene_file(
            merged_path,
            decimals=decimals,
            keep_instance_suffix=keep_instance_suffix,
        )

    return lookup, skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-scene SpatialEncoder merged bbox JSONL files into the "
            "bbox JSON dict format expected by evaluation/eval.py."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Results root containing scene*/merged_bbox_*.json files.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Output path. The content is one JSON dict: {scene_name: [bboxes]}.",
    )
    parser.add_argument(
        "--merged_name",
        type=str,
        default=DEFAULT_MERGED_NAME,
        help=f"Per-scene merged bbox filename. Default: {DEFAULT_MERGED_NAME}",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for bbox numbers.",
    )
    parser.add_argument(
        "--keep_instance_suffix",
        action="store_true",
        help="Keep labels like chair_1 instead of converting them to chair.",
    )
    parser.add_argument(
        "--require_done",
        action="store_true",
        help="Only convert scene directories that contain done.flag.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {args.input_dir}")

    bbox_lookup, skipped = build_bbox_lookup(
        args.input_dir,
        merged_name=args.merged_name,
        decimals=args.decimals,
        keep_instance_suffix=args.keep_instance_suffix,
        require_done=args.require_done,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(bbox_lookup, f, indent=2, ensure_ascii=False)

    nonempty = sum(1 for bboxes in bbox_lookup.values() if bboxes)
    total_boxes = sum(len(bboxes) for bboxes in bbox_lookup.values())
    print(
        f"Wrote {args.output_json}: "
        f"scenes={len(bbox_lookup)}, nonempty={nonempty}, total_boxes={total_boxes}"
    )

    if skipped:
        print(f"Skipped scenes: {len(skipped)}")
        for scene_name, reason in skipped[:20]:
            print(f"- {scene_name}: {reason}")


if __name__ == "__main__":
    main()
