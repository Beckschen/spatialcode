from glob import glob
import json
import os
from scipy.spatial.transform import Rotation as R

def load_txt(file_path):
    all_bboxes = []
    with open(file_path, "r") as f:
        boxes = [line for line in f.read().splitlines() if line.strip()]
    for box in boxes:
        bbox = box.split()
        if len(bbox) < 12:
            print(f"Invalid bbox length: {len(bbox)}")
            continue
        elif len(bbox) > 12:
            # cat is the first two elements
            cat = bbox[0:2]
            cat = " ".join(cat)
            bbox = bbox[2:]
        else:
            cat = bbox[0]
            bbox = bbox[1:]
        if cat=="tv monitor":
            cat = "tv"
        
        x, y, z, w, h, length, qw, qx, qy, qz, _ = bbox
        # axis convention: x forward, y left, z up
        qw_f, qx_f, qy_f, qz_f = float(qw), float(qx), float(qy), float(qz)
        rotation = R.from_quat([qx_f, qy_f, qz_f, qw_f])
        rotation = rotation.as_euler("xyz", degrees=False)
        roll, pitch, yaw = rotation
        all_bboxes.append(
            {
                "bbox_3d": [
                    round(float(x), 2),
                    round(float(y), 2),
                    round(float(z), 2),
                    round(float(w), 2),
                    round(float(h), 2),
                    round(float(length), 2),
                    0,  # roll
                    0,  # pitch
                    round(float(yaw), 2),
                ],
                "label": cat,
            }
        )
    return all_bboxes


def load_all_txt(dataset="scannetpp", bbox_path=None):
    prediction_files = glob(os.path.join(bbox_path, f"{dataset}/*.txt"))

    predictions = {}
    for prediction_file in prediction_files:
        scene_id = prediction_file.split("/")[-1].replace(".txt", "")
        bboxes = load_txt(prediction_file)
        for bbox in bboxes:
            predictions.setdefault(scene_id, [])
            predictions[scene_id].append(bbox)
    return predictions


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def load_jsonl(input_path):
    data = []
    with open(input_path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {i} in {input_path}: {e}")
    return data

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def add_to_jsonl(item, output_path):
    with open(output_path, "a") as f:
        f.write(json.dumps(item) + "\n")

def save_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")