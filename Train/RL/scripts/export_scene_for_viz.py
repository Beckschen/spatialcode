#!/usr/bin/env python3
"""
Export one or more scenes from spatialcode test/val data.json for the 3D viewer.

Usage:
  python scripts/export_scene_for_viz.py data/spatial_dataset_all/test/data.json
  python scripts/export_scene_for_viz.py data/spatial_dataset_all/test/data.json --index 0 --out viz/scene_0.json
  python scripts/export_scene_for_viz.py data/spatial_dataset_all/test/data.json --scene-name 41159525 --out viz/scene_41159525.json
  python scripts/export_scene_for_viz.py data/spatial_dataset_all/test/data.json --max 5 --out-dir viz/scenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export scenes from data.json for 3D visualization")
    parser.add_argument("data_json", type=str, help="Path to data.json (test or val)")
    parser.add_argument("--index", type=int, default=None, help="Export only this item index (0-based)")
    parser.add_argument("--scene-name", type=str, default=None, help="Export only item with this scene_name")
    parser.add_argument("--max", type=int, default=None, help="When exporting multiple, max number of scenes")
    parser.add_argument("--out", type=str, default=None, help="Output JSON file (single scene)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for multiple scene JSONs")
    args = parser.parse_args()

    path = Path(args.data_json)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        items = json.load(f)

    if not items:
        print("Error: no items in data.json", file=sys.stderr)
        sys.exit(1)

    # Single scene by index or scene_name
    if args.index is not None:
        if args.index < 0 or args.index >= len(items):
            print(f"Error: index must be in [0, {len(items)-1}]", file=sys.stderr)
            sys.exit(1)
        selected = [items[args.index]]
    elif args.scene_name is not None:
        selected = [it for it in items if it.get("scene_name") == args.scene_name]
        if not selected:
            print(f"Error: no item with scene_name={args.scene_name}", file=sys.stderr)
            sys.exit(1)
        selected = selected[:1]
    else:
        selected = items
        if args.max is not None:
            selected = selected[: args.max]

    out_path = args.out
    out_dir = Path(args.out_dir) if args.out_dir else None

    if len(selected) == 1 and out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene_name": selected[0].get("scene_name", "scene_0"),
            "questions": selected[0].get("questions"),
            "qa_type": selected[0].get("qa_type"),
            "solution": selected[0].get("solution"),
            "gt_3d_bboxes": selected[0].get("gt_3d_bboxes", []),
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_file} ({len(payload['gt_3d_bboxes'])} bboxes)")
        return

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, it in enumerate(selected):
            name = it.get("scene_name", f"scene_{i}")
            payload = {
                "scene_name": name,
                "questions": it.get("questions"),
                "qa_type": it.get("qa_type"),
                "solution": it.get("solution"),
                "gt_3d_bboxes": it.get("gt_3d_bboxes", []),
            }
            fpath = out_dir / f"{name}.json"
            with open(fpath, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote {fpath} ({len(payload['gt_3d_bboxes'])} bboxes)")
        # Also write an index for the viewer to list scenes
        index_path = out_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump([it.get("scene_name", f"scene_{i}") for i, it in enumerate(selected)], f, indent=2)
        print(f"Wrote {index_path}")
        return

    # Default: print first scene as JSON to stdout (for embedding in HTML)
    payload = {
        "scene_name": selected[0].get("scene_name", "scene_0"),
        "questions": selected[0].get("questions"),
        "qa_type": selected[0].get("qa_type"),
        "solution": selected[0].get("solution"),
        "gt_3d_bboxes": selected[0].get("gt_3d_bboxes", []),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
