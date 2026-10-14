import os


def _env_path(name):
    value = os.environ.get(name)
    return os.path.abspath(os.path.expanduser(value)) if value else None


def _path_remaps_from_env():
    remaps = []
    raw = os.environ.get("SPATIALCODE_PATH_REMAPS", "")
    for item in raw.split(";"):
        if not item.strip() or "=" not in item:
            continue
        old, new = item.split("=", 1)
        old = os.path.abspath(os.path.expanduser(old.strip()))
        new = os.path.abspath(os.path.expanduser(new.strip()))
        if old and new:
            remaps.append((old, new))
    return remaps


def _default_path_remaps():
    box_data_path = _env_path("BOX_DATA_PATH")
    if not box_data_path:
        return []

    return [
        (
            "/scratch/dwirtz1/shared/hyperism/hyperism",
            os.path.join(box_data_path, "hyperism", "hyperism"),
        ),
        (
            "/scratch/dwirtz1/shared/hyperism",
            os.path.join(box_data_path, "hyperism"),
        ),
        (
            "/scratch/dwirtz1/shared/aria_digital_twin",
            os.path.join(box_data_path, "aria_digital_twin"),
        ),
        (
            "/scratch/dwirtz1/shared/vsibench-scannet/val-json",
            os.path.join(
                box_data_path,
                "vsi-scannet",
                "vsibench-scannet",
                "val-json",
                "val_frame_meta",
            ),
        ),
        (
            "/scratch/dwirtz1/shared/vsibench-scannet",
            os.path.join(box_data_path, "vsi-scannet", "vsibench-scannet"),
        ),
        (
            "/scratch/dwirtz1/shared/vsibench-scannet",
            os.path.join(box_data_path, "vsibench-scannet"),
        ),
        (
            "/scratch/dwirtz1/shared/arkitscenes",
            os.path.join(box_data_path, "ARKitScenes_data"),
        ),
    ]


def _apply_prefix_remap(path, old_prefix, new_prefix):
    path = os.path.abspath(os.path.expanduser(path))
    old_prefix = os.path.abspath(os.path.expanduser(old_prefix))
    new_prefix = os.path.abspath(os.path.expanduser(new_prefix))
    if path == old_prefix:
        return new_prefix
    if path.startswith(old_prefix + os.sep):
        return os.path.join(new_prefix, path[len(old_prefix) + 1 :])
    return None


def resolve_existing_path(path, extra_candidates=None):
    """Resolve metadata paths that may contain machine-specific absolute roots."""
    candidates = []
    if extra_candidates:
        candidates.extend(extra_candidates)
    candidates.append(path)

    normalized = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if candidate not in normalized:
            normalized.append(candidate)

    for candidate in normalized:
        if os.path.exists(candidate):
            return candidate

    remaps = _path_remaps_from_env() + _default_path_remaps()
    for candidate in normalized:
        for old_prefix, new_prefix in remaps:
            mapped = _apply_prefix_remap(candidate, old_prefix, new_prefix)
            if mapped and os.path.exists(mapped):
                return mapped

    return normalized[0] if normalized else path
