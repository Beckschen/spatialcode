import math
import torch
import logging
from math import ceil, floor
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
import numpy as np


# ===========
# adapted from DetAny3D
# ===========
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def giou_diag(xyxy1, xyxy2):  # both [N,4]
    # intersection
    x1 = torch.max(xyxy1[:, 0], xyxy2[:, 0])
    y1 = torch.max(xyxy1[:, 1], xyxy2[:, 1])
    x2 = torch.min(xyxy1[:, 2], xyxy2[:, 2])
    y2 = torch.min(xyxy1[:, 3], xyxy2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # areas
    a1 = (xyxy1[:, 2] - xyxy1[:, 0]).clamp(min=0) * (xyxy1[:, 3] - xyxy1[:, 1]).clamp(
        min=0
    )
    a2 = (xyxy2[:, 2] - xyxy2[:, 0]).clamp(min=0) * (xyxy2[:, 3] - xyxy2[:, 1]).clamp(
        min=0
    )
    union = (a1 + a2 - inter).clamp(min=1e-12)

    iou = inter / union

    # enclosure for GIoU
    ex1 = torch.min(xyxy1[:, 0], xyxy2[:, 0])
    ey1 = torch.min(xyxy1[:, 1], xyxy2[:, 1])
    ex2 = torch.max(xyxy1[:, 2], xyxy2[:, 2])
    ey2 = torch.max(xyxy1[:, 3], xyxy2[:, 3])
    closure = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + 1e-12

    giou = iou - (closure - union) / closure
    return giou.clamp(min=-1.0, max=1.0)  # [N]


def project_to_image(points_3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    points_3d: [N,3] in camera coords
    K: [3,3] intrinsic matrix
    return: [N,2] pixel coordinates
    """
    device = points_3d.device
    ones = torch.ones((points_3d.shape[0], 1), device=device)
    points_3d_homo = torch.cat((points_3d, ones), dim=1).float()  # [N,4]

    # drop the last col in homogeneous multiplication since K is 3x3
    K = K.to(dtype=points_3d_homo.dtype, device=points_3d_homo.device)
    points_2d_homo = (K @ points_3d_homo[:, :3].T).T  # [N,3]

    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    return points_2d


def class2angle(cls, residual):
    """Inverse function to angle2class."""
    angle_per_class = 2 * torch.pi / float(12)
    angle_center = cls * angle_per_class + angle_per_class / 2
    angle = angle_center + residual
    angle = torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi
    return angle


def angle2class(angle):
    """Convert continuous angle to discrete class and residual."""
    angle = angle % (2 * torch.pi)
    angle_per_class = 2 * torch.pi / float(12)
    shifted_angle = (angle + angle_per_class / 2) % (2 * torch.pi)
    class_id = torch.floor(shifted_angle / angle_per_class)
    residual = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id.to(torch.long), residual


def iou_3d_aabb(boxes1, boxes2):
    """
    boxes1: (N, 6) = [x_min, y_min, z_min, x_max, y_max, z_max]
    boxes2: (M, 6) = same format
    return: (N, M) pairwise IoU
    """
    # 扩展维度以便广播
    b1 = boxes1[:, None, :]  # (N,1,6)
    b2 = boxes2[None, :, :]  # (1,M,6)

    # 交集坐标
    max_min = torch.max(b1[..., :3], b2[..., :3])  # (N,M,3)
    min_max = torch.min(b1[..., 3:], b2[..., 3:])  # (N,M,3)

    inter = (min_max - max_min).clamp(min=0)  # (N,M,3)
    inter_vol = inter.prod(dim=-1)  # (N,M)

    # 各自体积
    vol1 = (b1[..., 3:] - b1[..., :3]).prod(dim=-1)  # (N,1)
    vol2 = (b2[..., 3:] - b2[..., :3]).prod(dim=-1)  # (1,M)

    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / union_vol.clamp(min=1e-6)


def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    poly1 = Polygon(p1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(p2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

        (3) +---------+. (2)
            | ` .     |  ` .
            | (7) +---+-----+ (6)
            |     |   |     |
        (4) +-----+---+. (1)|
            ` .   |     ` . |
            (8) ` +---------+ (5)
    """
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


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R -> quaternion  (Shoemake, batch-safe)
    Returns [w, x, y, z]
    """
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    R00 = R[:, 0, 0]
    R11 = R[:, 1, 1]
    R22 = R[:, 2, 2]
    trace = R00 + R11 + R22

    eps = 1e-8

    # 分支0：trace > 0
    cond0 = trace > 0
    S0 = torch.sqrt(torch.clamp(trace[cond0] + 1.0, min=eps)) * 2
    q[cond0, 0] = 0.25 * S0
    q[cond0, 1] = (R[cond0, 2, 1] - R[cond0, 1, 2]) / S0
    q[cond0, 2] = (R[cond0, 0, 2] - R[cond0, 2, 0]) / S0
    q[cond0, 3] = (R[cond0, 1, 0] - R[cond0, 0, 1]) / S0

    # 分支1：R00 最大
    cond1 = (~cond0) & (R00 > R11) & (R00 > R22)
    S1 = (
        torch.sqrt(torch.clamp(1.0 + R00[cond1] - R11[cond1] - R22[cond1], min=eps)) * 2
    )
    q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / S1
    q[cond1, 1] = 0.25 * S1
    q[cond1, 2] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / S1
    q[cond1, 3] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / S1

    # 分支2：R11 最大
    cond2 = (~cond0) & (~cond1) & (R11 > R22)
    S2 = (
        torch.sqrt(torch.clamp(1.0 + R11[cond2] - R00[cond2] - R22[cond2], min=eps)) * 2
    )
    q[cond2, 0] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / S2
    q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / S2
    q[cond2, 2] = 0.25 * S2
    q[cond2, 3] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / S2

    # 分支3：剩下的一切 = R22 最大
    cond3 = (~cond0) & (~cond1) & (~cond2)
    S3 = (
        torch.sqrt(torch.clamp(1.0 + R22[cond3] - R00[cond3] - R11[cond3], min=eps)) * 2
    )
    q[cond3, 0] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / S3
    q[cond3, 1] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / S3
    q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / S3
    q[cond3, 3] = 0.25 * S3

    q = q / q.norm(dim=-1, keepdim=True)
    return q


def quat_to_rotmat(q: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Quaternion -> Rotation Matrix (Shoemake, batch-safe)
    Input: q of shape (B, 4) in [w, x, y, z] format
    Output: R of shape (B, 3, 3)
    """
    if isinstance(q, np.ndarray):
        is_numpy = True
        q = torch.tensor(q, dtype=torch.float32)
    else:
        is_numpy = False

    # ensure normalized quaternion
    q = q / q.norm(dim=-1, keepdim=True)

    w, x, y, z = q.unbind(dim=-1)

    B = q.shape[0]
    R = torch.zeros((B, 3, 3), dtype=q.dtype, device=q.device)

    # Precompute repeated terms
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z
    yz = y * z

    # Fill rotation matrix
    # Following standard quaternion → rotation matrix formula
    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)

    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)

    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    if is_numpy:
        return R.numpy()
    return R


def compute_3d_bbox_vertices(
    x, y, z, w, h, l, yaw=0.0, rotation_matrix=None, device=None
):
    """
    Compute the 8 corners of a 3D bounding box given its center, size, and rotation.

    Args:
        x, y, z: float or torch.Tensor, center of the box
        w, h, l: float or torch.Tensor, width (X), height (Y), length (Z)
        yaw: float or torch.Tensor, rotation around Y-axis (radians)
        rotation_matrix: [3,3] torch.Tensor, external rotation
        device: torch.device, optional

    Returns:
        corners_translated: [8,3] torch.Tensor, coordinates of bbox corners
        fore_plane_center: [1,3] torch.Tensor, coordinates of box front face center
    """
    if device is None:
        device = torch.device("cpu")

    # 8 corners relative to center
    corners = torch.tensor(
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
        dtype=torch.float32,
        device=device,
    )

    # Rotation matrix
    if rotation_matrix is not None and (isinstance(yaw, float) and yaw == 0):
        R = rotation_matrix
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32, device=device)
    elif rotation_matrix is None and (not isinstance(yaw, float) or yaw != 0):
        R = torch.tensor(
            [
                [torch.cos(yaw), 0, torch.sin(yaw)],
                [0, 1, 0],
                [-torch.sin(yaw), 0, torch.cos(yaw)],
            ],
            dtype=torch.float32,
            device=device,
        )
    elif rotation_matrix is not None and (not isinstance(yaw, float) or yaw != 0):
        R_yaw = torch.tensor(
            [
                [torch.cos(yaw), 0, torch.sin(yaw)],
                [0, 1, 0],
                [-torch.sin(yaw), 0, torch.cos(yaw)],
            ],
            dtype=torch.float32,
            device=device,
        )
        R = rotation_matrix @ R_yaw
    else:
        R = torch.eye(3, dtype=torch.float32, device=device)

    # Rotate
    corners_rotated = corners @ R.T
    # Translate
    translation = torch.tensor([x, y, z], dtype=torch.float32, device=device)
    corners_translated = corners_rotated + translation

    return corners_translated


def decode_bbox3d(bboxes_3d, bboxes_quat):
    """
    Decode 3D bounding boxes from center, size, and rotation quaternion to 8 corner points.

    Args:
        bboxes_3d: [N, 7] tensor, each row is (x, y, z, w, h, l, yaw)
        bboxes_quat: [N, 4] tensor, each row is (qw, qx, qy, qz)

    Returns:
        corners: [N, 8, 3] tensor, 8 corners of the 3D bounding boxes
    """
    corners = []
    for i in range(bboxes_3d.size(0)):
        x, y, z, w, h, l = bboxes_3d[i][:6]
        qw, qx, qy, qz = bboxes_quat[i]
        rot_mat = R.from_quat([qx.item(), qy.item(), qz.item(), qw.item()]).as_matrix()
        corners.append(
            compute_3d_bbox_vertices(
                x, y, z, w, h, l, rotation_matrix=rot_mat, device=bboxes_3d.device
            )
        )
    return torch.stack(corners)


# ===============
# code from VGGT
# ===============
def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(
            f"Tensor {loss_name} contains inf or nan values. Replacing with zeros."
        )
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor,
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


########
# adapted from 3D-MOOD
########
DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """
    x = x.to(dtype=torch.float32)
    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x, device=x.device, dtype=torch.float32)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    _, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def so3_relative_angle(
    R1: torch.Tensor | np.ndarray,
    R2: torch.Tensor | np.ndarray,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    if isinstance(R1, np.ndarray):
        R1 = torch.tensor(R1, dtype=torch.float32)
    if isinstance(R2, np.ndarray):
        R2 = torch.tensor(R2, dtype=torch.float32)
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)
