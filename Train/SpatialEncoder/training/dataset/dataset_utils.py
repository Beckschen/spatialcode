import os, math, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import time
import torch
from PIL import Image
from typing import Callable
from shapely.geometry import Polygon


def _safe_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    return img


def project_to_so3(R):
    # R: (...,3,3)
    U, S, Vh = np.linalg.svd(R)

    # initial projection
    R_proj = U @ Vh

    # check determinant (last axis: i.e., R_proj[..., :, :])
    det = np.linalg.det(R_proj)

    # mask: where det < 0
    neg_mask = det < 0

    if np.any(neg_mask):
        # flip last column of U for those matrices
        U[neg_mask, :, -1] *= -1
        R_proj = U @ Vh

    return R_proj


def _rect_iou_xyxy(a, b):
    # a, b: [..., 4] with [x1,y1,x2,y2]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, (a[..., 2] - a[..., 0])) * np.maximum(
        0.0, (a[..., 3] - a[..., 1])
    )
    area_b = np.maximum(0.0, (b[..., 2] - b[..., 0])) * np.maximum(
        0.0, (b[..., 3] - b[..., 1])
    )
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _project_points(K, Xc):
    """
    K: 3x3, Xc: Nx3 (camera coords), returns Nx2, valid_z mask
    """
    z = Xc[:, 2]
    valid = z > 1e-6
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    x[valid] = Xc[valid, 0] / z[valid]
    y[valid] = Xc[valid, 1] / z[valid]
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=-1), valid


def _world_to_cam(E34, Pw):
    """E34: 3x4 world->cam, Pw: Nx3 world points -> Nx3 camera points"""
    R = E34[:, :3]
    t = E34[:, 3]
    return (Pw @ R.T) + t


def _box3d_corners(center, dims, rotmat):
    """
    center: (3,) camera or world
    dims: (3,) (dx, dy, dz) along (x,y,z) axes
    rotmat: (3,3) rotation matrix
    returns (8,3) corners in the same coord as center
    """
    dx, dy, dz = np.abs(dims) * 0.5
    # local corners (x,y,z)
    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=np.float32,
    )
    # rotate
    corners = corners @ rotmat.T

    return corners + center[None, :]


def _draw_axes(img, K, z_ref=1.5, axis_len=0.3):
    """
    在相机坐标系原点处画 XYZ 轴，投影到图像 (X=红, Y=绿, Z=蓝)。
    """
    origin = np.array([[0, 0, z_ref]], dtype=np.float32)
    X = np.array([[axis_len, 0, z_ref]], dtype=np.float32)
    Y = np.array([[0, axis_len, z_ref]], dtype=np.float32)
    Z = np.array([[0, 0, z_ref + axis_len]], dtype=np.float32)
    P, v0 = _project_points(K, np.vstack([origin, X, Y, Z]))
    if v0.all():
        o, px, py, pz = P.astype(int)
        cv2.line(img, o, px, (0, 0, 255), 2)  # X red
        cv2.line(img, o, py, (0, 255, 0), 2)  # Y green
        cv2.line(img, o, pz, (255, 0, 0), 2)  # Z blue


def _polyline(img, pts, color, closed=True, thickness=2):
    pts = np.round(pts).astype(int)
    cv2.polylines(img, [pts], closed, color, thickness, lineType=cv2.LINE_AA)


def _make_palette(num_objs, seed=42):
    np.random.seed(seed)
    colors = []
    for i in range(num_objs):
        # 每个物体生成一个 (B,G,R)，亮度范围限制在 [64,255]
        colors.append(tuple(np.random.randint(64, 256, size=3).tolist()))
    return colors


def project_to_image(points_3d, K):
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    if K.shape[0] == 3:
        K_extended = np.hstack((K, np.zeros((3, 1))))
    else:
        K_extended = K
    points_2d = K_extended @ points_3d_homo.T

    new_points_2d = points_2d[:2, :] / points_2d[2:3, :]
    new_points_2d[0, :] = (
        np.sign(points_2d[0, :]) * np.abs(points_2d[0, :]) / np.abs(points_2d[2, :])
    )
    new_points_2d[1, :] = (
        np.sign(points_2d[1, :]) * np.abs(points_2d[1, :]) / np.abs(points_2d[2, :])
    )
    return new_points_2d.T


def remap_axes(R, extents, perm=(0, 1, 2), signs=(1, 1, 1)):
    """
    R: (3,3) 旋转矩阵（列是局部轴）
    extents: (3,) 全长 (w,h,l) 或 (x_len,y_len,z_len)；若你保存的是半长，使用同理
    perm: 轴置换，如 (0,2,1) 表示新局部 y ← 旧的 z
    signs: 轴翻号，如 (1, 1, -1) 表示新局部 z 取旧轴并取反

    返回: R_new, extents_new
    """
    P = np.zeros((3, 3))
    for j, (p, s) in enumerate(zip(perm, signs)):
        P[p, j] = s  # 列 j 选用旧轴 p，并可能翻号

    # 新旋转 & 新尺寸（确保 extents 非负；符号体现在 R_new 的列向量）
    R_new = R @ P
    extents_new = np.abs(P.T @ extents)  # 通常用非负长度

    # 若你想强制 R_new 正交（数值稳健性），可做一次极分解修正：
    # U, _, Vt = np.linalg.svd(R_new)
    # R_new = U @ Vt
    # if np.linalg.det(R_new) < 0:
    #     U[:, -1] *= -1
    #     R_new = U @ Vt

    return R_new, extents_new


def compute_3d_bbox_vertices(x, y, z, w, h, l, yaw=0, rotation_matrix=None):
    """
    output_corners_order:
        (3) +---------+. (2)
            | ` .     |  ` .
            | (7) +---+-----+ (6)
            |     |   |     |
        (4) +-----+---+. (1)|
            ` .   |     ` . |
            (8) ` +---------+ (5)

    """
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
        ]
    )

    # 2. 【新增逻辑】定义局部坐标系下的三个正方向向量
    # 这里长度取维度的一半，这样画出来的轴正好连接到面的中心，方便检查尺寸对不对
    local_axes = np.array(
        [
            [w / 2, 0, 0],  # Local X axis
            [0, h / 2, 0],  # Local Y axis
            [0, 0, l / 2],  # Local Z axis
        ]
    )
    # 如果你只需要单位向量（长度为1），可以将上面改为:
    # local_axes = np.array([[1,0,0], [0,1,0], [0,0,1]])

    # 3. 计算旋转矩阵 R
    if rotation_matrix is not None and yaw == 0:
        R = rotation_matrix
    elif rotation_matrix is None and yaw != 0:
        # 注意：这里的旋转矩阵定义决定了 yaw 的方向定义
        R = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        )
    elif rotation_matrix is not None and yaw != 0:
        R = rotation_matrix @ np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        )
    else:
        R = np.eye(3)

    # 4. 应用旋转和平移
    # 计算角点
    corners_rotated = corners @ R.T
    corners_translated = corners_rotated + np.array([x, y, z])

    # 先旋转
    axes_rotated = local_axes @ R.T
    # 再平移
    axes_endpoints = axes_rotated + np.array([x, y, z])

    return corners_translated, axes_endpoints


def compute_xyzlwhR_from_3d_bbox(corners_3d):
    """
    Recovers center (x, y, z), dimensions (l, w, h), and rotation matrix (R)
    from 8 3D bounding box corners. The recovered R is the Object-to-World matrix.

    Args:
        corners_3d (np.array): 8x3 array of global vertex coordinates.

    Returns:
        x, y, z (float): Recovered center coordinates.
        l, w, h (float): Recovered Length, Width, Height dimensions.
        rotmat (np.array): Recovered 3x3 rotation matrix (R).
    """

    # 1. Recover Center (x, y, z): The center is the mean of all 8 corners.
    x, y, z = np.mean(corners_3d, axis=0)

    # 2. Recover Dimensions (w, h, l): Calculated from the magnitude of the
    # vector differences between specific corners (0-7 indexing).

    # w (Width, local x'): Distance between corners 0 and 3
    w_recovered = np.linalg.norm(corners_3d[0] - corners_3d[3])

    # h (Height, local y'): Distance between corners 0 and 1
    h_recovered = np.linalg.norm(corners_3d[0] - corners_3d[1])

    # l (Length, local z'): Distance between corners 0 and 4
    l_recovered = np.linalg.norm(corners_3d[0] - corners_3d[4])

    # 3. Recover Rotation Matrix (R): R's rows are the normalized basis
    # vectors (x', y', z') of the object frame in global coordinates.

    # x' direction vector (width vector)
    x_prime_vec = corners_3d[0] - corners_3d[3]
    x_prime_unit = x_prime_vec / w_recovered

    # y' direction vector (height vector)
    y_prime_vec = corners_3d[0] - corners_3d[1]
    y_prime_unit = y_prime_vec / h_recovered

    # z' direction vector (length vector)
    z_prime_vec = corners_3d[0] - corners_3d[4]
    z_prime_unit = z_prime_vec / l_recovered

    # R is composed of these unit vectors as its rows.
    rotmat_recovered = np.vstack([x_prime_unit, y_prime_unit, z_prime_unit])

    # The order of returned dimensions matches the expected l, w, h signature.
    return x, y, z, l_recovered, w_recovered, h_recovered, rotmat_recovered


def draw_bbox_2d(image, points_2d, axes_2d=None, color=(0, 255, 0), thickness=2):
    """
    Args:
        image: cv2 image
        points_2d: (8, 2) 8个角点的2D投影坐标
        axes_2d: (3, 2) 三个轴端点的2D投影坐标 (对应上一段代码的 axes_endpoints 投影后的结果)
                 顺序必须是 [X轴端点, Y轴端点, Z轴端点]
        color: 包围盒边框颜色
        thickness: 线宽
    """
    # 确保输入是 int 类型
    points_2d = points_2d.astype(int)

    # --- 1: 计算整个 Bounding Box 的中心点 ---
    # 我们取所有8个点的平均值作为几何中心
    # 注意：在透视投影下，3D中心投影点 != 2D投影点的平均值，但作为可视化近似通常足够好
    box_center = np.mean(points_2d, axis=0).astype(int)
    box_center = tuple(box_center)

    # 将点转换为 tuple 列表供 cv2 使用
    points_2d = [tuple(point) for point in points_2d]

    # --- 绘制 Wireframe (线框) ---
    # 绘制底面 (0-1-2-3)
    for i in range(4):
        cv2.line(image, points_2d[i], points_2d[(i + 1) % 4], color, thickness)

    # 绘制顶面 (4-5-6-7)
    for i in range(4, 8):
        cv2.line(image, points_2d[i], points_2d[(i + 1) % 4 + 4], color, thickness)

    # 绘制连接柱 (0-4, 1-5, 2-6, 3-7)
    for i in range(4):
        cv2.line(image, points_2d[i], points_2d[i + 4], color, thickness)

    # --- 新增逻辑 2: 可视化 XYZ 三个坐标轴 ---
    if axes_2d is not None:
        # 定义颜色 (OpenCV使用 BGR 格式)
        # X轴: 红色, Y轴: 绿色, Z轴: 蓝色
        axis_colors = [
            (0, 0, 255),  # Red for X
            (0, 255, 0),  # Green for Y
            (255, 0, 0),  # Blue for Z
        ]

        # 遍历绘制三个轴
        for i in range(len(axes_2d)):
            # 获取当前轴的端点
            tip = axes_2d[i]

            # 确保端点是整数 tuple
            if not isinstance(tip, tuple):
                tip = tuple(map(int, tip))

            # 绘制连接线：从几何中心 -> 轴端点
            cv2.line(image, box_center, tip, axis_colors[i], thickness)

            # (可选) 在轴的末端画一个小圆点，增加立体感
            cv2.circle(image, tip, thickness + 2, axis_colors[i], -1)

            # (可选) 如果你想标上字母 x,y,z
            # axis_labels = ['x', 'y', 'z']
            # cv2.putText(image, axis_labels[i], tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_colors[i], 1)

    return image


def draw_text(
    im,
    text,
    pos,
    scale=0.4,
    color="auto",
    font=cv2.FONT_HERSHEY_SIMPLEX,
    bg_color=(0, 255, 255),
    blend=0.33,
    lineType=1,
):
    text = str(text)
    pos = [int(pos[0]), int(pos[1])]

    if color == "auto":
        if bg_color is not None:
            color = (
                (0, 0, 0)
                if ((bg_color[0] + bg_color[1] + bg_color[2]) / 3) > 127.5
                else (255, 255, 255)
            )
        else:
            color = (0, 0, 0)

    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s : y_e + 1, x_s : x_e + 1, 0] = im[
            y_s : y_e + 1, x_s : x_e + 1, 0
        ] * blend + bg_color[0] * (1 - blend)
        im[y_s : y_e + 1, x_s : x_e + 1, 1] = im[
            y_s : y_e + 1, x_s : x_e + 1, 1
        ] * blend + bg_color[1] * (1 - blend)
        im[y_s : y_e + 1, x_s : x_e + 1, 2] = im[
            y_s : y_e + 1, x_s : x_e + 1, 2
        ] * blend + bg_color[2] * (1 - blend)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)


def _check_data(
    video_frames,
    depth_frames,
    bboxes2d,
    bboxes3d,
    bboxsR3d,
    extrinsic,
    intrinsic,
    save_dir="temp/dataset_check",
    not_appear_thresh=-100,
):
    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap("rainbow")

    # convert each input to numpy
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    if isinstance(depth_frames, torch.Tensor):
        depth_frames = depth_frames.cpu().numpy()
    if isinstance(bboxes2d, torch.Tensor):
        bboxes2d = bboxes2d.cpu().numpy()
    if isinstance(bboxes3d, torch.Tensor):
        bboxes3d = bboxes3d.cpu().numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()
    if isinstance(bboxsR3d, torch.Tensor):
        bboxsR3d = bboxsR3d.cpu().numpy()

    N, H, W, _ = video_frames.shape
    _, M, _ = bboxes2d.shape
    assert bboxes3d.shape == (N, M, 6), "bboxes3d should be (N,M,6)"
    assert extrinsic.shape[:2] == (N, 3) and extrinsic.shape[2] == 4
    assert intrinsic.shape == (N, 3, 3)

    # 为每个物体分配固定颜色
    # palette = _make_palette(M)

    for n in range(N):
        depth_frames[n] = (
            (depth_frames[n] - depth_frames[n].min())
            / (depth_frames[n].max() - depth_frames[n].min())
            * 255
        )
        img = _safe_uint8(video_frames[n].copy())
        depth_img = cv2.applyColorMap(_safe_uint8(depth_frames[n]), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.6, depth_img, 0.4, 0)
        K = intrinsic[n].astype(np.float32)

        # 可视化主点
        # cx, cy = float(K[0, 2]), float(K[1, 2])
        # cv2.drawMarker(img, (int(round(cx)), int(round(cy))), (0, 255, 255),
        #                markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        # _draw_axes(img, K, z_ref=1.5, axis_len=0.4)

        # # 一些相机内参 sanity checks
        # cx_in = 0 <= cx < W
        # cy_in = 0 <= cy < H

        # 逐对象处理
        for m in range(M):
            color = cmap(m / M)  # 固定颜色
            color = (np.array(color) * 255).astype(np.uint8)
            color = tuple(map(int, color[:3]))
            bb2 = bboxes2d[n, m].astype(np.float32)
            if np.any(bb2[:2] < not_appear_thresh):
                continue  # 不出现

            # 2D 框可视化
            print(f"[Info] frame {n} obj {m} bbox2d={bb2}")
            p1 = (int(bb2[0]), int(bb2[1]))
            p2 = (int(bb2[2]), int(bb2[3]))
            cv2.rectangle(img, p1, p2, color, 2)
            cen2d = np.array(
                [(bb2[0] + bb2[2]) * 0.5, (bb2[1] + bb2[3]) * 0.5], dtype=np.float32
            )

            c3d = bboxes3d[n, m, :3].astype(np.float32)
            d3d = bboxes3d[n, m, 3:6].astype(np.float32)
            rotmat = bboxsR3d[n, m].astype(np.float32)

            # 投影中心点
            p_center, v_c = _project_points(K, c3d[None, :])
            if v_c[0]:
                cv2.circle(
                    img, (int(p_center[0, 0]), int(p_center[0, 1])), 3, color, -1
                )
            else:
                print(f"[Warning] frame {n} obj {m} center behind cam? c3d={c3d}")

            # 3D 八点 -> 投影多边形
            corners_cam = _box3d_corners(c3d, d3d, rotmat)
            proj, valid = _project_points(K, corners_cam.astype(np.float32))
            # valid_corners = int(valid.sum())
            x, y, z = c3d
            w, h, l = d3d
            corners_cam, axes_endpoints_cam = compute_3d_bbox_vertices(
                x, y, z, w, h, l, rotation_matrix=rotmat
            )
            axes_endpoints_img = project_to_image(axes_endpoints_cam, K)
            img = draw_bbox_2d(img, proj, axes_2d=axes_endpoints_img, color=color, thickness=2)
            # if valid_corners >= 4:
            #     idx_front = [0, 1, 2, 3, 0]
            #     idx_back = [4, 5, 6, 7, 4]
            #     for a, b in zip(idx_front[:-1], idx_front[1:]):
            #         cv2.line(
            #             img,
            #             tuple(proj[a].astype(int)),
            #             tuple(proj[b].astype(int)),
            #             color,
            #             2,
            #         )
            #     for a, b in zip(idx_back[:-1], idx_back[1:]):
            #         cv2.line(
            #             img,
            #             tuple(proj[a].astype(int)),
            #             tuple(proj[b].astype(int)),
            #             color,
            #             2,
            #         )
            #     for a, b in zip([0, 1, 2, 3], [4, 5, 6, 7]):
            #         cv2.line(
            #             img,
            #             tuple(proj[a].astype(int)),
            #             tuple(proj[b].astype(int)),
            #             color,
            #             1,
            #         )
            # else:
            #     print(
            #         f"[Warning] frame {n} obj {m} too few valid corners? valid={valid_corners}"
            #     )
        out_path = os.path.join(save_dir, f"frame_{n:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def make_bbox_edges(corners):
    """
    corners: (8,3)
    return edges: (12,2,3)
    """
    idx_pairs = [
        # front face (z = +dz)
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # back face (z = -dz)
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # side edges (连接前后)
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in idx_pairs], dtype=np.float32)


#####################
# copied from map anything
#####################


def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    RESOLUTION_MAPPINGS = {
        518: {
            1.000: (518, 518),  # 1:1
            1.321: (518, 392),  # 4:3
            1.542: (518, 336),  # 3:2
            1.762: (518, 294),  # 16:9
            2.056: (518, 252),  # 2:1
            3.083: (518, 168),  # 3.2:1
            0.757: (392, 518),  # 3:4
            0.649: (336, 518),  # 2:3
            0.567: (294, 518),  # 9:16
            0.486: (252, 518),  # 1:2
        },
        512: {
            1.000: (512, 512),  # 1:1
            1.333: (512, 384),  # 4:3
            1.524: (512, 336),  # 3:2
            1.778: (512, 288),  # 16:9
            2.000: (512, 256),  # 2:1
            3.200: (512, 160),  # 3.2:1
            0.750: (384, 512),  # 3:4
            0.656: (336, 512),  # 2:3
            0.562: (288, 512),  # 9:16
            0.500: (256, 512),  # 1:2
        },
    }
    ASPECT_RATIO_KEYS = {
        518: sorted(RESOLUTION_MAPPINGS[518].keys()),
        512: sorted(RESOLUTION_MAPPINGS[512].keys()),
    }

    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]


def rotate_image(img: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return img
    elif orientation == 1:  # left → 顺时针90°
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 2:  # upside-down → 180°
        return cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 3:  # right → 逆时针90°
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Invalid orientation id {orientation}")

def rotate_intrinsic(K: np.ndarray, orientation: int, width: int, height: int) -> np.ndarray:
    K_rotated = K.copy()
    if orientation == 0:
        return K_rotated
    elif orientation == 1:  # left → 顺时针90°
        K_rotated[0, 0], K_rotated[1, 1] = K[1, 1], K[0, 0]
        K_rotated[0, 2], K_rotated[1, 2] = height - K[1, 2], K[0, 2]
        return K_rotated
    elif orientation == 2:  # upside-down → 180°
        K_rotated[0, 2], K_rotated[1, 2] = width - K[0, 2], height - K[1, 2]
        return K_rotated
    elif orientation == 3:  # right → 逆时针90°
        K_rotated[0, 0], K_rotated[1, 1] = K[1, 1], K[0, 0]
        K_rotated[0, 2], K_rotated[1, 2] = K[1, 2], width - K[0, 2]
        return K_rotated
    else:
        raise ValueError(f"Invalid orientation id {orientation}")

# ==========================================================
# the code of loading mapanything images
def load_images(
    folder_or_list,
    resize_mode="fixed_mapping",
    size=None,
    norm_type="dinov2",
    patch_size=14,
    verbose=False,
    bayer_format=False,
    resolution_set=518,
    stride=1,
    orientation=None,
):
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        heif_support_enabled = True
    except ImportError:
        heif_support_enabled = False
    from PIL.ImageOps import exif_transpose
    import PIL
    from mapanything.utils.cropping import crop_resize_if_necessary
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
    import torchvision.transforms as tvf

    """
    Open and convert all images in a list or folder to proper input format for model

    Args:
        folder_or_list (str or list): Path to folder or list of image paths.
        resize_mode (str): Resize mode - "fixed_mapping", "longest_side", "square", or "fixed_size". Defaults to "fixed_mapping".
        size (int or tuple, optional): Required for "longest_side", "square", and "fixed_size" modes.
                                      - For "longest_side" and "square": int value for resize dimension
                                      - For "fixed_size": tuple of (width, height)
        norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to "dinov2".
        patch_size (int, optional): Patch size for image processing. Defaults to 14.
        verbose (bool, optional): If True, print progress messages. Defaults to False.
        bayer_format (bool, optional): If True, read images in Bayer format. Defaults to False.
        resolution_set (int, optional): Resolution set to use for "fixed_mapping" mode (518 or 512). Defaults to 518.
        stride (int, optional): Load every nth image from the input. stride=1 loads all images, stride=2 loads every 2nd image, etc. Defaults to 1.

    Returns:
        list: List of dictionaries containing image data and metadata
    """
    # Validate resize_mode and size parameter requirements
    valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
        raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

    # Validate size type based on resize mode
    if resize_mode in ["longest_side", "square"]:
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
            )
    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    # Get list of image paths
    if isinstance(folder_or_list, str) and not folder_or_list.endswith(
        (".jpg", ".jpeg", ".png", ".heic", ".heif")
    ):
        # If folder_or_list is a string, assume it's a path to a folder
        if verbose:
            print(f"Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, str) and folder_or_list.endswith(
        (".jpg", ".jpeg", ".png", ".heic", ".heif")
    ):
        # If folder_or_list is a string and points to a single image file
        if verbose:
            print(f"Loading a single image from {folder_or_list}")
        root, folder_content = "", [folder_or_list]
    elif isinstance(folder_or_list, list):
        # If folder_or_list is a list, assume it's a list of image paths
        if verbose:
            print(f"Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    elif isinstance(folder_or_list, PIL.Image.Image):
        # If folder_or_list is a PIL Image, treat it as a single image
        if verbose:
            print(f"Loading a single image from {folder_or_list}")
        root, folder_content = "", [folder_or_list]
    else:
        # If folder_or_list is neither a string nor a list, raise an error
        raise ValueError(f"Bad {folder_or_list=} ({type(folder_or_list)})")

    # Define supported image extensions
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    # First pass: Load all images and collect aspect ratios
    loaded_images = []
    aspect_ratios = []
    for i, path in enumerate(folder_content):
        # Skip images based on stride
        if i % stride != 0:
            continue

        # Check if the file has a supported image extension
        if not isinstance(path, PIL.Image.Image) and not path.lower().endswith(
            supported_images_extensions
        ):
            continue

        try:
            if bayer_format:
                # If bayer_format is True, read the image in Bayer format
                color_bayer = cv2.imread(os.path.join(root, path), cv2.IMREAD_UNCHANGED)
                color = cv2.cvtColor(color_bayer, cv2.COLOR_BAYER_RG2BGR)
                img = PIL.Image.fromarray(color)
                img = exif_transpose(img).convert("RGB")
            elif isinstance(path, PIL.Image.Image):
                img = exif_transpose(path).convert("RGB")
            else:
                # Otherwise, read the image normally
                img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert(
                    "RGB"
                )
            if orientation is not None:
                img = rotate_image(np.array(img), orientation)
                img = PIL.Image.fromarray(img)
            W1, H1 = img.size
            aspect_ratios.append(W1 / H1)
            loaded_images.append((path, img, W1, H1))

        except Exception as e:
            if verbose:
                print(f"Warning: Could not load {path}: {e}")
            continue

    # Check if any images were loaded
    if not loaded_images:
        raise ValueError("No valid images found")

    # Calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    if verbose:
        print(
            f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
        )

    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)
    elif resize_mode == "square":
        target_size = (
            round((size // patch_size)) * patch_size,
            round((size // patch_size)) * patch_size,
        )
    elif resize_mode == "longest_side":
        # Use average aspect ratio to determine size for all images
        # Longest side should be the input size
        if average_aspect_ratio >= 1:  # Landscape or square
            # Width is the longest side
            target_size = (
                size,
                round((size // patch_size) / average_aspect_ratio) * patch_size,
            )
        else:  # Portrait
            # Height is the longest side
            target_size = (
                round((size // patch_size) * average_aspect_ratio) * patch_size,
                size,
            )
    elif resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )

    if verbose:
        print(
            f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all images"
        )

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    # Second pass: Resize all images to the same target size
    imgs = []
    for path, img, W1, H1 in loaded_images:
        # Resize and crop the image to the target size
        img = crop_resize_if_necessary(img, resolution=target_size)[0]

        # Normalize image and add it to the list
        W2, H2 = img.size
        if verbose:
            print(f" - Adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")

        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
            )
        )

    assert imgs, "No images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")

    return imgs


# =========================================================
# the code adapted from depthanythingv3


def preprocess_inputs(
    image: list[np.ndarray | Image.Image | str],
    input_processor: Callable[[np.ndarray | Image.Image | str], np.ndarray],
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from depth_anything_3.utils.logger import logger

    """Preprocess input images using input processor."""
    start_time = time.time()
    imgs_cpu, extrinsics, intrinsics = input_processor(
        image,
        extrinsics.copy() if extrinsics is not None else None,
        intrinsics.copy() if intrinsics is not None else None,
        process_res,
        process_res_method,
    )
    end_time = time.time()
    # logger.info(
    #     "Processed Images Done taking",
    #     end_time - start_time,
    #     "seconds. Shape: ",
    #     imgs_cpu.shape,
    # )
    return imgs_cpu, extrinsics, intrinsics


def prepare_model_inputs(
    imgs_cpu: torch.Tensor,
    extrinsics: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Prepare tensors for model input."""

    # Move images to model device
    imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

    # Convert camera parameters to tensors
    ex_t = (
        extrinsics.to(device, non_blocking=True)[None].float()
        if extrinsics is not None
        else None
    )
    in_t = (
        intrinsics.to(device, non_blocking=True)[None].float()
        if intrinsics is not None
        else None
    )

    return imgs, ex_t, in_t


# ========================================================


def filter_object(objects_in_frame, first_frame_id, resolution=(1024, 1024)):
    filtered_objects = {}
    for obj_id, obj in objects_in_frame.items():
        valid = False
        frame_data = obj.get(first_frame_id, {})
        assert frame_data != {}, "Frame data should not be empty here."
        if "camera_box_2d_rend" in frame_data:
            x1, y1, x2, y2 = frame_data["camera_box_2d_rend"]
            x1_2d_p, y1_2d_p, x2_2d_p, y2_2d_p = frame_data["camera_box_2d_proj"]
            cx_2d_p = (x1_2d_p + x2_2d_p) / 2
            cy_2d_p = (y1_2d_p + y2_2d_p) / 2
            cz = frame_data["camera_center"][2]
            w, h, l = frame_data["camera_scale"]
            if (
                cz > 0.1
                and (w > 0.3 or h > 0.3 or l > 0.3)
                and (w > 0.10 and h > 0.10 and l > 0.10)
                and cx_2d_p > 10
                and cx_2d_p < resolution[1] - 10
                and cy_2d_p > 10
                and cy_2d_p < resolution[0] - 10
                and filter_object_2d(x1, y1, x2, y2, resolution[1], resolution[0])
            ):
                valid = True
        if valid:
            filtered_objects[obj_id] = obj
    return filtered_objects


def filter_object_2d(x1, y1, x2, y2, h, w):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if x2 - x1 < 30 or y2 - y1 < 30:
        return False
    if cx < 0 and x2 / (x2 - cx) < 0.2:
        return False
    if cy < 0 and y2 / (y2 - cy) < 0.2:
        return False
    if cx > 1024 and (w - x1) / (cx - x1) < 0.2:
        return False
    if cy > 1024 and (h - y1) / (cy - y1) < 0.2:
        return False
    if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
        return False
    return True  # this bbox is valid


DATASET_LOOKUP_TABLE = {
    "CA-1M": 0,
    "vsibench-scannet": 1,
    "vsibench-arkitscenes": 2,
    "vsibench-scannetpp": 3,
    "hyperism": 4,
    "ADT": 5
}
REVERSE_DATASET_LOOKUP_TABLE = {v: k for k, v in DATASET_LOOKUP_TABLE.items()}
NOT_APPEAR_FLAG = -1000


def box_xyxy_to_cxcywh(bbox):
    """
    把 (x1, y1, x2, y2) 转换成 (cx, cy, w, h)
    bbox: (..., 4)
    """
    cx = (bbox[..., 0] + bbox[..., 2]) / 2
    cy = (bbox[..., 1] + bbox[..., 3]) / 2
    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def get_alpha_from_bbox_2d(bbox_2d, K, yaw=None):
    """
    根据 2D bbox + 相机内参计算 alpha
    bbox_2d: [frame_num, object_num, 4]  (x1,y1,x2,y2)
             若该帧物体不存在，则为 -1000,-1000,-1000,-1000
    K:       [frame_num, 3, 3]  相机内参
    yaw:     [frame_num, object_num] 3D 物体的旋转角 (rotation_y)，可选
             如果不传，就只返回 arctan 部分
    """
    # 中心点 (cx, cy)
    bbox_cx = box_xyxy_to_cxcywh(bbox_2d)[..., 0]
    # 相机参数
    cx = K[:, None, 0, 2]  # shape: [frame_num, 1]
    fx = K[:, None, 0, 0]  # shape: [frame_num, 1]
    # 计算 alpha
    alpha = -np.arctan2(bbox_cx - cx, fx)
    if yaw is not None:
        yaw = yaw
        alpha = alpha + yaw
    # 归一化到 [-pi, pi]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    # 处理不存在的情况 (bbox = -1000)
    mask = bbox_2d[..., 0] == NOT_APPEAR_FLAG
    alpha[mask] = NOT_APPEAR_FLAG
    return alpha


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

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
