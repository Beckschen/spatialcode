import torch
import numpy as np

# ==========================================
# 1. 四元数基础运算
# ==========================================

def normalize_quat(q, eps=1e-8):
    """
    q: (..., 4) [w, x, y, z]
    """
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)


def quat_mul(q1, q2):
    """
    Hamilton 乘法: q = q1 * q2
    q1, q2: (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    q = torch.stack([w, x, y, z], dim=-1)
    return normalize_quat(q)


def quat_inv(q):
    """
    对单位四元数求逆: q^{-1} = [w, -x, -y, -z]
    q: (..., 4)
    """
    q = normalize_quat(q)
    w, xyz = q[..., 0:1], q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


def quat_log(q, eps=1e-8):
    """
    单位四元数的对数映射: log: S^3 -> R^3
    q: (..., 4)
    返回: (..., 3)
    """
    q = normalize_quat(q)
    w = torch.clamp(q[..., 0], -1.0 + 1e-7, 1.0 - 1e-7)
    v = q[..., 1:]  # (..., 3)
    v_norm = torch.norm(v, dim=-1, keepdim=True)  # (..., 1)
    
    # 角度
    theta = torch.acos(w)  # (...,)

    # 对于很小的旋转，log 接近 0，直接返回 0 更稳定
    small = v_norm < eps
    scale = torch.zeros_like(v_norm)
    scale[~small] = theta.unsqueeze(-1)[~small] / (v_norm[~small] + eps)
    
    omega = v * scale  # (..., 3)
    omega[small.expand_as(omega)] = 0.0
    return omega


def quat_exp(omega, eps=1e-8):
    """
    李代数到单位四元数的指数映射: exp: R^3 -> S^3
    omega: (..., 3)
    返回: (..., 4)
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)  # (..., 1)
    small = theta < eps
    
    w = torch.zeros_like(theta)
    xyz = torch.zeros_like(omega)
    
    # --- 1. 一般情况 (Large Angle) ---
    w[~small] = torch.cos(theta[~small])
    
    # 计算 scale (只计算大角度部分，避免除零)
    scale = torch.zeros_like(theta)
    scale[~small] = torch.sin(theta[~small]) / (theta[~small] + eps)

    mask_xyz = small.expand_as(xyz)
    product = omega * scale  
    xyz[~mask_xyz] = product[~mask_xyz]
    
    # --- 2. 小角度近似 (Small Angle) ---
    # exp(omega) ≈ [1, omega]
    w[small] = 1.0
    # 直接将 omega 的值赋给 xyz (此时 scale 默认为 1)
    xyz[mask_xyz] = omega[mask_xyz]
    
    q = torch.cat([w, xyz], dim=-1)
    return normalize_quat(q)


def ensure_quat_continuity(q_curr, q_prev):
    """
    处理 q 和 -q 的双覆盖问题，使得与上一帧 q_prev 尽量同向。
    q_curr, q_prev: (B, 4)
    """
    # 点积: (B, J, 1)
    dot = torch.sum(q_curr * q_prev, dim=-1, keepdim=True)
    mask_flip = dot < 0.0  # 需要翻转的地方
    q_curr_adj = torch.where(mask_flip, -q_curr, q_curr)
    return q_curr_adj


# ==========================================
# 2. 置信度到平滑系数的映射
# ==========================================

def conf_to_weight(c, beta=1.0):
    """
    将正数置信度 c (通常是 exp(raw_conf)) 压到 (0, 1) 区间。
    c: (B, 1) 或 (B,)
    返回: (B, 1)
    """
    if c.dim() == 1:
        c = c.view(-1, 1)
    elif c.dim() == 2 and c.shape[1] != 1:
        raise ValueError("confidence tensor shape 应为 (B,) 或 (B,1)")
    
    # 常见选择之一: s = c / (c + 1)，也可以改成 1 - exp(-beta * c)
    s = c / (c + 1.0)
    # s = 1.0 - torch.exp(-beta * c)
    return s


# ==========================================
# 3. 基于四元数和 depth confidence 的在线时序平滑
# ==========================================
class QuaternionSmoother:
    def __init__(
        self,
        alpha_rot_min=0.005, # 建议调小下限，让稳定时更稳
        alpha_rot_max=0.1, 
        device="cuda"
    ):
        self.device = device
        self.alpha_rot_min = alpha_rot_min
        self.alpha_rot_max = alpha_rot_max
        self.prev_quat = None
        self.prev_mask = None # 新增：记录上一帧的追踪状态

    def reset(self):
        self.prev_quat = None
        self.prev_mask = None

    @torch.no_grad()
    def __call__(self, quat, depth_conf, mask=None):
        """
        quat: (B, J, 4) 或 (B, 4)
        depth_conf: (B,)
        mask: (B,) boolean, 指示当前哪些物体是有效的(被追踪的)
        """
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float32, device=self.device)
        else:
            quat = quat.to(self.device)

        # 1. 维度处理与归一化
        input_dim = quat.dim()
        if input_dim == 2: # (B, 4) -> (B, 1, 4) 统一处理
            quat = quat.unsqueeze(1)
        
        B, J, _ = quat.shape
        quat = normalize_quat(quat)
        
        # 2. 处理 Mask (至关重要)
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=self.device)
        
        # 广播 mask 到 (B, 1, 1) 以便后续计算
        mask_expanded = mask.view(B, 1, 1).expand(B, J, 1) # used for values
        
        # 3. 初始化或重置
        if self.prev_quat is None or self.prev_quat.shape != quat.shape:
            self.prev_quat = quat.clone()
            self.prev_mask = mask.clone()
            # 第一帧直接返回，不做平滑
            return quat.squeeze(1) if input_dim == 2 else quat

        # 4. 识别“新出现的物体” (Rising Edge: False -> True)
        # 如果上一帧没追踪(False)，这一帧追踪了(True)，说明是新物体(或刚找回)，必须强制重置！
        # 否则它会和上一帧的“垃圾/零值/旧物体”平滑，导致第一帧倾斜。
        is_new_object = mask & (~self.prev_mask)
        is_new_object_expanded = is_new_object.view(B, 1, 1).expand(B, J, 1)

        # 5. 计算 Alpha
        if depth_conf.dim() == 1:
            depth_conf = depth_conf.view(B, 1)
        s = conf_to_weight(depth_conf)
        alpha_rot = self.alpha_rot_min + s * (self.alpha_rot_max - self.alpha_rot_min)
        alpha_rot = alpha_rot.view(B, 1, 1).expand(B, J, 1)

        # 6. 平滑逻辑
        # (a) 确保连续性
        q_curr = ensure_quat_continuity(quat, self.prev_quat)
        
        # (b) 计算增量 delta
        delta_q = quat_mul(quat_inv(self.prev_quat), q_curr)
        omega = quat_log(delta_q)
        omega_tilde = alpha_rot * omega
        delta_q_tilde = quat_exp(omega_tilde)
        
        # (c) 应用增量
        smooth_quat = quat_mul(self.prev_quat, delta_q_tilde)
        smooth_quat = normalize_quat(smooth_quat)
        
        # 7. 关键：混合输出与状态更新
        # 如果是新物体(is_new_object)，强制使用当前原始值(quat)，忽略平滑结果
        # 如果是正在追踪的物体(mask)，使用平滑结果(smooth_quat)
        # 如果是未追踪的物体(~mask)，保持当前值或历史值均可(反正输出会被外层过滤)
        
        final_quat = torch.where(is_new_object_expanded, quat, smooth_quat)
        
        # 更新历史状态
        # 只有当前被追踪(mask=True)时，才更新历史为 final_quat
        # 如果当前丢失(mask=False)，我们暂存当前的 raw input 作为历史，
        # 这样万一下一帧它又回来了且没触发 is_new_object (虽然逻辑上应该触发)，也能有个底。
        # 但最重要的是记录 mask 状态。
        self.prev_quat = torch.where(mask_expanded, final_quat, quat).detach()
        self.prev_mask = mask.clone() # 记录这一帧谁是有效的

        if input_dim == 2:
            final_quat = final_quat.squeeze(1)
            
        return final_quat