import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # 用于在开裂图上精确绘制红色洞口边框
import seaborn as sns
import time  # 用于控制动画播放的帧率间隔

# ==================== 🎨 全局无差别纯黑直角工业风 CSS 注入 ====================
st.markdown(
    """
    <style>
    # ==================== 🎨 极简流线型折叠面板 (Expander) CSS 强力注入 ====================
    /* 1. 彻底斩断折叠面板原生的四周外框、阴影和圆角，将其软化为纯透明底色 */
    div[data-testid="stExpander"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0px !important;
        
        /* 2. 核心改动：在每个组件下方强行拉出一条干净的直通水平线，作为区域分割线 */
        border-bottom: 1px solid #d3d3d3 !important; /* 优雅的工业浅灰线，可根据深色模式调整为 #444444 */
        
        /* 3. 精细微调间距，让标题、分割线紧凑排布，消除多余空隙 */
        margin-bottom: 10px !important;
        padding-bottom: 6px !important;
    }

    /* 4. 强制抹除鼠标悬浮在标题栏上时突兀蹦出来的原生灰色背景背景槽 */
    div[data-testid="stExpander"] > details > summary:hover {
        background-color: transparent !important;
    }

    /* 5. 消除点击展开后，内部表单/滑动条内容区域的多余边框与两侧缩进 */
    div[data-testid="stExpander"] > div {
        border: none !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ==================== 0. 基础配置与中文字体修复 ====================
st.set_page_config(page_title="砌体墙双板交互比对破坏预测系统", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib.font_manager as fm

custom_font_path = os.path.join(BASE_DIR, "SimHei.ttf")
GLOBAL_FONT_PROP = None

# 1. 尝试注册本地打包的 SimHei 字体
if os.path.exists(custom_font_path):
    try:
        fm.fontManager.addfont(custom_font_path)
        GLOBAL_FONT_PROP = fm.FontProperties(fname=custom_font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'Arial Unicode MS']
        st.toast("已成功加载仓库自定义 SimHei 字体库！")
    except Exception as e:
        pass

# 2. 如果本地字体缺失或加载失败，启动强力 Linux 原生防崩盘降级
if GLOBAL_FONT_PROP is None:
    linux_wqy_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if os.path.exists(linux_wqy_path):
        GLOBAL_FONT_PROP = fm.FontProperties(fname=linux_wqy_path)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False

# 📌 路径资产精准自适应锚定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts") 
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MANIFEST_PATH = os.path.join(DATASET_DIR, "processed_manifest_clean.xlsx")
STEP3_DATA_DIR = os.path.join(DATASET_DIR, "processed_data_step3")
WALL_TEXTURE_PATH = os.path.join(DATASET_DIR, "board.png")

MODEL_ONE_PATH = os.path.join(SCRIPTS_DIR, "best_model_one.pth")
MODEL_TWO_PATH = os.path.join(SCRIPTS_DIR, "best_model_two.pth")
MODEL_THREE_PATH = os.path.join(SCRIPTS_DIR, "best_step3_model.pth")

base_load_gt = 0.0

# ==================== 1. 严格同步后端的满血版网络声明 ====================
class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class MasonryCrackPredictor(nn.Module):
    def __init__(self, latent_dim=32): 
        super(MasonryCrackPredictor, self).__init__()
        
        self.base_embedding = nn.Embedding(num_embeddings=16, embedding_dim=32)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(24, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU()
        )
        
        self.fusion_bridge = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 64 * 8 * 8), nn.ReLU() 
        )
        
        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.res_block2 = ResBlock2d(in_channels=34, out_channels=32)
        
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.res_block1 = ResBlock2d(in_channels=18, out_channels=16)
        
        self.final_layer = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def _inject_coordinate_system(self, tensor):
        B, _, H, W = tensor.shape
        x_coords = torch.linspace(-1.0, 1.0, W, device=tensor.device).view(1, 1, 1, W).expand(B, 1, H, W)
        y_coords = torch.linspace(-1.0, 1.0, H, device=tensor.device).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([tensor, x_coords, y_coords], dim=1)

    def forward(self, base_id, target_scalars, spatial_masks):
        B, C, H, W = spatial_masks.shape
        spatial_with_coords = self._inject_coordinate_system(spatial_masks)
        
        E_base = self.base_embedding(base_id)
        E_scalar = self.scalar_encoder(target_scalars)
        E_spatial = self.spatial_cnn(spatial_with_coords)
        
        fused_vec = torch.cat([E_base, E_scalar, E_spatial], dim=-1)
        hidden_spatial = self.fusion_bridge(fused_vec).view(B, 64, 8, 8)
        
        x_16 = self.up2(hidden_spatial)                  
        x_16_with_coords = self._inject_coordinate_system(x_16) 
        x_16_feated = self.res_block2(x_16_with_coords)
        
        x_32 = self.up1(x_16_feated)                     
        x_32_with_coords = self._inject_coordinate_system(x_32) 
        x_32_feated = self.res_block1(x_32_with_coords)
        
        crack_matrix_raw = self.final_layer(x_32_feated)
        return crack_matrix_raw.view(B, 32, 32)

class MasonryLoadRegressor(nn.Module):
    def __init__(self, visual_bottleneck_dim=8):
        super(MasonryLoadRegressor, self).__init__()
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, visual_bottleneck_dim), nn.ReLU()
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(24, 32), nn.ReLU(), nn.Dropout(p=0.2)
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(visual_bottleneck_dim + 32, 32), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(32, 1)
        )
    def forward(self, target_scalars, spatial_masks, pred_crack_mask):
        if len(pred_crack_mask.shape) == 3:
            pred_crack_mask = pred_crack_mask.unsqueeze(1)
        mega_damage_tensor = torch.cat([spatial_masks, pred_crack_mask], dim=1)
        E_visual = self.spatial_cnn(mega_damage_tensor)    
        E_scalar = self.scalar_encoder(target_scalars)     
        E_fusion = torch.cat([E_visual, E_scalar], dim=-1)  
        return self.regressor_head(E_fusion).squeeze(-1)

class SpatialTemporalInflectionNet(nn.Module):
    def __init__(self, static_feature_dim=25, hidden_dim=64): 
        super(SpatialTemporalInflectionNet, self).__init__()
        
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        
        self.temporal_gru = nn.GRU(input_size=512, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim + static_feature_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x_disp, x_mask, x_static, lengths):
        batch_size, time_steps, H, W = x_disp.size()
        x_disp = x_disp * x_mask.unsqueeze(1)
        x_cnn_in = x_disp.view(batch_size * time_steps, 1, H, W)
        spatial_feats = self.spatial_cnn(x_cnn_in)
        spatial_feats = spatial_feats.view(batch_size, time_steps, -1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            spatial_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.temporal_gru(packed_input)
        temporal_features = h_n[-1]
        fused_features = torch.cat([temporal_features, x_static], dim=1)
        return self.regression_head(fused_features)

# ==================== 2. 环境控制与核心资产动态初始化 ====================
@st.cache_resource
def bootstrap_system_environment():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"找不到干净的特征清单表，请核对路径: {MANIFEST_PATH}")
        
    df = pd.read_excel(MANIFEST_PATH)
    unique_bases = sorted(df['基准板编号'].astype(str).unique())
    base_name_to_id = {name: idx for idx, name in enumerate(unique_bases)}
    
    L_max = 5615.0 
    
    model_one = MasonryCrackPredictor(latent_dim=32)
    model_two = MasonryLoadRegressor(visual_bottleneck_dim=8)
    model_three = SpatialTemporalInflectionNet(static_feature_dim=25, hidden_dim=64)
    
    if os.path.exists(MODEL_ONE_PATH):
        model_one.load_state_dict(torch.load(MODEL_ONE_PATH, map_location='cpu'))
    else:
        raise FileNotFoundError(f"找不到 Step 1 权重: {MODEL_ONE_PATH}")
        
    if os.path.exists(MODEL_TWO_PATH):
        model_two.load_state_dict(torch.load(MODEL_TWO_PATH, map_location='cpu'))
    else:
        raise FileNotFoundError(f"找不到 Step 2 权重: {MODEL_TWO_PATH}")
        
    if os.path.exists(MODEL_THREE_PATH):
        model_three.load_state_dict(torch.load(MODEL_THREE_PATH, map_location='cpu'))
    else:
        raise FileNotFoundError(f"找不到 Step 3 权重: {MODEL_THREE_PATH}")
        
    model_one.eval()
    model_two.eval()
    model_three.eval()
    
    return df, base_name_to_id, model_one, model_two, model_three, L_max

try:
    df, base_name_to_id, model_one, model_two, model_three, L_max = bootstrap_system_environment()
except Exception as e:
    st.error(f"系统启动时触发安全拦截: {str(e)}")
    st.stop()

def plot_matrix_3d_voxels(matrix, title, wall_mask, wall_len, wall_hit, wall_thick, crop_bounds=None):
    from mpl_toolkits.mplot3d import Axes3D
    
    if crop_bounds is not None:
        r_start, r_end, c_start, c_end = crop_bounds
        matrix = matrix[r_start:r_end, c_start:c_end]
        wall_mask = wall_mask[r_start:r_end, c_start:c_end]
        
    H, W = matrix.shape
    if int(wall_thick) == 180:
        D = 5  
        is_cavity_wall = True
    else:
        D = 2  
        is_cavity_wall = False
        
    filled = np.zeros((H, W, D), dtype=bool)
    colors = np.empty(filled.shape, dtype=object)
    
    for d in range(D):
        if is_cavity_wall and d == 2:
            continue 
        filled[:, :, d] = (wall_mask > 0)
        colors[filled[:, :, d], d] = '#abcdefcc' 
        
    crack_mask = (matrix > 0)
    for d in range(D):
        if is_cavity_wall and d == 2: 
            continue  
        filled[crack_mask, d] = True
        colors[crack_mask, d] = '#1c1c1c'
        
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.voxels(filled, facecolors=colors, edgecolors='#ffffff', linewidth=0.2)
    ax.set_box_aspect((wall_hit, wall_len, wall_thick))
    ax.view_init(elev=25, azim=-55)
    ax.set_axis_off()  
    ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=9, pad=2)
    return fig

# ==================== 3. 几何与力学高级后处理算法中枢 ====================
def repair_crack_connectivity(binary_mask, max_gap=2):
    mask = binary_mask.copy().astype(np.uint8)
    H, W = mask.shape
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    neighbor_count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere(neighbor_count == 11)
    
    for r, c in endpoints:
        found = False
        for d in range(1, max_gap + 1):
            if found: break
            for dr in range(-d, d + 1):
                if found: break
                for dc in range(-d, d + 1):
                    if abs(dr) != d and abs(dc) != d: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if mask[nr, nc] == 1:
                            cv2.line(mask, (int(c), int(r)), (int(nc), int(nr)), 1, 1)
                            found = True
                            break
    return mask.astype(np.float32)

def bridge_internal_broken_cracks(skeleton_binary, max_gap=4):
    mask = skeleton_binary.copy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 2:
        return mask.astype(np.float32)
        
    new_mask = mask.copy()
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    neighbor_count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
    comp_endpoints = {}
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        eps = np.argwhere(comp_mask & (neighbor_count == 11))
        if len(eps) == 0: eps = np.argwhere(comp_mask)
        comp_endpoints[i] = eps

    for i in range(1, num_labels):
        eps_i = comp_endpoints[i]
        for j in range(i + 1, num_labels):
            eps_j = comp_endpoints[j]
            dists = np.sum((eps_i[:, None, :] - eps_j[None, :, :]) ** 2, axis=-1)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = np.sqrt(dists[min_idx])
            
            if min_dist <= max_gap:
                pt_i = tuple(eps_i[min_idx[0]])
                pt_j = tuple(eps_j[min_idx[1]])
                cv2.line(new_mask, (pt_i[1], pt_i[0]), (pt_j[1], pt_j[0]), 1, 1)
    return new_mask.astype(np.float32)

def imitate_benchmark_crack_placement(skeleton_binary, base_crack_gt, boundary_mask, max_gap=7):
    mask = skeleton_binary.copy().astype(np.uint8)
    base_mask = (base_crack_gt > 0.5).astype(np.uint8)
    if np.sum(mask) == 0: return mask.astype(np.float32)
        
    boundary_pts = np.argwhere(boundary_mask > 0) if np.sum(boundary_mask) > 0 else None
    num_pred, labels_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_base, labels_base, stats_base, centroids_base = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
    
    st.toast(f"[双重因果辨识] 预测碎片: {num_pred-1} 条 | 基准主裂缝: {num_base-1} 条")
    new_mask = np.zeros_like(mask)
    
    for p_label in range(1, num_pred):
        p_pts = np.argwhere(labels_pred == p_label)
        p_centroid = centroids_pred[p_label]
        near_hole = False
        dr_hole, dc_hole = 0, 0
        if boundary_pts is not None:
            dists_to_hole = np.sum((p_pts[:, None, :] - boundary_pts[None, :, :]) ** 2, axis=-1)
            min_hole_idx = np.unravel_index(np.argmin(dists_to_hole), dists_to_hole.shape)
            min_hole_dist = np.sqrt(dists_to_hole[min_hole_idx])
            if min_hole_dist <= max_gap:
                near_hole = True
                p_anchor_hole = p_pts[min_hole_idx[0]]
                b_anchor_hole = boundary_pts[min_hole_idx[1]]
                dr_hole = b_anchor_hole[0] - p_anchor_hole[0]
                dc_hole = b_anchor_hole[1] - p_anchor_hole[1]
                
        best_b_label = None
        min_b_dist = float('inf')
        for b_label in range(1, num_base):
            b_centroid = centroids_base[b_label]
            dist = np.sum((p_centroid - b_centroid) ** 2)
            if dist < min_b_dist:
                min_b_dist = dist
                best_b_label = b_label
                
        dr_bench, dc_bench = 0, 0
        has_bench_match = False
        if best_b_label is not None and num_base > 1:
            b_pts = np.argwhere(labels_base == best_b_label)
            dists_between = np.sum((p_pts[:, None, :] - b_pts[None, :, :]) ** 2, axis=-1)
            min_bench_idx = np.unravel_index(np.argmin(dists_between), dists_between.shape)
            if np.sqrt(dists_between[min_bench_idx]) <= max_gap:
                has_bench_match = True
                p_anchor_bench = p_pts[min_bench_idx[0]]
                b_anchor_bench = b_pts[min_bench_idx[1]]
                dr_bench = b_anchor_bench[0] - p_anchor_bench[0]
                dc_bench = b_anchor_bench[1] - p_anchor_bench[1]

        if near_hole: dr, dc = dr_hole, dc_hole
        elif has_bench_match: dr, dc = dr_bench, dc_bench
        else: dr, dc = 0, 0
            
        for cr, cc in p_pts:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                new_mask[nr, nc] = 1
    return new_mask.astype(np.float32)

def bridge_cracks_to_wall_boundary(crack_binary, crop_bounds, max_gap=4):
    mask = crack_binary.copy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    r_start, r_end, c_start, c_end = crop_bounds
    new_mask = mask.copy()
    
    for label in range(1, num_labels):
        comp_mask = (labels == label).astype(np.uint8)
        comp_pts = np.argwhere(comp_mask > 0)
        min_d = float('inf')
        best_pt = None
        bound_dest = None 
        
        for r, c in comp_pts:
            d_top = abs(r - r_start)
            d_bottom = abs(r - (r_end - 1))
            d_left = abs(c - c_start)
            d_right = abs(c - (c_end - 1))
            local_min = min(d_top, d_bottom, d_left, d_right)
            if local_min < min_d:
                min_d = local_min
                best_pt = (r, c)
                if local_min == d_top: bound_dest = (r_start, c)
                elif local_min == d_bottom: bound_dest = (r_end - 1, c)
                elif local_min == d_left: bound_dest = (r, c_start)
                else: bound_dest = (r, c_end - 1)
                
        if 0 < min_d <= max_gap and best_pt is not None and bound_dest is not None:
            cv2.line(new_mask, (best_pt[1], best_pt[0]), (bound_dest[1], bound_dest[0]), 1, 1)
    return new_mask.astype(np.float32)

def bridge_directional_extension_segments(skeleton_binary, boundary_mask, max_gap=22):
    mask = skeleton_binary.copy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 2: return mask.astype(np.float32)
        
    new_mask = mask.copy()
    hole_associated_labels = []
    for i in range(1, num_labels):
        pts = np.argwhere(labels == i)
        for r, c in pts:
            if boundary_mask[r, c] > 0:
                hole_associated_labels.append(i)
                break
                
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    for h_label in hole_associated_labels:
        h_pts = np.argwhere(labels == h_label)
        neighbor_count = cv2.filter2D((labels == h_label).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        h_eps = np.argwhere((labels == h_label) & (neighbor_count == 11))
        if len(h_eps) == 0: h_eps = h_pts
            
        boundary_pts = np.argwhere(boundary_mask > 0)
        outer_ep = None
        max_d_to_hole = -1
        for er, ec in h_eps:
            dists = np.sum((boundary_pts - np.array([er, ec]))**2, axis=1)
            min_d = np.min(dists)
            if min_d > max_d_to_hole:
                max_d_to_hole = min_d
                outer_ep = (er, ec)
                
        if outer_ep is None: continue
        curr_r, curr_c = outer_ep
        dists_to_ep = np.sum((h_pts - np.array([curr_r, curr_c]))**2, axis=1)
        inner_indices = np.where((dists_to_ep > 0) & (dists_to_ep <= 16))[0]
        if len(inner_indices) > 0:
            mean_inner = np.mean(h_pts[inner_indices], axis=0)
            dr = curr_r - mean_inner[0]
            dc = curr_c - mean_inner[1]
            norm = np.sqrt(dr**2 + dc**2)
            if norm > 0: dr, dc = dr / norm, dc / norm
            else: dr, dc = 0.7, 0.7
        else:
            dr, dc = 0.7, 0.7
            
        while True:
            current_num, current_labels, _, _ = cv2.connectedComponentsWithStats(new_mask, connectivity=8)
            best_target_label = None
            best_target_pt = None
            min_target_dist = float('inf')
            
            for j in range(1, current_num):
                j_pts = np.argwhere(current_labels == j)
                if any((jr == curr_r and jc == curr_c) for jr, jc in j_pts): continue
                    
                for jr, jc in j_pts:
                    v_r = jr - curr_r
                    v_c = jc - curr_c
                    dist = np.sqrt(v_r**2 + v_c**2)
                    if dist <= max_gap:
                        cos_angle = (v_r * dr + v_c * dc) / (dist + 1e-6)
                        if cos_angle > 0.707:  
                            if dist < min_target_dist:
                                min_target_dist = dist
                                best_target_pt = (jr, jc)
                                best_target_label = j
                                
            if best_target_pt is not None:
                jr, jc = best_target_pt
                cv2.line(new_mask, (int(curr_c), int(curr_r)), (int(jc), int(jr)), 1, 1)
                tgt_pts = np.argwhere(current_labels == best_target_label)
                proj_distances = [(ptr - curr_r) * dr + (ptc - curr_c) * dc for ptr, ptc in tgt_pts]
                furthest_idx = np.argmax(proj_distances)
                new_r, new_c = tgt_pts[furthest_idx]
                new_dr, new_dc = new_r - curr_r, new_c - curr_c
                new_norm = np.sqrt(new_dr**2 + new_dc**2)
                if new_norm > 0: dr, dc = new_dr / new_norm, new_dc / new_norm
                curr_r, curr_c = new_r, new_c
            else:
                break
    return new_mask.astype(np.float32)

def get_boundary_crack_seeds(crack_binary, boundary_mask, crop_bounds, has_hole):
    seeds = []
    r_start, r_end, c_start, c_end = crop_bounds
    crack_pts = np.argwhere(crack_binary > 0)
    if len(crack_pts) == 0: return seeds
        
    if has_hole and np.sum(boundary_mask) > 0:
        boundary_pts = np.argwhere(boundary_mask > 0)
        for cr, cc in crack_pts:
            dists = np.sum((boundary_pts - np.array([cr, cc]))**2, axis=1)
            if np.min(dists) <= 2.0: seeds.append((cr, cc))
    else:
        for cr, cc in crack_pts:
            if cr == r_start or cr == r_end - 1 or cc == c_start or cc == c_end - 1:
                seeds.append((cr, cc))
    if len(seeds) == 0: seeds.append(tuple(crack_pts[0]))
    return seeds

def generate_crack_evolution_frames(crack_mask, seeds, num_frames=30):
    H, W = crack_mask.shape
    distance_field = np.full((H, W), -1, dtype=np.int32)
    if len(seeds) == 0: return [np.zeros((H, W)) for _ in range(num_frames)]
        
    queue = []
    for r, c in seeds:
        if crack_mask[r, c] > 0:
            distance_field[r, c] = 0
            queue.append((r, c))
    if len(queue) == 0:
        for r, c in seeds:
            distance_field[r, c] = 0
            queue.append((r, c))
    
    max_dist = 0
    while queue:
        r, c = queue.pop(0)
        curr_dist = distance_field[r, c]
        if curr_dist > max_dist: max_dist = curr_dist
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and crack_mask[nr, nc] > 0:
                    if distance_field[nr, nc] == -1:
                        distance_field[nr, nc] = curr_dist + 1
                        queue.append((nr, nc))
                        
    distance_field[(crack_mask > 0) & (distance_field == -1)] = max_dist + 1
    max_dist = max_dist + 1
    
    frames = []
    thresholds = np.linspace(0, max_dist, num=num_frames)
    for thres in thresholds:
        frame = np.zeros((H, W), dtype=np.float32)
        frame[(crack_mask > 0) & (distance_field <= thres)] = 1.0
        frames.append(frame)
    return frames

# 🌟【力学仿真级引擎高级重构】：高级制图函数（新增 hole_facecolor 参数与 zorder=10 强行覆盖机制）
def plot_matrix_heatmap(matrix, title, cmap="gray", vmin=0, vmax=1, hole_coords=None, crop_bounds=None, texture_path=None, texture_scale=1.0, dilate_first=False, hole_facecolor='none'):
    edge_color = 'red' if cmap == "gray_r" else '#444444'
    line_width = 2.0 if cmap == "gray_r" else 1.0

    # 🚀【精准矢量中点连线引擎】
    if cmap == "gray_r":
        binary = (matrix > 0.5).astype(np.uint8)  
        if dilate_first and np.any(binary):
            kernel = np.ones((3, 3), dtype=np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
        fig, _ax = plt.subplots(figsize=(3.5, 3.5))
        ax = _ax
        
        if crop_bounds is not None:
            r_start, r_end, c_start, c_end = crop_bounds
            binary_cropped = binary[r_start:r_end, c_start:c_end]
            if hole_coords is not None:
                hr_start, hr_end, hc_start, hc_end = hole_coords
                hole_coords = (hr_start - r_start, hr_end - r_start, hc_start - c_start, hc_end - c_start)
        else:
            binary_cropped = binary
            crop_bounds = (0, binary.shape[0], 0, binary.shape[1])
            
        h_c = crop_bounds[1] - crop_bounds[0]
        w_c = crop_bounds[3] - crop_bounds[2]
        
        if SKIMAGE_AVAILABLE and np.any(binary_cropped):
            skel = skeletonize(binary_cropped.astype(bool)).astype(np.uint8)
        else:
            skel = binary_cropped.astype(np.uint8)
            
        ax.set_xlim(0, w_c)
        ax.set_ylim(h_c, 0)  
        ax.set_facecolor('white')  
        ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=10, pad=8)
        ax.axis('off')
        
        if np.any(skel):
            rows, cols = np.where(skel == 1)
            pts = set(zip(rows, cols))
            drawn_nodes = set()
            for r, c in pts:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:  
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pts:
                        ax.plot([c + 0.5, nc + 0.5], [r + 0.5, nr + 0.5], color='black', linewidth=1.2, solid_capstyle='round')
                        drawn_nodes.add((r, c))
                        drawn_nodes.add((nr, nc))
            for r, c in pts:
                if (r, c) not in drawn_nodes:
                    ax.plot(c + 0.5, r + 0.5, marker='o', color='black', markersize=1.5)
                    
        if hole_coords is not None:
            r_start_h, r_end_h, c_start_h, c_end_h = hole_coords
            # 🛠️ 修改核心：引入 zorder=10，强制将图层移至最上面，并关联自定背景色 hole_facecolor
            rect = patches.Rectangle(
                (c_start_h - 0.5, r_start_h - 0.5),  # 👈 坐标向左上角移动了 0.5
                (c_end_h - c_start_h) + 1.0,         # 👈 宽度加了 1.0
                (r_end_h - r_start_h) + 1.0,         # 👈 高度加了 1.0
                linewidth=1.0,                 
                edgecolor='red',               
                facecolor=hole_facecolor,
                zorder=10
            )
            ax.add_patch(rect)
        ax.set_aspect('equal')
        return fig

    # 🧱 二维壁纸材质纹理平铺渲染引擎（仅对几何实体图 cmap="Blues" 生效）
    if cmap == "Blues" and texture_path and os.path.exists(texture_path):
        img = cv2.imread(texture_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            base_w = max(4, int(32 * texture_scale))
            base_h = max(4, int(32 * texture_scale))
            img_res = cv2.resize(img, (base_w, base_h))
            
            re_y = int(np.ceil(32 / img_res.shape[0])) + 1
            re_x = int(np.ceil(32 / img_res.shape[1])) + 1
            tiled = np.tile(img_res, (re_y, re_x, 1))
            full_texture = tiled[:32, :32, :]
            full_texture = full_texture.astype(np.float32) / 255.0
            
            if crop_bounds is not None:
                r_start, r_end, c_start, c_end = crop_bounds
                matrix_cropped = matrix[r_start:r_end, c_start:c_end]
                texture_cropped = full_texture[r_start:r_end, c_start:c_end]
                if hole_coords is not None:
                    hr_start, hr_end, hc_start, hc_end = hole_coords
                    hole_coords = (hr_start - r_start, hr_end - r_start, hc_start - c_start, hc_end - c_start)
            else:
                matrix_cropped = matrix
                texture_cropped = full_texture
                
            h_c, w_c = matrix_cropped.shape
            display_img = np.ones((h_c, w_c, 4), dtype=np.float32)
            bg_color = [240/255, 242/255, 246/255, 1.0]
            display_img[:] = bg_color
            
            mask_3d = (matrix_cropped == 1)
            display_img[mask_3d, :3] = texture_cropped[mask_3d]
            display_img[mask_3d, 3] = 1.0
            
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(display_img, extent=[0, w_c, h_c, 0])
            ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=10, pad=8)
            ax.axis('off')
            
            rect_outer = patches.Rectangle(
                (0, 0), w_c, h_c,
                linewidth=1.0,                 
                edgecolor='#444444',           
                facecolor='none'               
            )
            ax.add_patch(rect_outer)
            
            if hole_coords is not None:
                r_start_h, r_end_h, c_start_h, c_end_h = hole_coords
                rect = patches.Rectangle(
                    (c_start_h, r_start_h),            
                    c_end_h - c_start_h,               
                    r_end_h - r_start_h,               
                    linewidth=1.0,                 
                    edgecolor='#444444',           
                    facecolor='none'               
                )
                ax.add_patch(rect)
            return fig

    if crop_bounds is not None:
        r_start, r_end, c_start, c_end = crop_bounds
        matrix = matrix[r_start:r_end, c_start:c_end]
        if hole_coords is not None:
            hr_start, hr_end, hc_start, hc_end = hole_coords
            hole_coords = (hr_start - r_start, hr_end - r_start, hc_start - c_start, hc_end - c_start)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, square=True)
    ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=10, pad=8)
    
    if hole_coords is not None:
        r_start, r_end, c_start, c_end = hole_coords
        # 🛠️ 修改核心：引入 zorder=10，通用热力图绘制分支也做同步升级
        rect = patches.Rectangle(
            (c_start, r_start),            
            c_end - c_start,               
            r_end - r_start,               
            linewidth=2.0,                 
            edgecolor='red',               
            facecolor=hole_facecolor,
            zorder=10               
        )
        ax.add_patch(rect)
    return fig

def generate_spatial_masks(L, H, has_hole, hole_l, hole_h, x_dist, y_dist, SCALE):
    wall_mask = np.zeros((32, 32), dtype=np.uint8)
    w_px = int(round(L / SCALE))
    h_px = int(round(H / SCALE))
    w_px = min(32, max(1, w_px))
    h_px = min(32, max(1, h_px))
    
    r_wall_start, r_wall_end = 32 - h_px, 32
    c_wall_start, c_wall_end = 0, w_px
    wall_mask[r_wall_start:r_wall_end, c_wall_start:c_wall_end] = 1
    
    boundary_mask = np.zeros((32, 32), dtype=np.uint8)
    hole_coords = None
    
    if has_hole:
        hole_mask = np.zeros((32, 32), dtype=np.uint8)
        hole_w_px = int(round(hole_l / SCALE))
        hole_h_px = int(round(hole_h / SCALE))
        hole_left_px = int(round(x_dist / SCALE))
        hole_bottom_px = int(round(y_dist / SCALE))
        
        c_hole_start = max(0, c_wall_start + hole_left_px)
        c_hole_end = min(32, c_hole_start + hole_w_px)
        r_hole_end = min(32, r_wall_end - hole_bottom_px)
        r_hole_start = max(0, r_hole_end - hole_h_px)
        
        hole_mask[r_hole_start:r_hole_end, c_hole_start:c_hole_end] = 1
        wall_mask[hole_mask == 1] = 0
        
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary_mask = cv2.dilate(hole_mask, kernel, iterations=1) - hole_mask
        boundary_mask = boundary_mask * wall_mask
        hole_coords = (r_hole_start, r_hole_end, c_hole_start, c_hole_end)
        
    crop_bounds = (r_wall_start, r_wall_end, c_wall_start, c_wall_end)
    return wall_mask, boundary_mask, hole_coords, crop_bounds

# ==================== 4. 界面大标题与侧边栏配置 ====================
st.title("砌体墙双板交互比对破坏预测系统")
st.markdown("本系统搭载分工定制版 **直角坐标感知双塔专家解耦预测神经网络**。")
st.write("---")

with st.sidebar:
    st.markdown("### 输入新墙体基础数据")
    st.write("") 
    
    with st.expander("几何尺寸配置", expanded=True):
        wall_len = st.number_input("墙体长度 L (mm)", value=5615)
        wall_hit = st.number_input("墙体高度 H (mm)", value=2475)
        
        wall_layout = st.selectbox("墙体构造", options=['单叶墙', '空腔墙(50mm空腔)'], index=0)
        wall_thick = 65 if wall_layout == '单叶墙' else 180
        st.caption(f"当前构造自动锚定实体墙厚为: **{wall_thick} mm**")
        
        has_hole = st.checkbox("墙体是否包含开洞", value=False)
        if has_hole:
            hole_len = st.number_input("开洞长度 (mm)", value=2260)
            hole_hit = st.number_input("开洞高度 (mm)", value=1125)
            hole_x = st.number_input("洞口距左边界距离 (mm)", value=1677)
            hole_y = st.number_input("洞口距下边界距离 (mm)", value=900)
        else:
            hole_len, hole_hit, hole_x, hole_y = 0.0, 0.0, 0.0, 0.0

    with st.expander("四周边界约束条件"):
        b_top = st.selectbox("上边界约束类型", options=['BI', 'SS', 'FE'], index=2) 
        b_bottom = st.selectbox("下边界约束类型", options=['BI', 'SS', 'FE'], index=0) 
        b_left = st.selectbox("左边界约束类型", options=['BI', 'SS', 'FE'], index=1) 
        b_right = st.selectbox("右边界约束类型", options=['BI', 'SS', 'FE'], index=1) 

    with st.expander("材料属性配置"):
        wall_mat = st.selectbox("砌块材料类型", options=['B级面砖', '密实混凝土', 'A级工程砖'], index=0)
        wall_mortar = st.selectbox("砂浆配比比例", options=['1:01:06', '1:01:16', '1:1/2:4'], index=0)
        wall_asphalt = st.selectbox("是否存在底板沥青防潮层", options=['无', '有'], index=0)

    texture_scale = 0.10

st.sidebar.header("选择基准板")
selected_base_id = st.sidebar.selectbox("可对比基准板数据库", options=list(base_name_to_id.keys()), index=0)

SCALE = L_max / 32.0
pred_wall_mask, pred_boundary_mask, pred_hole_coords, pred_crop_bounds = generate_spatial_masks(wall_len, wall_hit, has_hole, hole_len, hole_hit, hole_x, hole_y, SCALE)

# ==================== 5. 左右双板沙盘渲染区 ====================
col_pred_info, col_base_info = st.columns(2)

# ==================== 🧱 模块一：待预测新构件当前空间形态 ====================
st.subheader("待预测新构件当前空间形态")
st.warning(f" **输入规格**: {wall_len}x{wall_hit}x{wall_thick}mm | **开洞状态**: {'开洞' if has_hole else '未开洞'}")

c_p1, c_p2 = st.columns([1.5, 1]) 
with c_p1: 
    st.pyplot(
        plot_matrix_heatmap(pred_wall_mask, "待预测新构件的相对墙体空间特征 (真实材质表面)", cmap="Blues", hole_coords=pred_hole_coords, crop_bounds=pred_crop_bounds, texture_path=WALL_TEXTURE_PATH, texture_scale=texture_scale), 
        use_container_width=True
    )
with c_p2: 
    st.markdown(f"""
    **力学特征清单速览：**
    * **构造墙厚**: {wall_layout} ({wall_thick}mm)
    * **材料配比**: {wall_mat} | 砂浆 [{wall_mortar}]
    * **边界约束**: 上`[{b_top}]` 下`[{b_bottom}]` 左`[{b_left}]` 右`[{b_right}]`
    * **沥青层**: {wall_asphalt}特殊处理
    """)

st.markdown("---")

# ==================== 🧱 模块二：选定基准板历史自检场追溯 ====================
st.subheader("选定的基准板情况")
try:
    base_self_row = df[(df['待预测板编号'] == selected_base_id) & (df['基准板编号'] == selected_base_id)].iloc[0]
    base_npz_path_raw = base_self_row['矩阵压缩包绝对路径']
    base_npz_filename = os.path.basename(base_npz_path_raw.replace('\\', '/'))
    actual_base_npz_path = os.path.join(DATASET_DIR, "processed_data", base_npz_filename)
    
    base_data = np.load(actual_base_npz_path)
    base_wall = base_data['wall_mask']
    base_boundary = base_data['boundary_mask']
    base_crack_gt = base_data['crack_mask']
    base_load_gt = float(base_data['load'])
    
    st.info(f"**基准板编号**: {selected_base_id}  |  **真实试验破坏荷载**: {base_load_gt:.2f} kN/m²")
    
    base_hole_coords = None
    base_crop_bounds = None
    r_w, c_w = np.where(base_wall == 1)
    if len(r_w) > 0 and len(c_w) > 0:
        base_crop_bounds = (r_w.min(), r_w.max() + 1, c_w.min(), c_w.max() + 1)
    
    if np.any(base_boundary > 0):
        rows, cols = np.where(base_wall == 0)
        if len(rows) > 0 and len(cols) > 0:
            if len(r_w) > 0:
                r_min, r_max = r_w.min(), r_w.max()
                c_min, c_max = c_w.min(), c_w.max()
                hole_rows = rows[(rows >= r_min) & (rows <= r_max) & (cols >= c_min) & (cols <= c_max)]
                hole_cols = cols[(rows >= r_min) & (rows <= r_max) & (cols >= c_min) & (cols <= c_max)]
                if len(hole_rows) > 0:
                    base_hole_coords = (hole_rows.min(), hole_rows.max() + 1, hole_cols.min(), hole_cols.max() + 1)

    img_cols = st.columns(2)
    
    with img_cols[0]: 
            st.pyplot(
                plot_matrix_heatmap(base_wall, "1. 基准板相对墙体几何尺度 (真实材质表面)", cmap="Blues", hole_coords=base_hole_coords, crop_bounds=base_crop_bounds, texture_path=WALL_TEXTURE_PATH, texture_scale=texture_scale), 
                use_container_width=True
            )
            
    with img_cols[1]: 
            # 🛠️ 修改位置：选定基准板情况的第二张矩阵（真实试验开裂模式图）增加 hole_facecolor='white'，强制移至顶层渲染
            st.pyplot(
                plot_matrix_heatmap(base_crack_gt, "2. 真实试验开裂模式图", cmap="gray_r", hole_coords=base_hole_coords, crop_bounds=base_crop_bounds, dilate_first=False, hole_facecolor='white'), 
                use_container_width=True
            )

except Exception as e:
    st.error(f"无法完整追溯基准板 [{selected_base_id}] 的自检实体矩阵文件: {str(e)}")
    base_load_gt = 0.0

# ==================== 6. 核心推理与级联因果后处理中枢 ====================
st.markdown("---")
st.subheader("开始通过神经网络进行推演")

#ui_skeleton_on = st.checkbox("开启独立后处理：对终极开裂模式图进行1像素细化展示", value=True, disabled=not SKIMAGE_AVAILABLE)

current_inputs_hash = (wall_len, wall_hit, wall_layout, has_hole, hole_len, hole_hit, hole_x, hole_y, b_top, b_bottom, b_left, b_right, wall_mat, wall_mortar, wall_asphalt, selected_base_id, ui_skeleton_on)
if st.session_state.get('inputs_hash') != current_inputs_hash:
    st.session_state.has_predicted = False

if st.button("启动预测", type="primary", use_container_width=True):
    if not os.path.exists(MODEL_ONE_PATH) or not os.path.exists(MODEL_TWO_PATH) or not os.path.exists(MODEL_THREE_PATH):
        st.error("运行时阻断：未在预设路径检测到完整的全套权重文件（Model 1, 2, 3），请检查 scripts 文件夹！")
    else:
        with st.spinner("级联物理因果逻辑链闭环演算中..."):
            try:
                BC_MAP = {'BI': [1.0, 1.0], 'SS': [1.0, 0.0], 'FE': [0.0, 0.0]}
                CONSTRUCT_MAP = {'单叶墙': [1.0, 0.0], '空腔墙(50mm空腔)': [0.0, 1.0]}
                MATERIAL_MAP  = {'A级工程砖': [1.0, 0.0, 0.0], 'B级面砖': [0.0, 1.0, 0.0], '密实混凝土': [0.0, 0.0, 1.0]}
                MORTAR_MAP    = {'1:01:06': [1.0, 0.0, 0.0], '1:01:16': [0.0, 1.0, 0.0], '1:1/2:4': [0.0, 0.0, 1.0]}
                ASPHALT_MAP   = {'有': [1.0, 0.0], '无': [0.0, 1.0]}
                
                rel_features = [
                    wall_hit / wall_len,
                    hole_len / wall_len if has_hole else 0.0,
                    hole_hit / wall_hit if has_hole else 0.0,
                    hole_x / wall_len if has_hole else 0.0,
                    hole_y / wall_hit if has_hole else 0.0
                ]
                bc_features = []
                for b_val in [b_top, b_bottom, b_left, b_right]:
                    bc_features.extend(BC_MAP[b_val])
                one_hot = []
                one_hot.extend(CONSTRUCT_MAP[wall_layout])
                one_hot.extend(MATERIAL_MAP[wall_mat])
                one_hot.extend(MORTAR_MAP[wall_mortar])
                one_hot.extend(ASPHALT_MAP[wall_asphalt])
                norm_thick = [float(wall_thick) / 180.0]
                
                scalar_vector = np.array(rel_features + bc_features + one_hot + norm_thick, dtype=np.float32)
                scalar_tensor = torch.from_numpy(scalar_vector).unsqueeze(0) 
                
                spatial_array = np.stack([pred_wall_mask, pred_boundary_mask], axis=0).astype(np.float32)
                spatial_tensor = torch.from_numpy(spatial_array).unsqueeze(0) 
                
                base_id_int = base_name_to_id[selected_base_id]
                base_id_tensor = torch.tensor([base_id_int], dtype=torch.long)
                
                with torch.no_grad():
                    crack_logits = model_one(base_id_tensor, scalar_tensor, spatial_tensor)
                    pred_crack_prob = torch.sigmoid(crack_logits).squeeze(0).numpy()
                    
                    pred_crack_mask_tensor = torch.sigmoid(crack_logits)
                    predicted_load_val = model_two(scalar_tensor, spatial_tensor, pred_crack_mask_tensor).item()
                
                pred_crack_binary = (pred_crack_prob > 0.5).astype(np.float32)
                if np.any(pred_crack_binary):
                    pred_crack_binary = repair_crack_connectivity(pred_crack_binary, max_gap=2)
                    
                    if ui_skeleton_on and SKIMAGE_AVAILABLE:
                        pred_crack_binary = skeletonize(pred_crack_binary.astype(bool)).astype(np.float32)
                    
                    if has_hole:
                        try:
                            b_match_row = df[(df['待预测板编号'] == selected_base_id) & (df['基准板编号'] == selected_base_id)].iloc[0]
                            b_npz_path_raw = b_match_row['矩阵压缩包绝对路径']
                            b_npz_filename = os.path.basename(b_npz_path_raw.replace('\\', '/'))
                            actual_b_npz_path = os.path.join(DATASET_DIR, "processed_data", b_npz_filename)
                            b_npz_data = np.load(actual_b_npz_path)
                            runtime_base_crack_gt = b_npz_data['crack_mask']
                            pred_crack_binary = imitate_benchmark_crack_placement(pred_crack_binary, runtime_base_crack_gt, pred_boundary_mask, max_gap=6)
                        except Exception as b_err:
                            st.caption(f"基准克隆模块静默：采用原生摆放 ({str(b_err)})")
                    
                    if has_hole:
                        pred_crack_binary = bridge_directional_extension_segments(pred_crack_binary, pred_boundary_mask, max_gap=22)
                    
                    pred_crack_binary = bridge_internal_broken_cracks(pred_crack_binary, max_gap=4)
                    pred_crack_binary = bridge_cracks_to_wall_boundary(pred_crack_binary, pred_crop_bounds, max_gap=12)
                
                if has_hole and pred_hole_coords is not None:
                    rh_start, rh_end, ch_start, ch_end = pred_hole_coords
                    pred_crack_binary[rh_start:rh_end, ch_start:ch_end] = 0.0
                
                boundary_seeds = get_boundary_crack_seeds(pred_crack_binary, pred_boundary_mask, pred_crop_bounds, has_hole)
                st.session_state.boundary_seeds = boundary_seeds
                st.session_state.evolution_frames = generate_crack_evolution_frames(pred_crack_binary, boundary_seeds, num_frames=18)

                st.session_state.predicted_load_val = predicted_load_val
                st.session_state.pred_crack_prob = pred_crack_prob
                st.session_state.pred_crack_binary = pred_crack_binary
                st.session_state.base_load_gt = base_load_gt  
                st.session_state.inputs_hash = current_inputs_hash
                
                step3_npz_filename = f"{selected_base_id}_step3_input.npz"
                step3_npz_path = os.path.join(STEP3_DATA_DIR, step3_npz_filename)
                
                if os.path.exists(step3_npz_path):
                    st.session_state.step3_available = True
                    s3_data = np.load(step3_npz_path)
                    disp_tensor_np = s3_data['displacement_tensor']
                    
                    disp_tensor = torch.tensor(disp_tensor_np, dtype=torch.float32).unsqueeze(0)
                    mask_tensor = torch.tensor(s3_data['sensor_mask'], dtype=torch.float32).unsqueeze(0)
                    static_tensor = torch.tensor(s3_data['base_scalars'], dtype=torch.float32).unsqueeze(0)
                    lengths_tensor = torch.tensor([disp_tensor.size(1)], dtype=torch.long)
                    
                    st.session_state.f_true = float(s3_data['f_load_gt'])
                    st.session_state.p_true = float(s3_data['p_load_gt'])
                    
                    with torch.no_grad():
                        pred_fp = model_three(disp_tensor, mask_tensor, static_tensor, lengths_tensor).squeeze(0).numpy()
                    st.session_state.pred_f_val = pred_fp[0]
                    st.session_state.pred_p_val = pred_fp[1]
                else:
                    st.session_state.step3_available = False
                
                st.session_state.has_predicted = True
                st.session_state.trigger_animation = True
                st.balloons()
                st.rerun()  
                
            except Exception as e:
                st.error(f"运行时推理机制中断，原因: {str(e)}")

# ==================== 7. 常驻状态图谱与动画回放渲染流 ====================
if st.session_state.get('has_predicted', False):
    
    m_c1, m_c2, m_c3 = st.columns(3)
    p_load = st.session_state.predicted_load_val
    b_load = st.session_state.base_load_gt
    load_delta = p_load - b_load if b_load > 0 else 0.0
    st_delta_str = f"环境增量: {load_delta:+.2f} kN/m²" if b_load > 0 else "基准未激活"
    
    with m_c1: st.metric(label="预测新墙破坏荷载", value=f"{p_load:.3f} kN/m²", delta=st_delta_str)
    with m_c2: st.metric(label="基准板的破坏荷载", value=f"{b_load:.2f} kN/m²" if b_load > 0 else "未知")
    
    st.markdown("---")
    
    if st.session_state.step3_available:
        fp_cols = st.columns(4)
        pf = st.session_state.pred_f_val
        tf = st.session_state.f_true
        pp = st.session_state.pred_p_val
        tp = st.session_state.p_true
        
        with fp_cols[0]: st.metric(label="A1. 预测 F 点破坏荷载", value=f"{pf:.2f} kN/m²", delta=f"误差: {pf - tf:+.2f}" if tf > 0 else None, delta_color="inverse")
        with fp_cols[1]: st.metric(label="A2. 试验设计的 F 点荷载", value=f"{tf:.2f} kN/m²")
        with fp_cols[2]: st.metric(label="B1. 预测 P 点破坏荷载", value=f"{pp:.2f} kN/m²", delta=f"误差: {pp - tp:+.2f}" if tp > 0 else None, delta_color="inverse")
        with fp_cols[3]: st.metric(label="B2. 试验真实的 P 点荷载", value=f"{tp:.2f} kN/m²")
    else:
        st.error(f"预测未触发：本地未检测到当前基准板的时序特征压缩文件。")
        st.code(f"系统正在寻觅的路径为：\n{os.path.join(STEP3_DATA_DIR, f'{selected_base_id}_step3_input.npz')}")

    st.markdown("---")
    st.write("")
    
    res_img_cols = st.columns(2)
    
    with res_img_cols[0]: 
        st.pyplot(plot_matrix_heatmap(st.session_state.pred_crack_prob, "A. 预测连续开裂演化概率场", cmap="gray_r", hole_coords=pred_hole_coords, crop_bounds=pred_crop_bounds, dilate_first=True))
    
    with res_img_cols[1]: 
        st.markdown("<h5 style='text-align: center; margin-bottom: -10px;'>B. 开裂模式动态演化动画沙盘</h5>", unsafe_allow_html=True)
        anim_placeholder = st.empty()
        progress_bar = st.progress(0)
        replay_triggered = st.button("重新播放动画", use_container_width=True)

    total_frames = len(st.session_state.evolution_frames)
        
    if replay_triggered or st.session_state.get('trigger_animation', False):
        st.session_state.trigger_animation = False 
        
        for idx, frame_matrix in enumerate(st.session_state.evolution_frames):
            if st.session_state.get('step3_available', False):
                p_val = st.session_state.pred_p_val  
                fail_load = st.session_state.predicted_load_val  
                current_load = p_val + (fail_load - p_val) * (idx / (total_frames - 1))
                status_text = "外部横向匀布荷载加载中"
            else:
                max_load = st.session_state.predicted_load_val
                current_load = max_load * ((idx + 1) / total_frames)
                status_text = "预测的破坏荷载加载中"
            
            fig_frame = plot_matrix_3d_voxels(
                frame_matrix, 
                title=f"3D损伤演化模拟 - 进度: {int((idx+1)/total_frames*100)}%\n{status_text}: {current_load:.2f} kN/m$^2$", 
                wall_mask=pred_wall_mask,
                wall_len=wall_len,
                wall_hit=wall_hit,
                wall_thick=wall_thick,
                crop_bounds=pred_crop_bounds
            )
            anim_placeholder.pyplot(fig_frame)
            plt.close(fig_frame)  
            progress_bar.progress((idx + 1) / total_frames)
            time.sleep(0.01)      
        st.rerun() 
        
    else:
        fig_static = plot_matrix_3d_voxels(
            st.session_state.evolution_frames[-1], 
            title="3D损伤裂缝最终贯通破坏形态", 
            wall_mask=pred_wall_mask,
            wall_len=wall_len,
            wall_hit=wall_hit,
            wall_thick=wall_thick,
            crop_bounds=pred_crop_bounds
        )
        anim_placeholder.pyplot(fig_static)
        plt.close(fig_static)
        progress_bar.progress(100)
        
        final_info = "**回放完毕**"
        st.info(final_info)
