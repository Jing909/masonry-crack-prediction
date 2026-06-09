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

# ==================== 全局无差别纯黑直角工业风 CSS 注入 ====================
st.markdown(
    """
    <style>
    /* 1. 彻底斩断折叠面板原生的四周外框、阴影和圆角，将其软化为纯透明底色 */
    div[data-testid="stExpander"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0px !important;
        border-bottom: 1px solid #d3d3d3 !important; 
        margin-bottom: 10px !important;
        padding-bottom: 6px !important;
    }

    /* 2. 强制抹除鼠标悬浮在标题栏上时突突出出来的原生灰色背景槽 */
    div[data-testid="stExpander"] > details > summary:hover {
        background-color: transparent !important;
    }

    /* 3. 消除点击展开后，内部表单/滑动条内容区域的多余边框与两侧缩进 */
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

# ==================== 全局无差别纯黑直角工业风 CSS 注入 ====================
st.markdown(
    """
    <style>
    /* 1. 彻底斩断折叠面板原生的四周外框、阴影和圆角，将其软化为纯透明底色 */
    div[data-testid="stExpander"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0px !important;
        border-bottom: 1px solid #d3d3d3 !important; 
        margin-bottom: 10px !important;
        padding-bottom: 6px !important;
    }

    /* 2. 强制抹除鼠标悬浮在标题栏上时突突出出来的原生灰色背景槽 */
    div[data-testid="stExpander"] > details > summary:hover {
        background-color: transparent !important;
    }

    /* 3. 消除点击展开后，内部表单/滑动条内容区域的多余边框与两侧缩进 */
    div[data-testid="stExpander"] > div {
        border: none !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
    }

    /* 🔥 4. 新增：解除 Metric 组件文本的截断锁定，允许其在空间不足时自动完整换行显示 */
    div[data-testid="stMetricLabel"] > div {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
    }
    div[data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================== 0. 基础配置与中文字体修复 ====================
st.set_page_config(page_title="砌体墙双板交互比对破坏预测系统", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib.font_manager as fm

custom_font_path = os.path.join(BASE_DIR, "SimHei.ttf")
GLOBAL_FONT_PROP = None

if os.path.exists(custom_font_path):
    try:
        fm.fontManager.addfont(custom_font_path)
        GLOBAL_FONT_PROP = fm.FontProperties(fname=custom_font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'Arial Unicode MS']
        st.toast("已成功加载仓库自定义 SimHei 字体库")
    except Exception as e:
        pass

if GLOBAL_FONT_PROP is None:
    linux_wqy_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if os.path.exists(linux_wqy_path):
        GLOBAL_FONT_PROP = fm.FontProperties(fname=linux_wqy_path)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False

# 路径资产精准自适应锚定
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
    return mask.astype(np.float32)

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

# ==================== 🛠️ 核心融合重构的 plot_matrix_heatmap ====================
def plot_matrix_heatmap(matrix, title, cmap="gray", vmin=0, vmax=1, hole_coords=None, crop_bounds=None, texture_path=None, texture_scale=1.0, dilate_first=False, hole_facecolor='none', use_mst=False, run_skeleton=True):
    edge_color = 'red' if cmap == "gray_r" else '#444444'
    line_width = 2.0 if cmap == "gray_r" else 1.0

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
            if GLOBAL_FONT_PROP:
                ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=10, pad=8)
            else:
                ax.set_title(title, fontsize=10, pad=8)
            ax.axis('off')
            
            rect_outer = patches.Rectangle((0, 0), w_c, h_c, linewidth=1.0, edgecolor='#444444', facecolor='none')
            ax.add_patch(rect_outer)
            
            if hole_coords is not None:
                r_start_h, r_end_h, c_start_h, c_end_h = hole_coords
                rect = patches.Rectangle((c_start_h, r_start_h), c_end_h - c_start_h, r_end_h - r_start_h, linewidth=1.0, edgecolor='#444444', facecolor='none', zorder=10)
                ax.add_patch(rect)
            return fig

    # 🚀【精准矢量中点连线引擎】完全同步自 app.py 满血版本
    if cmap == "gray_r":
        binary = (matrix > 0.5).astype(np.uint8)  
        
        if dilate_first and np.any(binary):
            kernel = np.ones((3, 3), dtype=np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
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
        
        # 1. 执行连通性保护骨架化
        if run_skeleton and SKIMAGE_AVAILABLE and np.any(binary_cropped):
            skel = skeletonize(binary_cropped.astype(bool)).astype(np.uint8)
        else:
            skel = binary_cropped.astype(np.uint8)
            
        H_c, W_c = skel.shape
        
        # 定义精准的外墙四周绝对边界网格集合
        wall_boundaries = set()
        for c in range(W_c):
            wall_boundaries.add((0, c))
            wall_boundaries.add((H_c - 1, c))
        for r in range(H_c):
            wall_boundaries.add((r, 0))
            wall_boundaries.add((r, W_c - 1))

        # 定义精准的洞口墙体侧紧邻网格集合
        hole_boundaries = set()
        if hole_coords is not None:
            hr_s, hr_e, hc_s, hc_e = hole_coords
            for r in range(hr_s, hr_e):
                if 0 <= hc_s - 1 < W_c: hole_boundaries.add((r, hc_s - 1)) # 洞左墙面
                if 0 <= hc_e < W_c:     hole_boundaries.add((r, hc_e))     # 洞右墙面
            for c in range(hc_s, hc_e):
                if 0 <= hr_s - 1 < H_c: hole_boundaries.add((hr_s - 1, c)) # 洞上墙面
                if 0 <= hr_e < H_c:     hole_boundaries.add((hr_e, c))     # 洞下墙面

        # ==================== 🧱 🛡️ 步骤一：初始无效纯悬空碎片裂缝初筛 ====================
        if np.any(skel):
            num_labels, labels = cv2.connectedComponents(skel, connectivity=8)
            filtered_skel = np.zeros_like(skel)
            for label_idx in range(1, num_labels):
                comp_mask = (labels == label_idx).astype(np.uint8)
                component_pts = np.argwhere(comp_mask == 1)
                pts_set = set(tuple(p) for p in component_pts)
                
                touches_wall_initially = any(p in wall_boundaries for p in pts_set)
                touches_hole_initially = any(p in hole_boundaries for p in pts_set) if hole_coords is not None else False
                
                if hole_coords is not None:
                    if not touches_wall_initially and not touches_hole_initially and len(pts_set) < 3:
                        continue
                else:
                    if not touches_wall_initially and len(pts_set) < 3:
                        continue
                filtered_skel[comp_mask == 1] = 1
            skel = filtered_skel

        # ==================== 🗺️ 步骤二：满足力学定义的零死胡同全通路强行扩展延伸 ====================
        if np.any(skel):
            num_labels, labels = cv2.connectedComponents(skel, connectivity=8)
            detect_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
            
            for label_idx in range(1, num_labels):
                comp_mask = (labels == label_idx).astype(np.uint8)
                component_pts = np.argwhere(comp_mask == 1)
                pts_set = set(tuple(p) for p in component_pts)
                
                neighbor_count = cv2.filter2D(comp_mask, -1, detect_kernel, borderType=cv2.BORDER_CONSTANT)
                endpoints = [tuple(p) for p in np.argwhere((comp_mask == 1) & (neighbor_count <= 1))]
                
                if len(endpoints) == 0:
                    endpoints = list(pts_set)
                
                if hole_coords is not None:
                    touches_hole = any(p in hole_boundaries for p in pts_set)
                    
                    if not touches_hole and hole_boundaries:
                        min_dist = float('inf')
                        best_ep, best_hb = None, None
                        for ep in endpoints:
                            for hb in hole_boundaries:
                                dist = (ep[0] - hb[0])**2 + (ep[1] - hb[1])**2
                                if dist < min_dist:
                                    min_dist = dist
                                    best_ep = ep
                                    best_hb = hb
                        if best_ep and best_hb:
                            cv2.line(skel, (int(best_ep[1]), int(best_ep[0])), (int(best_hb[1]), int(best_hb[0])), 1, 1)
                            if best_ep in endpoints:
                                endpoints.remove(best_ep)
                    
                    for ep in endpoints:
                        if ep in wall_boundaries or ep in hole_boundaries:
                            continue
                        min_dist = float('inf')
                        best_wb = None
                        for wb in wall_boundaries:
                            dist = (ep[0] - wb[0])**2 + (ep[1] - wb[1])**2
                            if dist < min_dist:
                                min_dist = dist
                                best_wb = wb
                        if best_wb:
                            cv2.line(skel, (int(ep[1]), int(ep[0])), (int(best_wb[1]), int(best_wb[0])), 1, 1)
                else:
                    for ep in endpoints:
                        if ep in wall_boundaries:
                            continue
                        min_dist = float('inf')
                        best_wb = None
                        for wb in wall_boundaries:
                            dist = (ep[0] - wb[0])**2 + (ep[1] - wb[1])**2
                            if dist < min_dist:
                                min_dist = dist
                                best_wb = wb
                        if best_wb:
                            cv2.line(skel, (int(ep[1]), int(ep[0])), (int(best_wb[1]), int(best_wb[0])), 1, 1)

        # ==================== 🛠️ 🔥 步骤三：基于非突变矢量白名单的 MST 去小环引擎 ====================
        edges_to_draw = set()          
        isolated_nodes_to_draw = set()  
        
        if np.any(skel):
            num_labels, labels = cv2.connectedComponents(skel, connectivity=8)
            
            for label_idx in range(1, num_labels):
                comp_mask = (labels == label_idx).astype(np.uint8)
                component_pts = [tuple(p) for p in np.argwhere(comp_mask == 1)]
                pts_set = set(component_pts)
                
                N_black_total = len(pts_set)
                loop_threshold = max(3, int(N_black_total * 0.25))  
                
                comp_edges = []
                for r, c in component_pts:
                    for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in pts_set:
                            edge = ((r, c), (nr, nc)) if (r, c) < (nr, nc) else ((nr, nc), (r, c))
                            comp_edges.append(edge)
                comp_edges = list(set(comp_edges))
                
                comp_edges.sort(key=lambda e: (e[0][0]-e[1][0])**2 + (e[0][1]-e[1][1])**2)
                
                parent_uf = {p: p for p in component_pts}
                def find_uf(n):
                    if parent_uf[n] == n: return n
                    parent_uf[n] = find_uf(parent_uf[n])
                    return parent_uf[n]
                def union_uf(n1, n2):
                    r1, r2 = find_uf(n1), find_uf(n2)
                    if r1 != r2:
                        parent_uf[r1] = r2
                        return True
                    return False
                
                mst_edges = []
                cycle_edges = []
                for e in comp_edges:
                    if union_uf(e[0], e[1]):
                        mst_edges.append(e)
                    else:
                        cycle_edges.append(e)
                
                edges_to_draw.update(mst_edges)
                
                mst_adj = {n: [] for n in component_pts}
                for e in mst_edges:
                    mst_adj[e[0]].append(e[1])
                    mst_adj[e[1]].append(e[0])
                
                for e in cycle_edges:
                    u, v = e[0], e[1]
                    queue = [(u, 1)]
                    visited = {u}
                    cycle_len = 0
                    while queue:
                        curr, dist = queue.pop(0)
                        if curr == v:
                            cycle_len = dist
                            break
                        for nxt in mst_adj[curr]:
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append((nxt, dist + 1))
                    
                    if cycle_len > loop_threshold:
                        edges_to_draw.add(e)
                        
                if len(component_pts) == 1:
                    isolated_nodes_to_draw.add(component_pts[0])
            
        ax.set_xlim(0, w_c)
        ax.set_ylim(h_c, 0)  
        
        ax.set_facecolor("white")  
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

        if GLOBAL_FONT_PROP:
            ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=10, pad=8)
        else:
            ax.set_title(title, fontsize=10, pad=8)
        
        def get_stretched_coords(r_idx, c_idx):
            x_plot = c_idx + 0.5
            y_plot = r_idx + 0.5
            if c_idx == 0: x_plot = 0.0
            elif c_idx == W_c - 1: x_plot = float(W_c)
            if r_idx == 0: y_plot = 0.0
            elif r_idx == H_c - 1: y_plot = float(H_c)
            if hole_coords is not None:
                hr_s, hr_e, hc_s, hc_e = hole_coords
                if hr_s <= r_idx < hr_e:
                    if c_idx == hc_s - 1: x_plot = float(hc_s) - 0.5  
                    elif c_idx == hc_e: x_plot = float(hc_e) + 0.5    
                if hc_s <= c_idx < hc_e:
                    if r_idx == hr_s - 1: y_plot = float(hr_s) - 0.5  
                    elif r_idx == hr_e: y_plot = float(hr_e) + 0.5    
            return x_plot, y_plot
        
        drawn_nodes = set()
        for e in edges_to_draw:
            u, v = e[0], e[1]
            x1, y1 = get_stretched_coords(u[0], u[1])
            x2, y2 = get_stretched_coords(v[0], v[1])
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.2, solid_capstyle='round')
            drawn_nodes.add(u)
            drawn_nodes.add(v)
            
        for pt in isolated_nodes_to_draw:
            if pt not in drawn_nodes:
                x, y = get_stretched_coords(pt[0], pt[1])
                ax.plot(x, y, marker='o', color='black', markersize=1.5)
                    
        if hole_coords is not None:
            r_start_h, r_end_h, c_start_h, c_end_h = hole_coords
            rect = patches.Rectangle(
                (c_start_h - 0.5, r_start_h - 0.5),            
                (c_end_h - c_start_h) + 1.0,               
                (r_end_h - r_start_h) + 1.0,               
                linewidth=1.0,                 
                edgecolor='red',               
                facecolor=hole_facecolor,
                zorder=10
            )
            ax.add_patch(rect)
            
        ax.set_aspect('equal')
        return fig

    if crop_bounds is not None:
        r_start, r_end, c_start, c_end = crop_bounds
        matrix = matrix[r_start:r_end, c_start:c_end]
        if hole_coords is not None:
            hr_start, hr_end, hc_start, hc_end = hole_coords
            hole_coords = (hr_start - r_start, hr_end - r_start, hc_start - c_start, hc_end - c_start)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, square=True)
    if GLOBAL_FONT_PROP:
        ax.set_title(title, fontproperties=GLOBAL_FONT_PROP, fontsize=10, pad=8)
    else:
        ax.set_title(title, fontsize=10, pad=8)
    
    if hole_coords is not None:
        r_start, r_end, c_start, c_end = hole_coords
        rect = patches.Rectangle(
            (c_start, r_start),            
            c_end - c_start,               
            r_end - r_start,               
            linewidth=2.5,                 
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

# ==================== 4. 页面头部区域 ====================
st.markdown("<h2 style='text-align: center; font-weight: 700;'>砌体墙双板交互比对破坏预测系统</h2>", unsafe_allow_html=True)
st.markdown("---")

# ==================== 5. 侧边栏配置栏 ====================
with st.sidebar:
    st.markdown("### 待预测新构件参数")
    
    with st.expander("几何尺寸配置", expanded=True):
        wall_len = st.number_input("墙体长度 L (mm)", value=5615)
        wall_hit = st.number_input("墙体高度 H (mm)", value=2475)
        
        wall_layout = st.selectbox("墙体构造", options=['单叶墙', '空腔墙(50mm空腔)'], index=0)
        wall_thick = 65 if wall_layout == '单叶墙' else 180
        st.caption(f"自动锚定实体墙厚为: **{wall_thick} mm**")
        
        has_hole = st.checkbox("墙体包含开洞", value=False)
        if has_hole:
            hole_len = st.number_input("开洞长度 (mm)", value=2260)
            hole_hit = st.number_input("开洞高度 (mm)", value=1125)
            hole_x = st.number_input("洞口距左边界 (mm)", value=1677)
            hole_y = st.number_input("洞口距下边界 (mm)", value=900)
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
        wall_asphalt = st.selectbox("是否存在沥青层", options=['无', '有'], index=0)

    st.markdown("---")
    st.markdown("### 对照实验锚定")
    selected_base_id = st.selectbox("基准板数据库", options=list(base_name_to_id.keys()), index=0)

texture_scale = 0.10
SCALE = L_max / 32.0
pred_wall_mask, pred_boundary_mask, pred_hole_coords, pred_crop_bounds = generate_spatial_masks(wall_len, wall_hit, has_hole, hole_len, hole_hit, hole_x, hole_y, SCALE)

# ==================== 6. 主画布与推理执行控制（标题与按钮同行对齐） ====================
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
except Exception as e:
    st.error(f"无法完整追溯基准板 [{selected_base_id}] 的数据: {str(e)}")
    base_load_gt = 0.0

col_step1_title, col_step1_btn = st.columns([4, 1])

with col_step1_title:
    st.markdown("### 板子形态展示")

current_inputs_hash = (wall_len, wall_hit, wall_layout, has_hole, hole_len, hole_hit, hole_x, hole_y, b_top, b_bottom, b_left, b_right, wall_mat, wall_mortar, wall_asphalt, selected_base_id)
if st.session_state.get('inputs_hash') != current_inputs_hash:
    st.session_state.has_predicted = False

with col_step1_btn:
    start_prediction = st.button("开始预测", type="primary", use_container_width=True)

col_target, col_benchmark = st.columns(2)

with col_target:
    st.markdown("#### 待预测新构件形态")
    st.caption(f"规格: {wall_len}x{wall_hit}x{wall_thick}mm | 状态: {'开洞' if has_hole else '未开洞'}")
    
    sub_t1, sub_t2 = st.columns(2)
    with sub_t1:
        fig_pred_geometry = plot_matrix_heatmap(pred_wall_mask, "待预测新构件相对墙体空间特征", cmap="Blues", hole_coords=pred_hole_coords, crop_bounds=pred_crop_bounds, texture_path=WALL_TEXTURE_PATH, texture_scale=texture_scale)
        st.pyplot(fig_pred_geometry, use_container_width=True)
    with sub_t2:
        st.markdown(f"""
        **构件力学参数一览：**
        * **力学约束边界**: 
          上 `[{b_top}]` · 下 `[{b_bottom}]` 
          左 `[{b_left}]` · 右 `[{b_right}]`
        * **材质物理属性**: {wall_mat}
        * **设计砂浆比例**: `[{wall_mortar}]`
        * **防潮层处理工艺**: `[{wall_asphalt}]`
        """)

with col_benchmark:
    st.markdown(f"#### 选定基准板")
    st.caption(f"基准板编号: {selected_base_id} | 试验破坏荷载: {base_load_gt:.2f} kN/m²")
    
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        fig_base_geom = plot_matrix_heatmap(base_wall, "基准板相对墙体空间特征", cmap="Blues", hole_coords=base_hole_coords, crop_bounds=base_crop_bounds, texture_path=WALL_TEXTURE_PATH, texture_scale=texture_scale)
        st.pyplot(fig_base_geom, use_container_width=True)
    with sub_c2:
        # 🧱 🛡️ 🔥 完全同步自 app.py：执行降采样后的对角线孤立间隙修护引擎，并传入全新矢量引擎参数
        base_crack_repaired = repair_crack_connectivity(base_crack_gt, max_gap=2)
        
        fig_base_crack = plot_matrix_heatmap(
            base_crack_repaired, 
            "真实试验的开裂模式图", 
            cmap="gray_r", 
            hole_coords=base_hole_coords, 
            crop_bounds=base_crop_bounds, 
            dilate_first=False, 
            hole_facecolor='white', 
            run_skeleton=True            # 🔥 替换成 app.py 专属的连通性骨架保护引擎开关！
        )
        st.pyplot(fig_base_crack, use_container_width=True)


# ==================== 7. 推理与级联计算核心逻辑 ====================
if start_prediction:
    if not os.path.exists(MODEL_ONE_PATH) or not os.path.exists(MODEL_TWO_PATH) or not os.path.exists(MODEL_THREE_PATH):
        st.error("运行时阻断：预设路径未检测到完整的全套权重文件（Model 1, 2, 3）。")
    else:
        with st.spinner("推演中..."):
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
                            st.caption(f"基准克隆回归原生摆放")
                    
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
                st.error(f"运行时推理机制中断: {str(e)}")

# ==================== 8. 全面整合的 Tab 可戏化看板 ====================
if st.session_state.get('step3_available', False):
            pf = st.session_state.pred_f_val
            tf = st.session_state.f_true
            pp = st.session_state.pred_p_val
            tp = st.session_state.p_true
            
            # 🚀 核心改动：用一个外层的 columns(2) 强行把 F点 和 P点 拉到同一行
            analysis_row = st.columns(2)
            
            # ==================== 左半边：F 点荷载比对分析 ====================
            with analysis_row[0]:
                st.markdown("#### 🔹 F 点荷载比对分析")
                f_cols = st.columns(2)  # 嵌套内层 2 列
                with f_cols[0]: 
                    st.metric(
                        label="A1. 预测 F 点荷载", 
                        value=f"{pf:.2f} kN/m²", 
                        delta=f"绝对误差: {pf - tf:+.2f} kN/m²" if tf > 0 else None, 
                        delta_color="inverse"
                    )
                    if tf > 0:
                        st.caption("💡 误差基准：相对基准板理论 F 点")
                with f_cols[1]: 
                    st.metric(label="A2. 基准板试验理论 F 点", value=f"{tf:.2f} kN/m²")
            
            # ==================== 右半边：P 点荷载比对分析 ====================
            with analysis_row[1]:
                st.markdown("#### 🔹 P 点荷载比对分析")
                p_cols = st.columns(2)  # 嵌套内层 2 列
                with p_cols[0]: 
                    st.metric(
                        label="B1. 预测 P 点荷载", 
                    value=f"{pp:.2f} kN/m²", 
                        delta=f"绝对误差: {pp - tp:+.2f} kN/m²" if tp > 0 else None, 
                        delta_color="inverse"
                    )
                    if tp > 0:
                        st.caption("💡 误差基准：相对基准板真实 P 点")
                with p_cols[1]: 
                    st.metric(label="B2. 基准板试验真实 P 点", value=f"{tp:.2f} kN/m²")
        else:
            st.warning("时序特征文件缺失（如选择的基准板为SB09，时序特征文件缺失则是正常现象，因为缺乏SB09的破坏试验过程数据）。")

    with tab_prob:
        col_pb1, col_pb2 = st.columns([1.2, 1])
        with col_pb1:
            fig_prob = plot_matrix_heatmap(st.session_state.pred_crack_prob, "神经网络预测的待预测新构件开裂模式图", cmap="gray_r", hole_coords=pred_hole_coords, crop_bounds=pred_crop_bounds, dilate_first=True, run_skeleton=True)
            st.pyplot(fig_prob, use_container_width=True)

    with tab_animation:
        col_ani1, col_ani2 = st.columns([1.5, 1])
        
        with col_ani1:
            anim_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        with col_ani2:
            col_view_title, col_view_btn = st.columns([1.8, 1])
            with col_view_title:
                st.markdown("#### 3D 破坏形态静止视图")
            with col_view_btn:
                replay_triggered = st.button("回放", use_container_width=True)
        
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
                    status_text = "破坏荷载加载中"
                
                fig_frame = plot_matrix_3d_voxels(
                    frame_matrix, 
                    title=f"3D损伤演化模拟 - 推演进度: {int((idx+1)/total_frames*100)}%\n{status_text}: {current_load:.2f} kN/m^2", 
                    wall_mask=pred_wall_mask, wall_len=wall_len, wall_hit=wall_hit, wall_thick=wall_thick, crop_bounds=pred_crop_bounds
                )
                anim_placeholder.pyplot(fig_frame)
                plt.close(fig_frame)  
                progress_bar.progress((idx + 1) / total_frames)
                time.sleep(0.01)      
            st.rerun() 
        else:
            fig_static = plot_matrix_3d_voxels(
                st.session_state.evolution_frames[-1], 
                title="待预测新构件的3D损伤裂缝的开裂模式图", 
                wall_mask=pred_wall_mask, wall_len=wall_len, wall_hit=wall_hit, wall_thick=wall_thick, crop_bounds=pred_crop_bounds
            )
            with col_ani1:
                anim_placeholder.pyplot(fig_static)
                plt.close(fig_static)
                progress_bar.progress(100)
            with col_ani2:
                st.info("损伤模拟回放已完毕。如需再观看，请点击旁边的回放按钮。")
