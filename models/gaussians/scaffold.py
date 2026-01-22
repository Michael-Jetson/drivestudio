from typing import Dict, Optional
from functools import reduce
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

from models.gaussians.basics import k_nearest_sklearn, rasterization

logger = logging.getLogger()


class ScaffoldGaussians(nn.Module):
    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 1.0,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 0,
        num_cams: int = 0,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg if reg is not None else OmegaConf.create()
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.num_cams = int(num_cams)
        self.device = device

        self.step = 0
        self.in_test_set = False

        # scaffold parameters
        self.feat_dim = int(self.ctrl_cfg.get("feat_dim", 32))
        self.n_offsets = int(self.ctrl_cfg.get("n_offsets", 5))
        self.voxel_size = float(self.ctrl_cfg.get("voxel_size", 0.01))
        self.use_feat_bank = bool(self.ctrl_cfg.get("use_feat_bank", False))
        self.appearance_dim = int(self.ctrl_cfg.get("appearance_dim", 0))
        self.appearance_num = int(self.ctrl_cfg.get("appearance_num", 0))
        self.ratio = int(self.ctrl_cfg.get("ratio", 1))
        self.add_opacity_dist = bool(self.ctrl_cfg.get("add_opacity_dist", False))
        self.add_cov_dist = bool(self.ctrl_cfg.get("add_cov_dist", False))
        self.add_color_dist = bool(self.ctrl_cfg.get("add_color_dist", False))
        self.opacity_thresh = float(self.ctrl_cfg.get("opacity_thresh", 0.0))
        self.frustum_culling = bool(self.ctrl_cfg.get("frustum_culling", True))
        self.near_plane = float(self.ctrl_cfg.get("near_plane", 0.0))
        self.far_plane = self.ctrl_cfg.get("far_plane", None)
        self.voxelize_on_cpu = bool(self.ctrl_cfg.get("voxelize_on_cpu", True))
        self.use_knn_scale = bool(self.ctrl_cfg.get("use_knn_scale", False))
        self.use_prefilter = bool(self.ctrl_cfg.get("use_prefilter", True))
        # Scaffold-GS 生长/剪枝相关参数（与官方训练流程对齐）
        self.update_depth = int(self.ctrl_cfg.get("update_depth", 3))
        self.update_init_factor = int(self.ctrl_cfg.get("update_init_factor", 16))
        self.update_hierachy_factor = int(self.ctrl_cfg.get("update_hierachy_factor", 4))
        self.start_stat = int(self.ctrl_cfg.get("start_stat", 500))
        self.update_from = int(self.ctrl_cfg.get("update_from", 1500))
        self.update_interval = int(self.ctrl_cfg.get("update_interval", self.ctrl_cfg.get("refine_interval", 100)))
        self.update_until = int(self.ctrl_cfg.get("update_until", 15000))
        self.success_threshold = float(self.ctrl_cfg.get("success_threshold", 0.8))
        self.densify_grad_thresh = float(
            self.ctrl_cfg.get("densify_grad_threshold", self.ctrl_cfg.get("densify_grad_thresh", 0.0002))
        )
        self.min_opacity = float(self.ctrl_cfg.get("min_opacity", 0.005))

        # init placeholder parameters (will be overwritten in create_from_pcd)
        self._anchor = Parameter(torch.zeros(1, 3, device=self.device))
        self._offset = Parameter(torch.zeros(1, self.n_offsets, 3, device=self.device))
        self._anchor_feat = Parameter(torch.zeros(1, self.feat_dim, device=self.device))
        self._scaling = Parameter(torch.zeros(1, 6, device=self.device))
        # 训练统计：用于生长/剪枝
        self.opacity_accum = None
        self.offset_gradient_accum = None
        self.offset_denom = None
        self.anchor_demon = None
        # 保存上一次前向中的可见性与mask，供后处理统计使用
        self._last_anchor_visible_mask = None
        self._last_opacity_mask = None
        self._last_opacity_raw = None
        self._last_scales = None  # 保存当前步生成的高斯尺度，用于官方式正则

        # feature bank (optional)
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(4, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1),
            ).to(self.device)
        else:
            self.mlp_feature_bank = None

        # mlps for opacity/cov/color
        opacity_in_dim = self.feat_dim + 3 + (1 if self.add_opacity_dist else 0)
        self.mlp_opacity = nn.Sequential(
            nn.Linear(opacity_in_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh(),
        ).to(self.device)

        cov_in_dim = self.feat_dim + 3 + (1 if self.add_cov_dist else 0)
        self.mlp_cov = nn.Sequential(
            nn.Linear(cov_in_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).to(self.device)

        color_in_dim = self.feat_dim + 3 + (1 if self.add_color_dist else 0) + self.appearance_dim
        self.mlp_color = nn.Sequential(
            nn.Linear(color_in_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid(),
        ).to(self.device)

        # 若未显式指定 appearance_num，则默认使用相机数量
        if self.appearance_dim > 0 and self.appearance_num <= 0:
            self.appearance_num = max(self.num_cams, 1)
            if self.num_cams <= 0:
                logger.warning("appearance_dim > 0 但未提供 num_cams，将 appearance_num 设为 1")

        if self.appearance_dim > 0 and self.appearance_num > 0:
            self.embedding_appearance = nn.Embedding(self.appearance_num, self.appearance_dim).to(self.device)
        else:
            self.embedding_appearance = None

    @property
    def num_points(self) -> int:
        return self._anchor.shape[0]

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step

    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        if self._anchor.numel() == 0:
            return None
        if xys_grad.numel() == 0 or radii.numel() == 0:
            return None
        with torch.no_grad():
            # 与 Scaffold-GS 相同的统计/更新调度
            if self.step < self.update_until and self.step > self.start_stat:
                self._training_statis(radii=radii, xys_grad=xys_grad)
                if self.step > self.update_from and self.step % self.update_interval == 0:
                    self._adjust_anchor(optimizer=optimizer)
            elif self.step == self.update_until:
                self.opacity_accum = None
                self.anchor_demon = None
                self.offset_gradient_accum = None
                self.offset_denom = None
        return None

    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        scale_reg = self.reg_cfg.get("scale_reg", None)
        if scale_reg is not None:
            w = scale_reg.get("w", 0.0)
            if w > 0:
                # 对齐 Scaffold-GS：使用生成高斯的体积（prod）做正则
                if self._last_scales is not None and self._last_scales.numel() > 0:
                    loss_dict["scale_reg"] = self._last_scales.prod(dim=1).mean() * w
        return loss_dict

    def _init_training_stats(self) -> None:
        """初始化生长/剪枝所需的统计缓存。"""
        num_anchors = self._anchor.shape[0]
        total_offsets = num_anchors * self.n_offsets
        # 每个 anchor 的访问与不透明度统计 + 每个 offset 的梯度统计
        self.opacity_accum = torch.zeros((num_anchors, 1), device=self.device)
        self.anchor_demon = torch.zeros((num_anchors, 1), device=self.device)
        self.offset_gradient_accum = torch.zeros((total_offsets, 1), device=self.device)
        self.offset_denom = torch.zeros((total_offsets, 1), device=self.device)

    def _training_statis(self, radii: torch.Tensor, xys_grad: torch.Tensor) -> None:
        """统计 opacity 与 2D 梯度，用于后续生长/剪枝判定。"""
        if (
            self._last_anchor_visible_mask is None
            or self._last_opacity_mask is None
            or self._last_opacity_raw is None
        ):
            return
        if self.opacity_accum is None or self.anchor_demon is None:
            self._init_training_stats()

        # 将“可见 anchor”和“可用 offset”映射回全局 offset 统计
        anchor_visible_mask = self._last_anchor_visible_mask
        offset_selection_mask = self._last_opacity_mask
        opacity_raw = self._last_opacity_raw

        if anchor_visible_mask.sum().item() == 0 or offset_selection_mask.numel() == 0:
            return

        temp_opacity = opacity_raw.view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view(-1, self.n_offsets)
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        update_filter = (radii > 0).view(-1)
        if update_filter.numel() == 0:
            return

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat(1, self.n_offsets).view(-1)
        combined_mask = torch.zeros(
            (self.offset_gradient_accum.shape[0],),
            dtype=torch.bool,
            device=self.device,
        )
        if anchor_visible_mask.sum().item() != offset_selection_mask.shape[0]:
            return
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        if temp_mask.sum().item() != update_filter.shape[0]:
            return
        combined_mask[temp_mask] = update_filter
        if update_filter.sum().item() == 0:
            return

        # 2D 屏幕空间梯度：用于判断哪些 offset 需要生长
        grads = xys_grad.reshape(-1, xys_grad.shape[-1])
        grad_norm = torch.norm(grads[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _cat_tensors_to_optimizer(self, optimizer: torch.optim.Optimizer, tensors_dict: Dict[str, torch.Tensor]) -> Dict[str, Parameter]:
        """扩展优化器中的参数并同步 Adam 动量状态。"""
        # 在不重建 optimizer 的情况下扩展参数与其动量状态
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            name = group.get("name", "")
            if name not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            old_param = group["params"][0]
            extension_tensor = tensors_dict[name].to(device=old_param.device, dtype=old_param.dtype)
            if extension_tensor.numel() == 0:
                optimizable_tensors[name] = old_param
                continue
            stored_state = optimizer.state.get(old_param, None)
            new_param = nn.Parameter(torch.cat((old_param, extension_tensor), dim=0).requires_grad_(True))
            group["params"][0] = new_param
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del optimizer.state[old_param]
                optimizer.state[new_param] = stored_state
            optimizable_tensors[name] = new_param
        return optimizable_tensors

    def _prune_anchor_optimizer(self, optimizer: torch.optim.Optimizer, valid_mask: torch.Tensor) -> Dict[str, Parameter]:
        """按 mask 剪枝 anchor 相关参数，并同步优化器状态。"""
        # 仅剪枝与 anchor 相关的参数组，并同步 optimizer 的 state
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            name = group.get("name", "")
            if not name.startswith(self.class_prefix):
                continue
            component = name.split("#")[-1]
            if component not in {"anchor", "offset", "anchor_feat", "scaling"}:
                continue
            assert len(group["params"]) == 1
            old_param = group["params"][0]
            stored_state = optimizer.state.get(old_param, None)
            new_param = nn.Parameter(old_param[valid_mask].contiguous().requires_grad_(True))
            group["params"][0] = new_param
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                del optimizer.state[old_param]
                optimizer.state[new_param] = stored_state
            optimizable_tensors[component] = new_param
        return optimizable_tensors

    def _prune_anchor(self, optimizer: torch.optim.Optimizer, prune_mask: torch.Tensor) -> None:
        """执行剪枝并更新内部参数引用。"""
        valid_mask = ~prune_mask
        optimizable_tensors = self._prune_anchor_optimizer(optimizer, valid_mask)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]

    def _anchor_growing(
        self,
        optimizer: torch.optim.Optimizer,
        grads: torch.Tensor,
        threshold: float,
        offset_mask: torch.Tensor,
    ) -> None:
        """基于梯度阈值与体素网格生成新 anchor，并写回优化器。"""
        # 按梯度阈值与体素网格生成新 anchor（含去重）
        init_length = self._anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            candidate_mask = grads >= cur_threshold
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self._anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device=self.device)],
                    dim=0,
                )

            all_xyz = self._anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            size_factor = max(size_factor, 1)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self._anchor / cur_size).int()

            selected_xyz = all_xyz.view(-1, 3)[candidate_mask]
            if selected_xyz.numel() == 0:
                continue
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (
                    1 if grid_coords.shape[0] % chunk_size != 0 else 0
                )
                remove_duplicates_list = []
                for j in range(max_iters):
                    cur_chunk = grid_coords[j * chunk_size : (j + 1) * chunk_size, :]
                    if cur_chunk.numel() == 0:
                        continue
                    cur_remove_duplicates = (
                        (selected_grid_coords_unique.unsqueeze(1) == cur_chunk).all(-1).any(-1).view(-1)
                    )
                    remove_duplicates_list.append(cur_remove_duplicates)
                if len(remove_duplicates_list) == 0:
                    remove_duplicates = torch.zeros(
                        (selected_grid_coords_unique.shape[0],), dtype=torch.bool, device=self.device
                    )
                else:
                    remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (
                    (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
                )

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] == 0:
                continue

            new_scaling = torch.ones_like(candidate_anchor).repeat(1, 2).float() * cur_size
            new_scaling = torch.log(new_scaling)

            new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(1, self.n_offsets, 1).float()

            candidate_feat = (
                self._anchor_feat.unsqueeze(dim=1)
                .repeat(1, self.n_offsets, 1)
                .view(-1, self.feat_dim)[candidate_mask]
            )
            new_feat_full = torch.full(
                (selected_grid_coords_unique.shape[0], self.feat_dim),
                -float("inf"),
                device=self.device,
            )
            index = inverse_indices.unsqueeze(1).expand(-1, self.feat_dim)
            # 使用 PyTorch 自带 scatter_reduce_ 代替 torch_scatter 的 scatter_max
            new_feat_full.scatter_reduce_(0, index, candidate_feat, reduce="amax", include_self=True)
            new_feat = new_feat_full[remove_duplicates]
            new_feat[new_feat == -float("inf")] = 0

            new_count = candidate_anchor.shape[0]
            self.anchor_demon = torch.cat(
                [self.anchor_demon, torch.zeros((new_count, 1), device=self.device)], dim=0
            )
            self.opacity_accum = torch.cat(
                [self.opacity_accum, torch.zeros((new_count, 1), device=self.device)], dim=0
            )

            param_dict = {
                self.class_prefix + "anchor": candidate_anchor,
                self.class_prefix + "scaling": new_scaling,
                self.class_prefix + "anchor_feat": new_feat,
                self.class_prefix + "offset": new_offsets,
            }
            optimizable_tensors = self._cat_tensors_to_optimizer(optimizer, param_dict)
            self._anchor = optimizable_tensors[self.class_prefix + "anchor"]
            self._scaling = optimizable_tensors[self.class_prefix + "scaling"]
            self._anchor_feat = optimizable_tensors[self.class_prefix + "anchor_feat"]
            self._offset = optimizable_tensors[self.class_prefix + "offset"]

    def _adjust_anchor(self, optimizer: torch.optim.Optimizer) -> None:
        """统一执行生长 + 剪枝 + 统计重置。"""
        # 生长 + 剪枝 + 统计重置（与 Scaffold-GS 逻辑一致）
        grads = self.offset_gradient_accum / self.offset_denom.clamp(min=1.0)
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > self.update_interval * self.success_threshold * 0.5).squeeze(dim=1)

        self._anchor_growing(optimizer, grads_norm, self.densify_grad_thresh, offset_mask)

        self.offset_denom[offset_mask] = 0
        self.offset_gradient_accum[offset_mask] = 0

        total_offsets = self._anchor.shape[0] * self.n_offsets
        pad_len = total_offsets - self.offset_denom.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, 1), device=self.device)
            self.offset_denom = torch.cat([self.offset_denom, padding], dim=0)
            self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding.clone()], dim=0)

        prune_mask = (self.opacity_accum < self.min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > self.update_interval * self.success_threshold).squeeze(dim=1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)

        if prune_mask.shape[0] > 0:
            offset_denom = self.offset_denom.view(-1, self.n_offsets)[~prune_mask].reshape(-1, 1)
            self.offset_denom = offset_denom
            offset_gradient_accum = self.offset_gradient_accum.view(-1, self.n_offsets)[~prune_mask].reshape(-1, 1)
            self.offset_gradient_accum = offset_gradient_accum

        if anchors_mask.sum().item() > 0:
            num_reset = int(anchors_mask.sum().item())
            self.opacity_accum[anchors_mask] = torch.zeros((num_reset, 1), device=self.device)
            self.anchor_demon[anchors_mask] = torch.zeros((num_reset, 1), device=self.device)

        self.opacity_accum = self.opacity_accum[~prune_mask]
        self.anchor_demon = self.anchor_demon[~prune_mask]

        if prune_mask.sum().item() > 0:
            self._prune_anchor(optimizer, prune_mask)

    def _voxelize_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.voxel_size <= 0:
            return points
        if self.voxelize_on_cpu:
            pts = points.detach().cpu()
            coords = torch.round(pts / self.voxel_size)
            coords = torch.unique(coords, dim=0)
            return (coords * self.voxel_size).to(self.device)
        coords = torch.round(points / self.voxel_size)
        coords = torch.unique(coords, dim=0)
        return coords * self.voxel_size

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: Optional[torch.Tensor] = None) -> None:
        points = init_means
        if self.ratio > 1:
            points = points[:: self.ratio]
        points = points.to(self.device)

        # 自适应 voxel size：参考 Scaffold-GS 的 distCUDA2 逻辑
        if self.voxel_size <= 0 and points.shape[0] > 3:
            distances, _ = k_nearest_sklearn(points.detach(), 3)
            distances = torch.from_numpy(distances).to(self.device)
            mean_dist2 = (distances ** 2).mean(dim=-1)
            kth = max(int(mean_dist2.numel() * 0.5), 1)
            median_dist2 = torch.kthvalue(mean_dist2, kth).values
            self.voxel_size = float(median_dist2.item())

        anchors = self._voxelize_points(points)
        if anchors.shape[0] == 0:
            raise ValueError("No anchors created; check voxel_size or input points.")

        if self.use_knn_scale and anchors.shape[0] > 3:
            # 参考 distCUDA2：使用 3-NN 的均方距离开根号作为尺度
            distances, _ = k_nearest_sklearn(anchors.detach(), 3)
            distances = torch.from_numpy(distances).to(self.device)
            avg_dist = torch.sqrt((distances ** 2).mean(dim=-1).clamp(min=1e-12))
            base_scale = avg_dist.clamp(min=1e-6)
        else:
            base = max(self.voxel_size, 1e-6)
            base_scale = torch.full((anchors.shape[0],), base, device=self.device)

        scales = torch.log(base_scale)[:, None].repeat(1, 6)

        self._anchor = Parameter(anchors)
        self._offset = Parameter(torch.zeros(anchors.shape[0], self.n_offsets, 3, device=self.device))
        self._anchor_feat = Parameter(torch.zeros(anchors.shape[0], self.feat_dim, device=self.device))
        self._scaling = Parameter(scales)
        # 初始化后重置统计量
        self._init_training_stats()

    def _get_anchor_visible_mask(self, cam) -> torch.Tensor:
        if not self.frustum_culling:
            return torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self.device)

        w2c = torch.linalg.inv(cam.camtoworlds)
        xyz_cam = (w2c[:3, :3] @ self._anchor.T + w2c[:3, 3:4]).T
        z = xyz_cam[:, 2]
        valid = z > self.near_plane
        if self.far_plane is not None:
            valid = valid & (z < float(self.far_plane))

        proj = (cam.Ks @ xyz_cam.T).T
        u = proj[:, 0] / (proj[:, 2] + 1e-6)
        v = proj[:, 1] / (proj[:, 2] + 1e-6)
        valid = valid & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        return valid

    def _prefilter_anchor_visible_mask(self, cam) -> torch.Tensor:
        """使用 gsplat 做可见性预过滤（近似 Scaffold-GS 的 visible_filter）。"""
        if not self.frustum_culling:
            return torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self.device)

        # 先做一次粗筛，减少 rasterization 负担
        coarse_mask = self._get_anchor_visible_mask(cam)
        if coarse_mask.sum().item() == 0:
            return coarse_mask

        anchors = self._anchor[coarse_mask]
        scales = self.get_scaling[coarse_mask][:, :3]
        quats = torch.zeros((anchors.shape[0], 4), device=self.device)
        quats[:, 0] = 1.0
        opacities = torch.ones((anchors.shape[0],), device=self.device)
        colors = torch.zeros((anchors.shape[0], 3), device=self.device)

        with torch.no_grad():
            _, _, info = rasterization(
                means=anchors,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],
                Ks=cam.Ks[None, ...],
                width=cam.W,
                height=cam.H,
                near_plane=self.near_plane,
                far_plane=float(self.far_plane) if self.far_plane is not None else 1e10,
                packed=False,
                absgrad=False,
                sparse_grad=False,
                rasterize_mode="classic",
            )
        radii = info["radii"]
        if radii.dim() > 1:
            radii = radii[0]
        visible = radii > 0

        full_mask = torch.zeros_like(coarse_mask)
        full_mask[coarse_mask] = visible
        return full_mask

    def get_gaussians(self, cam) -> Optional[Dict[str, torch.Tensor]]:
        if self._anchor.numel() == 0:
            self._last_anchor_visible_mask = None
            self._last_opacity_mask = None
            self._last_opacity_raw = None
            self._last_scales = None
            return None

        # 预过滤可见 anchor（对齐 Scaffold-GS 的可见性过滤逻辑）
        if self.use_prefilter:
            visible_mask = self._prefilter_anchor_visible_mask(cam)
        else:
            visible_mask = self._get_anchor_visible_mask(cam)
        if visible_mask.sum() == 0:
            self._last_anchor_visible_mask = None
            self._last_opacity_mask = None
            self._last_opacity_raw = None
            self._last_scales = None
            return None

        anchor = self._anchor[visible_mask]
        feat = self._anchor_feat[visible_mask]
        offsets = self._offset[visible_mask]
        scaling = self.get_scaling[visible_mask]

        cam_center = cam.camtoworlds[:3, 3]
        ob_view = anchor - cam_center
        ob_dist = ob_view.norm(dim=1, keepdim=True).clamp(min=1e-6)
        ob_view = ob_view / ob_dist

        if self.use_feat_bank and self.mlp_feature_bank is not None:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1)
            if feat.shape[1] >= 4:
                feat = feat.unsqueeze(dim=-1)
                feat = (
                    feat[:, ::4, :1].repeat(1, 4, 1) * bank_weight[:, :, :1]
                    + feat[:, ::2, :1].repeat(1, 2, 1) * bank_weight[:, :, 1:2]
                    + feat[:, ::1, :1] * bank_weight[:, :, 2:]
                )
                feat = feat.squeeze(dim=-1)
            else:
                feat = feat

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

        appearance = None
        if self.appearance_dim > 0:
            if self.embedding_appearance is not None:
                # 使用相机编号做 per-camera 外观嵌入
                cam_uid = 0
                if hasattr(cam, "uid") and cam.uid is not None:
                    cam_uid = int(cam.uid)
                appearance_idx = torch.full(
                    (anchor.shape[0],),
                    cam_uid,
                    dtype=torch.long,
                    device=self.device,
                )
                appearance = self.embedding_appearance(appearance_idx)
            else:
                appearance = torch.zeros((anchor.shape[0], self.appearance_dim), device=self.device)

        opacity_in = cat_local_view if self.add_opacity_dist else cat_local_view_wodist
        opacity_raw = self.mlp_opacity(opacity_in)
        opacity_raw = opacity_raw.reshape(-1, 1)
        opacity_mask = (opacity_raw > self.opacity_thresh).view(-1)

        if opacity_mask.sum() == 0:
            self._last_anchor_visible_mask = None
            self._last_opacity_mask = None
            self._last_opacity_raw = None
            self._last_scales = None
            return None

        color_in = cat_local_view if self.add_color_dist else cat_local_view_wodist
        if appearance is not None:
            color_in = torch.cat([color_in, appearance], dim=1)
        color = self.mlp_color(color_in)
        color = color.reshape(-1, 3)

        cov_in = cat_local_view if self.add_cov_dist else cat_local_view_wodist
        scale_rot = self.mlp_cov(cov_in)
        scale_rot = scale_rot.reshape(-1, 7)

        offsets = offsets.reshape(-1, 3)
        scaling_repeat = scaling.repeat_interleave(self.n_offsets, dim=0)
        anchor_repeat = anchor.repeat_interleave(self.n_offsets, dim=0)

        # apply opacity mask
        opacity = opacity_raw[opacity_mask]
        color = color[opacity_mask]
        scale_rot = scale_rot[opacity_mask]
        offsets = offsets[opacity_mask]
        scaling_repeat = scaling_repeat[opacity_mask]
        anchor_repeat = anchor_repeat[opacity_mask]

        # build gaussians
        scales = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        quats = self.quat_act(scale_rot[:, 3:7])
        offsets = offsets * scaling_repeat[:, :3]
        means = anchor_repeat + offsets

        # 保存前向的可见性与 mask，供后处理统计使用
        self._last_anchor_visible_mask = visible_mask
        self._last_opacity_mask = opacity_mask
        self._last_opacity_raw = opacity_raw.detach()
        self._last_scales = scales  # 用于官方式尺度正则

        gs_dict = {
            "_means": means,
            "_opacities": opacity,
            "_rgbs": color,
            "_scales": scales,
            "_quats": quats,
        }
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in scaffold gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in scaffold gaussian {k} at step {self.step}")
        return gs_dict

    def get_param_groups(self) -> Dict[str, list]:
        param_groups = {
            self.class_prefix + "anchor": [self._anchor],
            self.class_prefix + "offset": [self._offset],
            self.class_prefix + "anchor_feat": [self._anchor_feat],
            self.class_prefix + "scaling": [self._scaling],
            self.class_prefix + "mlp_opacity": list(self.mlp_opacity.parameters()),
            self.class_prefix + "mlp_cov": list(self.mlp_cov.parameters()),
            self.class_prefix + "mlp_color": list(self.mlp_color.parameters()),
        }
        if self.mlp_feature_bank is not None:
            param_groups[self.class_prefix + "mlp_feature_bank"] = list(self.mlp_feature_bank.parameters())
        if self.embedding_appearance is not None:
            param_groups[self.class_prefix + "embedding_appearance"] = list(self.embedding_appearance.parameters())
        return param_groups

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        if "_anchor" in state_dict:
            self._anchor = Parameter(torch.zeros_like(state_dict["_anchor"], device=self.device))
        if "_offset" in state_dict:
            self._offset = Parameter(torch.zeros_like(state_dict["_offset"], device=self.device))
        if "_anchor_feat" in state_dict:
            self._anchor_feat = Parameter(torch.zeros_like(state_dict["_anchor_feat"], device=self.device))
        if "_scaling" in state_dict:
            self._scaling = Parameter(torch.zeros_like(state_dict["_scaling"], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        if self._anchor is not None and self._anchor.numel() > 0:
            # 载入权重后重置统计量，避免尺寸不一致
            self._init_training_stats()
        return msg
