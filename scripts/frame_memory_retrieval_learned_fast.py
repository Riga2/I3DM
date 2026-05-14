import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.amp import autocast
from einops import rearrange, repeat

from frame_memory_retrieval_optimized import (
    KeyFrameMemoryBankLearnedOccRetrieval,
    project_points_to_camera,
)


class KeyFrameMemoryBankLearnedOccRetrievalFast(KeyFrameMemoryBankLearnedOccRetrieval):
    """
    Fast learned occupancy retrieval.

    The original retrieve_camera_occ_perf_test scores every memory candidate with
    the learned scene decoder. This version keeps the learned coverage selection,
    but first builds a small proposal set with cheap geometric FOV overlap. The
    expensive learned model therefore runs on a bounded candidate pool instead of
    the full memory bank.
    """

    def __init__(
        self,
        *args,
        candidate_pool_size: int = 64,
        proposal_points: int = 8192,
        always_include_recent: int = 8,
        learned_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.candidate_pool_size = candidate_pool_size
        self.proposal_points = proposal_points
        self.always_include_recent = always_include_recent
        self.learned_batch_size = learned_batch_size
        self.proposal_points_local = self.cached_points_local[: min(proposal_points, self.cached_points_local.shape[0])]
        self.last_retrieval_profile: Dict[str, float] = {}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_now(self) -> float:
        self._sync()
        return time.perf_counter()

    def _build_candidate_pool(
        self,
        query_c2w: torch.Tensor,
        query_fxfycxcy: torch.Tensor,
        last_idx: int,
        candidate_pool_size: Optional[int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        profile: Dict[str, float] = {}
        t0 = self._profile_now()

        all_candidate_indices = torch.arange(last_idx, device=self.device, dtype=torch.long)
        if all_candidate_indices.numel() == 0:
            return all_candidate_indices, {"proposal_time": 0.0, "proposal_candidates": 0}

        pool_size = candidate_pool_size or self.candidate_pool_size
        pool_size = min(pool_size, all_candidate_indices.numel())
        if all_candidate_indices.numel() <= pool_size:
            profile["proposal_time"] = self._profile_now() - t0
            profile["proposal_candidates"] = int(all_candidate_indices.numel())
            return all_candidate_indices, profile

        if query_c2w.dim() == 2:
            query_c2w = query_c2w.unsqueeze(0)
            query_fxfycxcy = query_fxfycxcy.unsqueeze(0)

        query_centers = query_c2w[:, :3, 3]
        points_local = self.proposal_points_local
        points_w = (query_centers.unsqueeze(1) + points_local.unsqueeze(0)).reshape(-1, 3)

        _, in_fov_query = project_points_to_camera(
            points_w,
            query_c2w,
            query_fxfycxcy,
            self.H,
            self.W,
        )
        in_fov_query = in_fov_query.any(dim=1)

        if in_fov_query.any():
            points_w = points_w[in_fov_query]
            _, in_fov_all = project_points_to_camera(
                points_w,
                self.c2w_list[:last_idx],
                self.fxfycxcy_list[:last_idx],
                self.H,
                self.W,
            )
            overlap_scores = in_fov_all.sum(dim=0).float()
        else:
            query_center = query_centers.mean(dim=0, keepdim=True)
            mem_centers = self.c2w_list[:last_idx, :3, 3]
            overlap_scores = -torch.linalg.norm(mem_centers - query_center, dim=1)

        if self.always_include_recent > 0:
            recent_start = max(0, last_idx - self.always_include_recent)
            recent_indices = torch.arange(recent_start, last_idx, device=self.device, dtype=torch.long)
        else:
            recent_indices = torch.empty(0, device=self.device, dtype=torch.long)

        boosted_scores = overlap_scores.clone()
        if recent_indices.numel() > 0:
            score_boost = overlap_scores.max() - overlap_scores.min() + 1.0
            boosted_scores[recent_indices] += score_boost
        candidate_indices = torch.topk(boosted_scores, k=pool_size).indices.sort().values

        profile["proposal_time"] = self._profile_now() - t0
        profile["proposal_candidates"] = int(candidate_indices.numel())
        return candidate_indices, profile

    def _calc_learned_confidence_maps(
        self,
        query_c2w: torch.Tensor,
        query_fxfycxcy: torch.Tensor,
        last_idx: int,
        candidate_indices: torch.Tensor,
        batch_size: int,
        disable_checkpoint: bool = True,
    ) -> torch.Tensor:
        confidence_maps = []
        all_rgbs_processed = self.rgb_list_processed
        all_fxfycxcys_processed = self.fxfycxcy_list_processed
        query_fxfycxcy_processed = self.preprocess_intrinsics_only(
            query_fxfycxcy,
            src_H=self.H,
            src_W=self.W,
            target_H=self.target_size[0],
            target_W=self.target_size[1],
        )

        last_c2w = self.c2w_list[last_idx : last_idx + 1]
        for start in range(0, candidate_indices.numel(), batch_size):
            batch_indices = candidate_indices[start : start + batch_size]
            current_batch_size = batch_indices.numel()

            batch_last_c2w = last_c2w.unsqueeze(0).expand(current_batch_size, -1, -1, -1)
            batch_cand_c2w = self.c2w_list[batch_indices].unsqueeze(1)
            batch_query_c2w = query_c2w.unsqueeze(0).expand(current_batch_size, -1, -1, -1)
            batch_union_c2ws = torch.cat([batch_last_c2w, batch_cand_c2w, batch_query_c2w], dim=1)
            batch_union_c2ws_processed = self.preprocess_batch_poses(batch_union_c2ws, scene_scale_factor=1.35)

            indices_last = torch.full(
                (current_batch_size,),
                last_idx,
                device=self.device,
                dtype=torch.long,
            )
            imgs_last = all_rgbs_processed[indices_last]
            imgs_cand = all_rgbs_processed[batch_indices]
            batch_input_images = torch.stack([imgs_last, imgs_cand], dim=1)

            fx_last = all_fxfycxcys_processed[indices_last]
            fx_cand = all_fxfycxcys_processed[batch_indices]
            batch_input_fx = torch.stack([fx_last, fx_cand], dim=1)
            batch_target_fx = query_fxfycxcy_processed.unsqueeze(0).expand(current_batch_size, -1, -1)

            data_input_batch = {
                "image": batch_input_images.to(dtype=self.dtype),
                "c2w": batch_union_c2ws_processed[:, :2].to(dtype=self.dtype),
                "fxfycxcy": batch_input_fx.to(dtype=self.dtype),
            }
            data_target_batch = {
                "c2w": batch_union_c2ws_processed[:, 2:].to(dtype=self.dtype),
                "fxfycxcy": batch_target_fx.to(dtype=self.dtype),
            }

            with torch.inference_mode():
                with autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                    if disable_checkpoint:
                        conf_map = self._calc_conf_batch_no_checkpoint(data_input_batch, data_target_batch).detach().float()
                    else:
                        conf_map = self.nvs_model.calc_conf_batch(data_input_batch, data_target_batch).detach().float()
            if conf_map.dim() == 5:
                conf_map = conf_map.squeeze(2)
            elif conf_map.dim() == 3:
                conf_map = conf_map.unsqueeze(1)
            confidence_maps.append(conf_map)

        return torch.cat(confidence_maps, dim=0)

    def _calc_conf_batch_no_checkpoint(self, input_data_batch, target_data_batch):
        """
        Same math as SceneDecoderOnlyOccAnalysis.calc_conf_batch, but disables
        activation checkpointing during inference. Checkpointing is useful for
        training memory, but it adds avoidable overhead in no-grad retrieval.
        """
        model = self.nvs_model
        im_H, im_W = model.image_size_H, model.image_size_W

        val_input = model.process_data(input_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)
        val_target = model.process_data(target_data_batch, im_H=im_H, im_W=im_W, compute_rays=True)

        posed_input_images = model.get_posed_input(
            images=val_input["image"],
            ray_o=val_input["ray_o"],
            ray_d=val_input["ray_d"],
        )
        b, v_input, _, _, _ = posed_input_images.size()

        input_img_tokens = model.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

        target_pose_cond = model.get_posed_input(
            ray_o=val_target["ray_o"],
            ray_d=val_target["ray_d"],
        )
        _, v_target, _, _, _ = target_pose_cond.size()
        target_pose_tokens = model.target_pose_tokenizer(target_pose_cond)

        repeated_input_img_tokens = repeat(
            input_img_tokens,
            "b np d -> (b v_target) np d",
            v_target=v_target,
        )
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)
        concat_img_tokens = model.transformer_input_layernorm(transformer_input)

        mid_tokens = model.pass_layers(
            concat_img_tokens,
            start_layer=0,
            end_layer=model.exit_layer,
            gradient_checkpoint=False,
            checkpoint_every=model.grad_checkpoint_every,
        )

        _, mid_target_tokens = mid_tokens.split([v_input * n_patches, n_patches], dim=1)
        conf_log_var = model.confidence_head(mid_target_tokens)

        return -rearrange(
            conf_log_var,
            "(b v) (h w) 1 -> b v h w",
            b=b,
            v=v_target,
            h=model.image_size_H // model.patch_size,
            w=model.image_size_W // model.patch_size,
        )

    def _select_diverse_coverage(
        self,
        confidence_maps: torch.Tensor,
        candidate_indices: torch.Tensor,
        num_to_select: int,
    ) -> List[int]:
        if confidence_maps.numel() == 0 or num_to_select <= 0:
            return []

        candidate_count = confidence_maps.shape[0]
        flat_maps = confidence_maps.flatten(1)
        current_best = torch.full_like(flat_maps[0], -float("inf"))
        available = torch.ones(candidate_count, dtype=torch.bool, device=confidence_maps.device)
        selected: List[int] = []

        for _ in range(min(num_to_select, candidate_count)):
            gains = (torch.maximum(flat_maps, current_best.unsqueeze(0)) - current_best.unsqueeze(0)).mean(dim=1)
            gains = gains.masked_fill(~available, -float("inf"))
            best_pos = int(torch.argmax(gains).item())
            if not torch.isfinite(gains[best_pos]) or gains[best_pos].item() <= 0:
                break
            selected.append(int(candidate_indices[best_pos].item()))
            available[best_pos] = False
            current_best = torch.maximum(current_best, flat_maps[best_pos])

        return selected

    def retrieve_camera_occ_fast(
        self,
        query_c2w: torch.Tensor,
        query_fxfycxcy: torch.Tensor,
        k: int = 4,
        batch_size: Optional[int] = None,
        candidate_pool_size: Optional[int] = None,
        return_profile: bool = False,
    ):
        t_total = self._profile_now()
        if self.frames_num <= k:
            top_indices = list(range(self.frames_num))
            self.last_retrieval_profile = {
                "memory_size": self.frames_num,
                "proposal_candidates": self.frames_num,
                "proposal_time": 0.0,
                "learned_time": 0.0,
                "selection_time": 0.0,
                "total_time": self._profile_now() - t_total,
            }
        else:
            last_idx = self.frames_num - 1
            candidate_indices, proposal_profile = self._build_candidate_pool(
                query_c2w=query_c2w,
                query_fxfycxcy=query_fxfycxcy,
                last_idx=last_idx,
                candidate_pool_size=candidate_pool_size,
            )

            t_learned = self._profile_now()
            confidence_maps = self._calc_learned_confidence_maps(
                query_c2w=query_c2w,
                query_fxfycxcy=query_fxfycxcy,
                last_idx=last_idx,
                candidate_indices=candidate_indices,
                batch_size=batch_size or self.learned_batch_size,
                disable_checkpoint=True,
            )
            learned_time = self._profile_now() - t_learned

            t_select = self._profile_now()
            selected = self._select_diverse_coverage(
                confidence_maps=confidence_maps,
                candidate_indices=candidate_indices,
                num_to_select=k - 1,
            )
            selection_time = self._profile_now() - t_select

            top_indices = [last_idx] + selected
            self.last_retrieval_profile = {
                "memory_size": self.frames_num,
                "proposal_candidates": int(candidate_indices.numel()),
                "proposal_time": proposal_profile["proposal_time"],
                "learned_time": learned_time,
                "selection_time": selection_time,
                "total_time": self._profile_now() - t_total,
            }

        while len(top_indices) < k and len(top_indices) > 0:
            top_indices.append(top_indices[-1])

        if return_profile:
            return top_indices, dict(self.last_retrieval_profile)
        return top_indices

    def retrieve_camera_occ_full_optimized(
        self,
        query_c2w: torch.Tensor,
        query_fxfycxcy: torch.Tensor,
        k: int = 4,
        batch_size: Optional[int] = None,
        disable_checkpoint: bool = True,
        return_profile: bool = False,
    ):
        """
        Exact learned retrieval with lower overhead.

        This keeps the original retrieve_camera_occ_perf_test search space: every
        memory frame except the latest frame is scored by the learned occupancy
        model, then the latest frame is prepended. It only optimizes execution
        details: no geometric candidate pruning, no per-candidate Python map
        appends, vectorized coverage selection, no inference-time checkpointing
        by default, and fewer GPU synchronizations.
        """
        t_total = self._profile_now()
        if self.frames_num <= k:
            top_indices = list(range(self.frames_num))
            self.last_retrieval_profile = {
                "memory_size": self.frames_num,
                "proposal_candidates": self.frames_num,
                "proposal_time": 0.0,
                "learned_time": 0.0,
                "selection_time": 0.0,
                "total_time": self._profile_now() - t_total,
            }
        else:
            last_idx = self.frames_num - 1
            candidate_indices = torch.arange(last_idx, device=self.device, dtype=torch.long)

            t_learned = self._profile_now()
            confidence_maps = self._calc_learned_confidence_maps(
                query_c2w=query_c2w,
                query_fxfycxcy=query_fxfycxcy,
                last_idx=last_idx,
                candidate_indices=candidate_indices,
                batch_size=batch_size or self.learned_batch_size,
                disable_checkpoint=disable_checkpoint,
            )
            learned_time = self._profile_now() - t_learned

            t_select = self._profile_now()
            selected = self._select_diverse_coverage(
                confidence_maps=confidence_maps,
                candidate_indices=candidate_indices,
                num_to_select=k - 1,
            )
            selection_time = self._profile_now() - t_select

            top_indices = [last_idx] + selected
            self.last_retrieval_profile = {
                "memory_size": self.frames_num,
                "proposal_candidates": int(candidate_indices.numel()),
                "proposal_time": 0.0,
                "learned_time": learned_time,
                "selection_time": selection_time,
                "total_time": self._profile_now() - t_total,
            }

        while len(top_indices) < k and len(top_indices) > 0:
            top_indices.append(top_indices[-1])

        if return_profile:
            return top_indices, dict(self.last_retrieval_profile)
        return top_indices
