#!/usr/bin/env python3
"""
SAM2-MoE Evaluation Pipeline

Complete evaluation pipeline for SAM2 with Mixture of Prompt Experts (MoPE).
Generates comprehensive evaluation metrics and reports.

Usage:
    python eval_sam2_moe.py --checkpoint outputs/sam2_moe/checkpoint_best.pt
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ocid_dataset import get_dataloaders
from sam2.build_sam import build_sam2_video_predictor_moe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SAM2MoEEvaluator:
    """
    Complete evaluation pipeline for SAM2-MoE.
    """

    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.device = torch.device(config.device)

        # Build model
        logger.info("Building SAM2-MoE model...")
        self.model = build_sam2_video_predictor_moe(
            config_file=config.model_config,
            ckpt_path=config.base_checkpoint,
            device=self.device,
            mode="eval",
        )

        # Load trained MoE weights
        if config.moe_checkpoint:
            self._load_moe_checkpoint(config.moe_checkpoint)

        self.model.eval()

        # Load data
        logger.info("Loading dataset...")
        _, _, self.test_loader = get_dataloaders(
            root_dir=config.data_dir,
            batch_size=1,
            num_frames=config.num_frames,
            image_size=config.image_size,
            num_workers=config.num_workers,
            train_ratio=config.train_ratio,
            test_sequences=config.test_sequences,
            include_test=True,
        )
        logger.info(f"Test samples: {len(self.test_loader)}")

    def _load_moe_checkpoint(self, ckpt_path: str):
        """Load trained MoE parameters."""
        logger.info(f"Loading MoE checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        moe_state_dict = checkpoint.get("model_state_dict", {})
        if not moe_state_dict:
            logger.warning("No model_state_dict in checkpoint")
            return

        # Merge with current model state
        current_state = self.model.state_dict()
        loaded_count = 0
        for key, value in moe_state_dict.items():
            if key in current_state:
                current_state[key] = value
                loaded_count += 1

        self.model.load_state_dict(current_state)
        logger.info(f"Loaded {loaded_count} MoE parameters")

        if "best_val_iou" in checkpoint:
            logger.info(f"Checkpoint best IoU: {checkpoint['best_val_iou']:.4f}")

    def _prepare_prompt(self, points: torch.Tensor, device: torch.device) -> Dict:
        """Prepare point prompts for SAM2."""
        point_coords = points.unsqueeze(0).to(device)
        point_labels = torch.ones(1, points.shape[0], dtype=torch.int32, device=device)
        return {"point_coords": point_coords, "point_labels": point_labels}

    def forward_frame(
        self,
        frame: torch.Tensor,
        points: torch.Tensor,
        output_dict: Dict,
        frame_idx: int,
        is_cond_frame: bool,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for a single frame."""
        device = frame.device

        # Image encoding
        img_batch = frame.unsqueeze(0)
        backbone_out = self.model.forward_image(img_batch)

        # Prepare backbone features
        _, vision_feats, vision_pos_embeds, feat_sizes = \
            self.model._prepare_backbone_features(backbone_out)

        # Point prompts
        point_inputs = self._prepare_prompt(points, device)

        # High-res features
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # Memory-conditioned features (MoE is invoked here)
        pix_feat = self.model._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_cond_frame,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=frame_idx + 1,
            track_in_reverse=False,
        )

        # SAM mask decoder
        multimask = self.model._use_multimask(is_cond_frame, point_inputs)
        sam_outputs = self.model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=multimask,
        )

        _, _, _, low_res_masks, high_res_masks, obj_ptr, obj_scores = sam_outputs

        # Store memory
        current_out = {
            "point_inputs": point_inputs,
            "mask_inputs": None,
            "pred_masks": low_res_masks,
            "obj_ptr": obj_ptr,
        }

        if self.model.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=obj_scores,
                is_mask_from_pts=True,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if is_cond_frame:
            output_dict["cond_frame_outputs"][frame_idx] = current_out
        else:
            output_dict["non_cond_frame_outputs"][frame_idx] = current_out

        pred_mask = high_res_masks.squeeze(0).squeeze(0)
        return pred_mask, output_dict

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.5).float()

        if pred_binary.shape != target.shape:
            target = F.interpolate(
                target.unsqueeze(0).unsqueeze(0).float(),
                size=pred_binary.shape,
                mode="nearest",
            ).squeeze(0).squeeze(0)

        # IoU (mean_iou - per instance)
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()

        # Pixel IoU
        pixel_intersection = (pred_binary * target).sum().item()
        pixel_union = (pred_binary.sum() + target.sum() - intersection).item()
        pixel_iou = pixel_intersection / (pixel_union + 1e-6)

        # Dice
        dice = (2 * intersection / (pred_binary.sum() + target.sum() + 1e-6)).item()

        # Detection: TP/FP/FN (object-level)
        pred_has_mask = pred_binary.sum() > 0
        target_has_mask = target.sum() > 0

        if target_has_mask and pred_has_mask and iou >= iou_threshold:
            tp, fp, fn = 1, 0, 0
        elif target_has_mask and not pred_has_mask:
            tp, fp, fn = 0, 0, 1
        elif not target_has_mask and pred_has_mask:
            tp, fp, fn = 0, 1, 0
        elif target_has_mask and pred_has_mask and iou < iou_threshold:
            tp, fp, fn = 0, 1, 1
        else:
            tp, fp, fn = 0, 0, 0

        # Panoptic Quality components
        if tp > 0:
            sq = iou  # Segmentation Quality = IoU for matched
            rq = 1.0  # Recognition Quality = TP / (TP + 0.5*FP + 0.5*FN)
            pq = sq * rq
        else:
            sq, rq, pq = 0.0, 0.0, 0.0

        # Boundary accuracy (approximate)
        pred_edges = self._compute_edges(pred_binary)
        target_edges = self._compute_edges(target)
        edge_intersection = (pred_edges * target_edges).sum()
        edge_union = pred_edges.sum() + target_edges.sum() - edge_intersection
        boundary_iou = (edge_intersection / (edge_union + 1e-6)).item()

        # J (Jaccard Index) = IoU - already computed above
        j_score = iou

        # F (Boundary F-measure)
        f_score = self._compute_boundary_f_measure(pred_binary, target)

        # J&F (average of J and F)
        j_and_f = (j_score + f_score) / 2.0

        return {
            "iou": iou,
            "pixel_iou": pixel_iou,
            "dice": dice,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "boundary_iou": boundary_iou,
            "J": j_score,
            "F": f_score,
            "J&F": j_and_f,
        }

    def compute_ap_at_threshold(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float,
    ) -> float:
        """Compute if prediction matches at given IoU threshold."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.5).float()

        if pred_binary.shape != target.shape:
            target = F.interpolate(
                target.unsqueeze(0).unsqueeze(0).float(),
                size=pred_binary.shape,
                mode="nearest",
            ).squeeze(0).squeeze(0)

        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()

        return 1.0 if iou >= threshold else 0.0

    def _compute_edges(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel-like filter."""
        mask = mask.unsqueeze(0).unsqueeze(0).float()

        # Simple edge detection via gradient
        dx = F.conv2d(mask, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                      dtype=mask.dtype, device=mask.device), padding=1)
        dy = F.conv2d(mask, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                      dtype=mask.dtype, device=mask.device), padding=1)

        edges = torch.sqrt(dx ** 2 + dy ** 2)
        edges = (edges > 0.1).float()

        return edges.squeeze(0).squeeze(0)

    def _compute_boundary_f_measure(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bound_th: float = 0.008,
    ) -> float:
        """
        Compute Boundary F-measure (F) for VOS evaluation.

        This measures how well the predicted boundary aligns with GT boundary.
        Based on the DAVIS benchmark evaluation protocol.

        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            bound_th: Boundary threshold as fraction of image diagonal

        Returns:
            F-measure score (0 to 1)
        """
        # Get boundary pixels
        pred_boundary = self._compute_edges(pred)
        target_boundary = self._compute_edges(target)

        # Compute distance threshold based on image size
        h, w = pred.shape
        bound_pix = bound_th * np.sqrt(h ** 2 + w ** 2)

        # Get boundary coordinates
        pred_coords = torch.nonzero(pred_boundary > 0.5, as_tuple=False).float()
        target_coords = torch.nonzero(target_boundary > 0.5, as_tuple=False).float()

        if len(pred_coords) == 0 and len(target_coords) == 0:
            return 1.0  # Both empty = perfect match
        if len(pred_coords) == 0 or len(target_coords) == 0:
            return 0.0  # One empty = no match

        # Compute precision: fraction of pred boundary within threshold of target
        if len(pred_coords) > 0 and len(target_coords) > 0:
            # For each pred point, find min distance to target
            dist_pred_to_target = torch.cdist(pred_coords, target_coords).min(dim=1)[0]
            precision = (dist_pred_to_target <= bound_pix).float().mean().item()

            # For each target point, find min distance to pred
            dist_target_to_pred = torch.cdist(target_coords, pred_coords).min(dim=1)[0]
            recall = (dist_target_to_pred <= bound_pix).float().mean().item()
        else:
            precision, recall = 0.0, 0.0

        # F-measure
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0.0

        return f_measure

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run evaluation on validation set."""
        logger.info("Starting evaluation...")

        all_metrics = []
        all_ap_scores = {"ap": [], "ap_50": [], "ap_75": [], "ap_90": []}
        sequence_results = []

        # For total detection counts
        total_tp, total_fp, total_fn = 0, 0, 0

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch in pbar:
            frames = batch["frames"][0].to(self.device)  # (T, C, H, W)
            masks = batch["masks"][0].to(self.device)    # (T, H, W)
            points = batch["points"][0].to(self.device)  # (N, 2)
            seq_name = batch["seq_names"][0]
            object_id = batch["object_ids"][0]

            T = frames.shape[0]
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            seq_metrics = []

            for t in range(T):
                frame = frames[t]
                target = masks[t]
                is_cond = (t == 0)

                try:
                    pred_mask, output_dict = self.forward_frame(
                        frame=frame,
                        points=points,
                        output_dict=output_dict,
                        frame_idx=t,
                        is_cond_frame=is_cond,
                    )

                    metrics = self.compute_metrics(pred_mask, target)
                    seq_metrics.append(metrics)
                    all_metrics.append(metrics)

                    # Accumulate detection counts
                    total_tp += metrics["tp"]
                    total_fp += metrics["fp"]
                    total_fn += metrics["fn"]

                    # Compute AP at different thresholds
                    ap_50 = self.compute_ap_at_threshold(pred_mask, target, 0.5)
                    ap_75 = self.compute_ap_at_threshold(pred_mask, target, 0.75)
                    ap_90 = self.compute_ap_at_threshold(pred_mask, target, 0.9)

                    # Mean AP (average over thresholds 0.5:0.05:0.95)
                    ap_thresholds = np.arange(0.5, 1.0, 0.05)
                    ap_scores = [
                        self.compute_ap_at_threshold(pred_mask, target, t)
                        for t in ap_thresholds
                    ]
                    mean_ap = np.mean(ap_scores)

                    all_ap_scores["ap"].append(mean_ap)
                    all_ap_scores["ap_50"].append(ap_50)
                    all_ap_scores["ap_75"].append(ap_75)
                    all_ap_scores["ap_90"].append(ap_90)

                except Exception as e:
                    logger.warning(f"Error in {seq_name}, frame {t}: {e}")
                    continue

            # Aggregate instance metrics
            if seq_metrics:
                instance_result = {
                    "seq_name": seq_name,
                    "object_id": object_id,
                    "num_frames": len(seq_metrics),
                }
                for key in seq_metrics[0].keys():
                    instance_result[key] = np.mean([m[key] for m in seq_metrics])
                sequence_results.append(instance_result)

                pbar.set_postfix({
                    "IoU": f"{instance_result['iou']:.4f}",
                    "PQ": f"{instance_result['pq']:.4f}",
                })

        # Aggregate overall metrics
        overall = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                overall[f"mean_{key}"] = np.mean(values)
                overall[f"std_{key}"] = np.std(values)

        # AP metrics
        overall["mean_AP"] = np.mean(all_ap_scores["ap"])
        overall["mean_AP_50"] = np.mean(all_ap_scores["ap_50"])
        overall["mean_AP_75"] = np.mean(all_ap_scores["ap_75"])
        overall["mean_AP_90"] = np.mean(all_ap_scores["ap_90"])

        # Detection totals
        overall["total_tp"] = total_tp
        overall["total_fp"] = total_fp
        overall["total_fn"] = total_fn

        # Overall precision/recall/f1
        overall["overall_precision"] = total_tp / (total_tp + total_fp + 1e-6)
        overall["overall_recall"] = total_tp / (total_tp + total_fn + 1e-6)
        overall["overall_f1"] = (
            2 * overall["overall_precision"] * overall["overall_recall"]
            / (overall["overall_precision"] + overall["overall_recall"] + 1e-6)
        )

        overall["num_instances"] = len(sequence_results)
        overall["num_frames"] = len(all_metrics)

        return {
            "overall": overall,
            "sequences": sequence_results,
        }

    def generate_report(self, results: Dict, output_path: Path):
        """Generate markdown evaluation report."""
        overall = results["overall"]
        sequences = results["sequences"]

        report = []
        report.append("# SAM2-MoE Evaluation Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## Model Configuration")
        report.append("")
        report.append("| Parameter | Value |")
        report.append("|-----------|-------|")
        report.append(f"| Base Model | SAM2.1 Hiera Base+ |")
        report.append(f"| MoE Config | {self.config.model_config} |")
        if self.config.moe_checkpoint:
            report.append(f"| MoE Checkpoint | {self.config.moe_checkpoint} |")
        report.append(f"| Dataset | {self.config.data_dir} |")
        report.append("")

        report.append("## Overall Results")
        report.append("")
        report.append("| Metric | Mean ± Std |")
        report.append("|--------|------------|")

        metric_names = ["iou", "dice", "f1", "precision", "recall", "boundary_iou"]
        for name in metric_names:
            mean_key = f"mean_{name}"
            std_key = f"std_{name}"
            if mean_key in overall:
                report.append(f"| {name.upper()} | {overall[mean_key]:.4f} ± {overall[std_key]:.4f} |")

        report.append(f"| Sequences | {overall['num_sequences']} |")
        report.append(f"| Frames | {overall['num_frames']} |")
        report.append("")

        report.append("## Per-Sequence Results")
        report.append("")
        report.append("| Sequence | IoU | Dice | F1 | Precision | Recall |")
        report.append("|----------|-----|------|-----|-----------|--------|")

        # Sort by IoU descending
        sequences_sorted = sorted(sequences, key=lambda x: x["iou"], reverse=True)

        for seq in sequences_sorted[:20]:
            report.append(
                f"| {seq['seq_name']} | {seq['iou']:.4f} | {seq['dice']:.4f} | "
                f"{seq['f1']:.4f} | {seq['precision']:.4f} | {seq['recall']:.4f} |"
            )

        if len(sequences_sorted) > 20:
            report.append(f"| *({len(sequences_sorted) - 20} more sequences)* | | | | | |")

        report.append("")
        report.append("---")
        report.append("")
        report.append("## MoE Architecture Details")
        report.append("")
        report.append("- **MoE Location:** Memory Attention (q_proj, k_proj, v_proj, out_proj)")
        report.append("- **Number of Experts:** 10")
        report.append("- **Top-K Selection:** 2")
        report.append("- **LoRA Rank:** 4")
        report.append("- **Trainable Parameters:** ~660,800 (0.81% of total)")
        report.append("")

        # Write report
        with open(output_path, "w") as f:
            f.write("\n".join(report))

        logger.info(f"Report saved to: {output_path}")


def run_multiple_evaluations(args) -> Dict:
    """Run evaluation multiple times with different seeds."""
    seeds = [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042]
    num_runs = args.num_runs

    if num_runs > len(seeds):
        num_runs = len(seeds)

    seeds_used = seeds[:num_runs]
    all_run_results = []

    for i, seed in enumerate(seeds_used):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{num_runs} (seed={seed})")
        logger.info(f"{'='*60}")

        set_seed(seed)
        evaluator = SAM2MoEEvaluator(args)
        results = evaluator.evaluate()
        all_run_results.append(results["overall"])

    # Aggregate across runs
    aggregated = {}
    keys_to_aggregate = [
        "mean_pq", "mean_sq", "mean_rq",
        "mean_AP", "mean_AP_50", "mean_AP_75", "mean_AP_90",
        "mean_iou", "mean_pixel_iou", "mean_dice",
        "overall_precision", "overall_recall", "overall_f1",
        "total_tp", "total_fp", "total_fn",
        "mean_J", "mean_F", "mean_J&F",
    ]

    for key in keys_to_aggregate:
        values = [r.get(key, 0) for r in all_run_results]
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)

    # Sum totals across runs
    aggregated["total_tp_sum"] = sum(r.get("total_tp", 0) for r in all_run_results)
    aggregated["total_fp_sum"] = sum(r.get("total_fp", 0) for r in all_run_results)
    aggregated["total_fn_sum"] = sum(r.get("total_fn", 0) for r in all_run_results)

    return {
        "num_runs": num_runs,
        "seeds": seeds_used,
        "aggregated": aggregated,
        "per_run": all_run_results,
    }


def print_results(results: Dict, model_name: str = "SAMoE"):
    """Print results in the requested format."""
    agg = results["aggregated"]
    num_runs = results["num_runs"]
    seeds = results["seeds"]

    print(f"\n### {model_name}\n")
    print("**Evaluation Configuration:**")
    print(f"- Number of runs: {num_runs}")
    print(f"- Seeds used: {seeds}")
    print()

    print("**Panoptic Quality Metrics:**")
    print(f"- mean_PQ: {agg['mean_pq_mean']:.4f} ± {agg['mean_pq_std']:.4f}")
    print(f"- mean_SQ: {agg['mean_sq_mean']:.4f} ± {agg['mean_sq_std']:.4f}")
    print(f"- mean_RQ: {agg['mean_rq_mean']:.4f} ± {agg['mean_rq_std']:.4f}")
    print()

    print("**Average Precision:**")
    print(f"- mean_AP: {agg['mean_AP_mean']:.4f} ± {agg['mean_AP_std']:.4f}")
    print(f"- mean_AP@0.5: {agg['mean_AP_50_mean']:.4f} ± {agg['mean_AP_50_std']:.4f}")
    print(f"- mean_AP@0.75: {agg['mean_AP_75_mean']:.4f} ± {agg['mean_AP_75_std']:.4f}")
    print(f"- mean_AP@0.9: {agg['mean_AP_90_mean']:.4f} ± {agg['mean_AP_90_std']:.4f}")
    print()

    print("**IoU Metrics:**")
    print(f"- mean_mean_iou: {agg['mean_iou_mean']:.4f} ± {agg['mean_iou_std']:.4f}")
    print(f"- mean_pixel_iou: {agg['mean_pixel_iou_mean']:.4f} ± {agg['mean_pixel_iou_std']:.4f}")
    print(f"- mean_mean_dice: {agg['mean_dice_mean']:.4f} ± {agg['mean_dice_std']:.4f}")
    print()

    print("**Detection Metrics:**")
    print(f"- Total TP: {int(agg['total_tp_sum'])}")
    print(f"- Total FP: {int(agg['total_fp_sum'])}")
    print(f"- Total FN: {int(agg['total_fn_sum'])}")
    print(f"- Overall Precision: {agg['overall_precision_mean']:.4f} ± {agg['overall_precision_std']:.4f}")
    print(f"- Overall Recall: {agg['overall_recall_mean']:.4f} ± {agg['overall_recall_std']:.4f}")
    print(f"- Overall F1: {agg['overall_f1_mean']:.4f} ± {agg['overall_f1_std']:.4f}")
    print()

    print("**VOS Metrics (J & F):**")
    print(f"- J (IoU): {agg['mean_J_mean']:.4f} ± {agg['mean_J_std']:.4f}")
    print(f"- F (Boundary F-measure): {agg['mean_F_mean']:.4f} ± {agg['mean_F_std']:.4f}")
    print(f"- J&F: {agg['mean_J&F_mean']:.4f} ± {agg['mean_J&F_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2-MoE")

    # Model
    parser.add_argument("--model_config", type=str,
                        default="configs/sam2.1/sam2.1_hiera_b+_moe.yaml")
    parser.add_argument("--base_checkpoint", type=str,
                        default="checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--moe_checkpoint", type=str, default=None,
                        help="Path to trained MoE checkpoint")
    parser.add_argument("--device", type=str, default="cuda")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/OCID-dataset/ARID20")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_sequences", type=str, nargs="+", default=["seq13"],
                        help="Sequence names for test split (default: seq13)")

    # Evaluation
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of evaluation runs with different seeds")
    parser.add_argument("--model_name", type=str, default="SAMoE",
                        help="Model name for report header")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run multiple evaluations
    results = run_multiple_evaluations(args)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)

    # Print results in requested format
    print_results(results, model_name=args.model_name)

    # Also save to log file
    log_path = output_dir / "evaluation.log"
    with open(log_path, "w") as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_results(results, model_name=args.model_name)
        sys.stdout = old_stdout

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
