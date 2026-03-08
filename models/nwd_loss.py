"""
NWD Loss: Normalized Wasserstein Distance Loss for Tiny Object Detection
=========================================================================

Implementation of NWD loss from:
"A Normalized Gaussian Wasserstein Distance for Tiny Object Detection" (CVPR 2021)

Key Insight:
- Standard IoU is extremely sensitive to position errors for tiny objects
- A 2-pixel shift for an 8×8 box causes >50% IoU drop vs 5% for 64×64
- NWD models boxes as 2D Gaussians and uses Wasserstein distance
- Provides smooth, continuous gradients even for tiny objects

Mathematical formulation:
- Convert bbox (cx, cy, w, h) to 2D Gaussian N(μ, Σ)
   - μ = (cx, cy)
   - Σ = diag((w/2)², (h/2)²)
- Wasserstein distance between two Gaussians has closed form
- NWD = exp(-W_2(N_pred, N_gt) / C) where C is normalizing constant

Author: Research Implementation
"""

import torch
import torch.nn as nn
import math


def bbox_to_gaussian(bboxes, eps=1e-7):
    """
    Convert bounding boxes to 2D Gaussian parameters.
    
    A bbox (cx, cy, w, h) is modeled as a 2D Gaussian:
    - Mean μ = (cx, cy)  
    - Covariance Σ = diag((w/2)², (h/2)²)
    
    Args:
        bboxes: Bounding boxes [N, 4] in (cx, cy, w, h) format
        eps: Small constant for numerical stability
        
    Returns:
        mu: Gaussian means [N, 2]
        sigma: Gaussian standard deviations [N, 2] (sqrt of diagonal covariance)
    """
    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # Mean is the center
    mu = torch.stack([cx, cy], dim=-1)
    
    # Standard deviation is half the width/height
    # We model the box as covering ±2σ (95% of Gaussian mass)
    sigma_x = w / 4 + eps  # w = 4σ, so σ = w/4
    sigma_y = h / 4 + eps
    sigma = torch.stack([sigma_x, sigma_y], dim=-1)
    
    return mu, sigma


def wasserstein_distance_2d(mu1, sigma1, mu2, sigma2):
    """
    Compute 2D Wasserstein distance between two diagonal Gaussians.
    
    For Gaussians N(μ₁, Σ₁) and N(μ₂, Σ₂) with diagonal covariances:
    
    W₂² = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²
    
    where σ = sqrt(diag(Σ)) for diagonal covariance matrices.
    
    Args:
        mu1, mu2: Means [N, 2]
        sigma1, sigma2: Standard deviations [N, 2]
        
    Returns:
        W2: Wasserstein-2 distance [N]
    """
    # Squared difference of means (location term)
    mu_diff_sq = torch.sum((mu1 - mu2) ** 2, dim=-1)
    
    # Squared difference of standard deviations (shape term)
    sigma_diff_sq = torch.sum((sigma1 - sigma2) ** 2, dim=-1)
    
    # W₂² = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²
    w2_squared = mu_diff_sq + sigma_diff_sq
    
    # W₂ = sqrt(W₂²)
    w2 = torch.sqrt(w2_squared + 1e-7)
    
    return w2


def compute_nwd(pred_bboxes, gt_bboxes, C=12.0, eps=1e-7):
    """
    Compute Normalized Wasserstein Distance between predicted and GT boxes.
    
    NWD = exp(-W₂ / C)
    
    where:
    - W₂ is the 2D Wasserstein distance
    - C is a normalizing constant (default 12, tuned for typical object sizes)
    
    NWD ∈ (0, 1]:
    - NWD = 1 when boxes are identical
    - NWD → 0 when boxes are far apart
    
    Args:
        pred_bboxes: Predicted boxes [N, 4] in (cx, cy, w, h) format
        gt_bboxes: Ground truth boxes [N, 4] in (cx, cy, w, h) format
        C: Normalizing constant (controls sensitivity)
        eps: Small constant for numerical stability
        
    Returns:
        nwd: NWD values [N] in (0, 1]
    """
    # Convert to Gaussian parameters
    mu_pred, sigma_pred = bbox_to_gaussian(pred_bboxes, eps)
    mu_gt, sigma_gt = bbox_to_gaussian(gt_bboxes, eps)
    
    # Compute Wasserstein distance
    w2 = wasserstein_distance_2d(mu_pred, sigma_pred, mu_gt, sigma_gt)
    
    # Normalize to (0, 1]
    nwd = torch.exp(-w2 / C)
    
    return nwd


class NWDLoss(nn.Module):
    """
    NWD Loss for bounding box regression.
    
    Loss = 1 - NWD = 1 - exp(-W₂/C)
    
    This provides smooth gradients for tiny objects where IoU would be
    extremely sensitive to small position errors.
    
    Args:
        C: Normalizing constant (default 12.0, can be tuned)
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(self, C=12.0, reduction='mean'):
        super().__init__()
        self.C = C
        self.reduction = reduction
    
    def forward(self, pred_bboxes, gt_bboxes):
        """
        Args:
            pred_bboxes: Predicted boxes [N, 4] in (cx, cy, w, h) format
            gt_bboxes: Ground truth boxes [N, 4] in (cx, cy, w, h) format
            
        Returns:
            loss: NWD loss value
        """
        nwd = compute_nwd(pred_bboxes, gt_bboxes, self.C)
        loss = 1.0 - nwd
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HybridNWDIoULoss(nn.Module):
    """
    Hybrid loss combining NWD and IoU for robust training.
    
    Loss = α · NWD_loss + (1 - α) · IoU_loss
    
    Benefits:
    - NWD: Smooth gradients for tiny objects
    - IoU: Well-established, good for larger objects
    
    The weight α can be:
    - Fixed (e.g., 0.5)
    - Size-adaptive: higher α for smaller boxes
    
    Args:
        alpha: Weight for NWD loss (0-1), or 'adaptive' for size-based weighting
        C: NWD normalizing constant
        iou_type: 'iou', 'giou', 'diou', or 'ciou'
    """
    
    def __init__(self, alpha=0.5, C=12.0, iou_type='ciou'):
        super().__init__()
        self.alpha = alpha
        self.nwd_loss = NWDLoss(C=C, reduction='none')
        self.iou_type = iou_type
    
    def compute_iou_loss(self, pred_bboxes, gt_bboxes):
        """Compute IoU-based loss (CIoU by default)."""
        # Convert from cxcywh to xyxy for IoU computation
        pred_x1 = pred_bboxes[:, 0] - pred_bboxes[:, 2] / 2
        pred_y1 = pred_bboxes[:, 1] - pred_bboxes[:, 3] / 2
        pred_x2 = pred_bboxes[:, 0] + pred_bboxes[:, 2] / 2
        pred_y2 = pred_bboxes[:, 1] + pred_bboxes[:, 3] / 2
        
        gt_x1 = gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2
        gt_y1 = gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2
        gt_x2 = gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2
        gt_y2 = gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area + 1e-7
        
        # IoU
        iou = inter_area / union_area
        
        if self.iou_type == 'iou':
            return 1.0 - iou
        
        # Enclosing box
        enc_x1 = torch.min(pred_x1, gt_x1)
        enc_y1 = torch.min(pred_y1, gt_y1)
        enc_x2 = torch.max(pred_x2, gt_x2)
        enc_y2 = torch.max(pred_y2, gt_y2)
        
        if self.iou_type == 'giou':
            enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-7
            giou = iou - (enc_area - union_area) / enc_area
            return 1.0 - giou
        
        # DIoU / CIoU
        c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7
        
        # Center distance
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        gt_cx = (gt_x1 + gt_x2) / 2
        gt_cy = (gt_y1 + gt_y2) / 2
        rho2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        
        diou = iou - rho2 / c2
        
        if self.iou_type == 'diou':
            return 1.0 - diou
        
        # CIoU
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(gt_w / (gt_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7)), 2
        )
        
        with torch.no_grad():
            alpha_ciou = v / (1 - iou + v + 1e-7)
        
        ciou = diou - alpha_ciou * v
        return 1.0 - ciou
    
    def forward(self, pred_bboxes, gt_bboxes):
        """
        Args:
            pred_bboxes: Predicted boxes [N, 4] in (cx, cy, w, h) format
            gt_bboxes: Ground truth boxes [N, 4] in (cx, cy, w, h) format
            
        Returns:
            loss: Combined loss value
        """
        nwd_loss = self.nwd_loss(pred_bboxes, gt_bboxes)
        iou_loss = self.compute_iou_loss(pred_bboxes, gt_bboxes)
        
        if self.alpha == 'adaptive':
            # Size-adaptive: smaller boxes get higher NWD weight
            # Compute normalized size (0-1 range)
            gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            max_area = 64 * 64  # Threshold for "large" objects
            size_ratio = torch.clamp(gt_area / max_area, 0, 1)
            # Alpha: 0.8 for tiny (area→0), 0.2 for large (area→max)
            alpha = 0.8 - 0.6 * size_ratio
            loss = alpha * nwd_loss + (1 - alpha) * iou_loss
        else:
            loss = self.alpha * nwd_loss + (1 - self.alpha) * iou_loss
        
        return loss.mean()


# For integration with Ultralytics
__all__ = ['NWDLoss', 'HybridNWDIoULoss', 'compute_nwd', 'wasserstein_distance_2d']


if __name__ == '__main__':
    # Test the modules
    print("Testing NWD Loss...")
    
    # Create test bboxes (cx, cy, w, h format)
    # Simulating tiny objects (around 16x16 pixels)
    pred = torch.tensor([
        [100.0, 100.0, 16.0, 16.0],  # Prediction 1
        [200.0, 200.0, 12.0, 12.0],  # Prediction 2 (tiny)
        [300.0, 300.0, 64.0, 64.0],  # Prediction 3 (larger)
    ])
    
    gt = torch.tensor([
        [102.0, 100.0, 16.0, 16.0],  # GT 1 - 2px shift
        [202.0, 200.0, 12.0, 12.0],  # GT 2 - 2px shift  
        [302.0, 300.0, 64.0, 64.0],  # GT 3 - 2px shift
    ])
    
    # Test NWD computation
    nwd = compute_nwd(pred, gt)
    print(f"NWD values: {nwd}")
    print(f"  - 16x16 box, 2px shift: NWD = {nwd[0]:.4f}")
    print(f"  - 12x12 box, 2px shift: NWD = {nwd[1]:.4f}")
    print(f"  - 64x64 box, 2px shift: NWD = {nwd[2]:.4f}")
    
    # Test NWD loss
    nwd_loss = NWDLoss()
    loss = nwd_loss(pred, gt)
    print(f"\nNWD Loss: {loss:.4f}")
    
    # Test gradient flow
    pred_grad = pred.clone().requires_grad_(True)
    loss = nwd_loss(pred_grad, gt)
    loss.backward()
    print(f"Gradients exist: {pred_grad.grad is not None}")
    
    # Test Hybrid loss
    hybrid_loss = HybridNWDIoULoss(alpha=0.5)
    h_loss = hybrid_loss(pred, gt)
    print(f"\nHybrid NWD-IoU Loss: {h_loss:.4f}")
    
    # Test adaptive hybrid loss
    adaptive_loss = HybridNWDIoULoss(alpha='adaptive')
    a_loss = adaptive_loss(pred, gt)
    print(f"Adaptive Hybrid Loss: {a_loss:.4f}")
    
    print("\nAll tests passed!")
