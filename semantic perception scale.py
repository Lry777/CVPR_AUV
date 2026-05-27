import torch

def compute_all_semantic_scales(logits, svd_mode='D_HW'):
    """
    logits: Tensor[B, C, D, H, W]
    Returns: Tensor of semantic scales, shape [B, C]
    """
    B, C, D, H, W = logits.shape
    semantic_scales = torch.zeros((B, C), device=logits.device)

    for c in range(1, C):  # 遍历所有类别
        for b in range(B):  # 遍历每个batch样本
            cls_logit = logits[b, c]  # [D, H, W]

            # reshape 为 2D 矩阵做 SVD
            if svd_mode == 'D_HW':
                mat = cls_logit.view(D, H * W)
                mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            elif svd_mode == 'H_DW':
                mat = cls_logit.permute(1, 0, 2).reshape(H, D * W)
                mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            elif svd_mode == 'W_DH':
                mat = cls_logit.permute(2, 0, 1).reshape(W, D * H)
                mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError("Unsupported svd_mode")

            try:
                U, S, Vh = torch.linalg.svd(mat.float(), full_matrices=False)
                S = S + 1e-6
                singular_dis = S / (S.sum())
            except RuntimeError:
                singular_dis = torch.tensor(float('nan'), device=logits.device)

            # 信息熵封装,semantic_scales越高代表信息含量越丰富
            entropy = -torch.sum(singular_dis * torch.log(singular_dis))
            max_entropy = torch.log(torch.tensor(len(S), dtype=torch.float32))
            normalized_singular_entropy = entropy / max_entropy  # ∈ [0, 1]

            semantic_scales[b, c] = normalized_singular_entropy

    return semantic_scales  # [B, C]