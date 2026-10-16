"""Match online teacher labels to the student's existing action domain.

The original deterministic student and its deployment adapter both project
normalized velocity commands onto the unit ball (physical speed <= 2 m/s).
PPO instead bounds each component in its goal frame. Projection is an explicit
LABEL ablation, not an alteration of PPO weights or the evaluation controller.
It preserves feasible labels and the direction of every finite command.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


TARGET_MODES = ("raw", "student-norm", "student-box")


def student_feasible_target(target: torch.Tensor) -> torch.Tensor:
    """Project normalized commands [...,3] to the deployed unit-speed ball.

Work in float32 even under autocast. Do not sanitize invalid labels: the actor
excludes their rows, and the learner rejects non-finite supervision.
"""
    if target.ndim < 2 or target.shape[-1] != 3:
        raise ValueError("Expected normalized velocity commands [...,3]")
    value = target.float()
    magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value / magnitude.clamp_min(1.)


def online_teacher_target(target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return target
    if mode == "student-norm":
        return student_feasible_target(target)
    if mode == "student-box":
        return target.float().clamp(-1., 1.)
    raise ValueError(f"Unknown teacher target mode: {mode!r}")


def target_diagnostics(raw: torch.Tensor, target: torch.Tensor,
                       valid: torch.Tensor) -> dict[str, float]:
    """Diagnostics on precisely the fresh rows sent to the learner."""
    raw, target = raw[valid].float(), target[valid].float()
    if not len(raw):
        return {}
    return {
        "teacher_raw_speed_over_2_fraction": float((raw.norm(dim=-1)>1.0001).float().mean()),
        "teacher_raw_mean_speed_mps": float(raw.norm(dim=-1).mean()*2.),
        "supervision_mean_speed_mps": float(target.norm(dim=-1).mean()*2.),
        "teacher_label_projection_mae_mps": float((raw-target).abs().mean()*2.),
    }


def current_state_anchor_loss(prediction: torch.Tensor, reference: torch.Tensor,
                              world_size: int, total_samples: int) -> torch.Tensor:
    """Whole-chunk trust penalty on the SAME freshly visited student states.

    The reference is the frozen ORIGINAL policy, not an offline data batch or
    a future observation. Match the learner's uneven-DDP sample weighting.
    No gradient may flow to the reference, even if a caller forgot no_grad.
    """
    if prediction.shape != reference.shape or prediction.shape[1:] != (10, 3):
        raise ValueError("Expected matched original/student chunks [B,10,3]")
    if total_samples <= 0:
        return prediction.sum()*0.
    error = F.smooth_l1_loss(prediction.float(), reference.detach().float(),
                             reduction="none", beta=.25)
    return error.mean((1, 2)).sum()*world_size/total_samples


def masked_online_tail_loss(prediction: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor, world_size: int,
                            tail_samples: int) -> torch.Tensor:
    """Supervise commands 1..9, excluding post-terminal/invalid padding.

    Each row with at least one valid tail command has equal weight. The first
    action keeps its existing standalone loss, so this term does not weaken
    first-action supervision by dividing it by the ten-step horizon.
    """
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise ValueError("Online tail shapes differ")
    if prediction.shape[1:] != (10, 3):
        raise ValueError("Expected ten three-dimensional commands")
    if tail_samples <= 0:
        return prediction.sum()*0.
    error = F.smooth_l1_loss(prediction[:, 1:].float(), target[:, 1:].float(),
                             reduction="none", beta=.25).mean(-1)
    valid = mask[:, 1:].float()
    row_loss = (error*valid).sum(-1)/valid.sum(-1).clamp_min(1.)
    return row_loss.sum()*world_size/tail_samples


def selective_online_correction_loss(prediction: torch.Tensor,
                                     target: torch.Tensor,
                                     reference: torch.Tensor,
                                     threshold_mps: float,
                                     world_size: int,
                                     total_samples: int, correction_eligible=None):
    """Choose ONE supervision source per fresh state, never two competing ones.

    A frozen original policy sees the same current online student inputs.
    If its first command and the feasible PPO command disagree by more than
    ``threshold_mps``, supervise action zero with PPO. Otherwise preserve the
    original entire chunk. No old-data replay, future labels, rollout reward,
    or deployment-time intervention is introduced by this training objective.

    Commands use the ORIGINAL expert's physical scale of 2 m/s. The detached
    selector is defined by original-versus-teacher disagreement, NOT by the
    changing student's error, so correcting a sample does not remove its label.
    Returned metrics use the same global-state mean as the loss, allowing
    uneven/empty DDP ranks without overweighting either supervision source.
    """
    if (prediction.ndim != 3 or prediction.shape[1:] != (10, 3)
            or reference.shape != prediction.shape
            or target.shape != (len(prediction), 3)):
        raise ValueError("Require prediction/reference [B,10,3] and PPO target [B,3]")
    if not 0 < threshold_mps < float("inf") or world_size < 1:
        raise ValueError("Require a finite positive physical correction threshold")
    if total_samples <= 0:
        zero = prediction.sum()*0.
        return zero, torch.zeros(4, device=prediction.device)
    original, teacher = reference.detach().float(), target.detach().float()
    gap = (original[:, 0]-teacher).norm(dim=-1)*2.
    correct = gap > float(threshold_mps)
    if correction_eligible is not None:
        if (correction_eligible.shape != (len(prediction),)
                or not bool(((correction_eligible == 0) | (correction_eligible == 1)).all())):
            raise ValueError("Correction eligibility must be a binary label-only mask [B]")
        correct &= correction_eligible.detach().bool()
    teacher_error = F.smooth_l1_loss(
        prediction[:, 0].float(), teacher, reduction="none", beta=.25).mean(-1)
    keep_error = F.smooth_l1_loss(
        prediction.float(), original, reduction="none", beta=.25).mean((1, 2))
    scale = world_size/total_samples
    loss = torch.where(correct, teacher_error, keep_error).sum()*scale
    metrics = torch.stack((correct.float().sum(), gap.sum(),
                           (teacher_error*correct).sum(),
                           (keep_error*~correct).sum())).detach()*scale
    return loss, metrics


def selective_flow_correction_loss(prediction, target, reference, threshold_mps,
                                   world_size, total_samples, tail_weight=.5,
                                   correction_eligible=None):
    """PPO corrects only action zero; protect the other nine on corrected rows.

    Agreement rows retain the existing whole-chunk preservation objective.
    There is no competing reference loss on the PPO-supervised first action.
    All terms use the global fresh-state denominator (including uneven ranks).
    """
    if not 0 <= tail_weight < float("inf"):
        raise ValueError("Tail preservation weight must be finite and nonnegative")
    loss, metrics = selective_online_correction_loss(
        prediction, target, reference, threshold_mps, world_size, total_samples,
        correction_eligible)
    if total_samples <= 0:
        return loss, metrics, prediction.sum()*0.
    original = reference.detach().float()
    correction = ((original[:, 0]-target.detach().float()).norm(dim=-1)*2 > threshold_mps)
    if correction_eligible is not None:
        correction &= correction_eligible.detach().bool()
    error = F.smooth_l1_loss(prediction[:, 1:].float(), original[:, 1:],
                             reduction="none", beta=.25).mean((1, 2))
    tail = (error*correction).sum()*world_size/total_samples
    return loss+tail_weight*tail, metrics, tail
