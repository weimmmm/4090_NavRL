"""One Isaac actor + differentiable SFT learner per GPU, sharing its student.

Only CPU gradient tensors use Gloo. CUDA model/context is not duplicated by
a separate learner process. Controller, sensing, braking and K=10 execution
are all supplied by the original actor and online_refill.
"""
from __future__ import annotations

from datetime import timedelta
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from lidar_wam.models.flow_online_sft import DifferentiableFlowPolicy
from lidar_wam.runner.online_supervision import selective_flow_correction_loss
from lidar_wam.runner.super_action_expert import _atomic_save, _autocast
from online_sft_jobs import atomic_json


class LocalFlowSFT:
    synchronize_online = True

    def __init__(self, cfg, checkpoint, out):
        self.cfg, self.checkpoint, self.out = cfg, checkpoint, Path(out)
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.device = torch.device(str(cfg.device))
        self.step = 0
        self.teacher_samples = self.nonzero_updates = 0
        self.lr = float(cfg.online.get("lr", 5e-6))
        self.threshold = float(cfg.online.get("correction_threshold_mps", .3))
        self.tail_weight = float(cfg.online.get("correction_tail_weight", .5))
        self.failure_chunks = int(cfg.online.get("failure_window_chunks", 0))
        self.failure_labels = None
        if self.failure_chunks:
            from online_failure_labels import DelayedFailureLabels
            self.failure_labels = DelayedFailureLabels(self.failure_chunks*10)
        self.save_every = int(cfg.online.get("save_every", 250))
        self.started = time.time()
        self.reply = self.packet()
        self.student = self.expert = self.writer = None
        dist.init_process_group("gloo", rank=self.rank, world_size=self.world,
                                init_method=str(cfg.online.init_method),
                                timeout=timedelta(seconds=180))
        self.payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.payload.pop("optimizer", None)
        self.payload.pop("online_sft_optimizer", None)
        self.invariants = json.loads(Path(str(cfg.online.invariants)).read_text())
        self.check_invariants()
        self.out.mkdir(parents=True, exist_ok=True)

    def packet(self):
        return {"step": self.step, "teacher_target_mode": "student-box",
                "supervision_horizon": 1, "freeze_policy_output_head": True}

    def check_invariants(self):
        root = Path("/workspace/NavRL/isaac-training")
        for name, expected in self.invariants.items():
            actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Protected control/evaluation logic changed: {name}")

    def attach(self, student, packet):
        if self.expert is not None:
            if student is not self.student or int(packet["step"]) != self.step:
                raise ValueError("Local actor/learner identity or version changed")
            return
        self.student, self.expert = student, student.model
        if not getattr(self.expert, "flow_only", False) or self.expert.sample_steps != 10:
            raise ValueError("Local SFT requires the original ten-step Flow policy")
        self.expert.requires_grad_(False)
        for block in self.expert.diffusion.blocks[-2:]:
            block.requires_grad_(True)
        self.parameters = [p for p in self.expert.parameters() if p.requires_grad]
        self.names = [n for n, p in self.expert.named_parameters() if p.requires_grad]
        assert all(n.startswith(("diffusion.blocks.6.", "diffusion.blocks.7.")) for n in self.names)
        reference = copy.deepcopy(self.expert).eval().requires_grad_(False)
        self.reference = DifferentiableFlowPolicy(reference, checkpoint_steps=False)
        self.policy = DifferentiableFlowPolicy(self.expert, checkpoint_steps=True)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr,
                                           weight_decay=0., betas=(.9, .95))
        self.generator = torch.Generator(device=self.device).manual_seed(20261016+self.rank*100003)
        if self.rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard = Path(str(self.cfg.online.get("tensorboard", self.out/"tensorboard")))
            self.writer = SummaryWriter(tensorboard)
            atomic_json(self.out/"config.json", {
                "source_checkpoint": str(self.checkpoint), "source_step": self.payload["step"],
                "world_size": self.world, "one_simulator_and_shared_student_per_gpu": True,
                "cpu_gloo_gradient_reduction": True, "trainable_names": self.names,
                "lr": self.lr, "correction_threshold_mps": self.threshold,
                "correction_tail_weight": self.tail_weight, "weight_decay": 0.,
                "failure_window_chunks": self.failure_chunks,
                "terminal_feedback_is_loss_selector_not_policy_input": bool(self.failure_chunks),
                "sample_steps": 10, "supervision_horizon": 1, "execution_k": 10,
                "uses_offline_replay": False, "shared_student_reference_noise": True,
                "protected_logic_sha256": self.invariants,
                "local_num_envs": int(self.cfg.env.num_envs),
                "tensorboard": str(tensorboard),
                "target_updates": int(self.cfg.online.target_updates)})
            self.save()
        dist.barrier()

    def recv(self):
        return self.reply

    def send(self, message):
        if message["command"] in ("finish", "session_done"):
            self.reply = {"step": self.step, "ack": message["command"]}
            return
        if message["command"] != "update" or int(message["student_version"]) != self.step:
            raise ValueError("Only fresh, synchronized chunk-start inputs are accepted")
        # run_refill's inference loop deliberately uses no_grad. Enable grads
        # here only; no gradient ever enters the simulator or PPO teacher.
        with torch.enable_grad():
            self.update(message)

    def update(self, message):
        fresh = message["batch"]
        if self.failure_labels is not None:
            fresh = self.failure_labels.push(fresh, message["row_slots"], message["row_episodes"],
                                             int(message["physics_step"]), message["failure_events"])
        n = len(fresh["target"])
        if (np.asarray(fresh["target"]).shape != (n, 3)
                or any(not np.isfinite(x).all() for x in fresh.values())
                or np.any(np.abs(fresh["target"]) > 1.0001)):
            raise ValueError("Malformed per-axis, current-state PPO targets")
        count = torch.tensor(n, dtype=torch.float64)
        dist.all_reduce(count)
        total = int(count)
        if not total:
            self.reply = dict(self.packet(), skipped_empty_global_batch=True)
            return
        self.optimizer.zero_grad(set_to_none=True)
        values = torch.zeros(9, dtype=torch.float64)
        if n:
            batch = {k: torch.from_numpy(v).to(self.device).float() for k, v in fresh.items()}
            noise = torch.randn(n, 10, 3, device=self.device, generator=self.generator)
            with _autocast(self.device, "bf16"):
                prediction = self.policy(batch["latent"], batch["goal"], batch["proprio"], noise)
                with torch.no_grad():
                    original = self.reference(batch["latent"], batch["goal"], batch["proprio"], noise)
                loss, stats, tail_loss = selective_flow_correction_loss(
                    prediction, batch["target"], original, self.threshold, 1, n, self.tail_weight,
                    batch.get("correction_eligible"))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite online loss")
            (loss*n).backward()
            first_mae = (prediction[:, 0].detach().float()-batch["target"]).abs().mean()*2
            saturation = (prediction.detach().float().abs() >= .999).float().mean()
            eligible_fraction = batch.get("correction_eligible", torch.ones(n, device=self.device)).mean()
            values[:] = torch.stack((loss.detach(), first_mae, saturation,
                                      *stats, tail_loss.detach(), eligible_fraction)).double().cpu()*n
        # Sum per-state gradients, not per-rank means. Empty and uneven ranks
        # contribute the correct weight and still enter every collective.
        flat = torch.cat([(p.grad.detach().float().cpu().flatten()
                           if p.grad is not None else torch.zeros(p.numel())) for p in self.parameters])
        dist.all_reduce(flat)
        flat /= total
        norm = flat.norm()
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Non-finite global gradient; update rejected")
        flat /= norm.clamp_min(1.)
        offset = 0
        for p in self.parameters:
            p.grad = flat[offset:offset+p.numel()].to(self.device).view_as(p)
            offset += p.numel()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        dist.all_reduce(values)
        values /= total
        self.teacher_samples += int(round(float(values[3])*total))
        self.nonzero_updates += int(float(norm) > 1e-12)
        rows = [None]*self.world
        dist.all_gather_object(rows, {"rank": self.rank, "samples": n,
                                     "physics_step": int(message["physics_step"]),
                                     **message.get("diagnostics", {})})
        if self.rank == 0:
            metrics = dict(zip(("loss", "first_action_mae_mps", "axis_saturation_fraction",
                                "correction_fraction", "reference_teacher_gap_mps",
                                "teacher_bc", "preservation_bc", "corrected_tail_bc",
                                "failure_eligible_fraction"), values.tolist()))
            metrics.update(step=self.step, global_valid_batch=total, global_offline_batch=0,
                           teacher_corrected_samples_total=self.teacher_samples,
                           nonzero_gradient_updates=self.nonzero_updates,
                           grad_norm=float(norm), lr=self.lr, elapsed_s=time.time()-self.started,
                           full_fleet_resets=sum(x.get("full_fleet_resets", 0) for x in rows),
                           refilled_slots=sum(x.get("refilled_slots", 0) for x in rows),
                           per_rank_samples=[x["samples"] for x in rows],
                           updated_at=time.time())
            atomic_json(self.out/"progress.json", metrics)
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar("train/"+key, value, self.step)
            if self.step == 1 or self.step % 10 == 0:
                print(json.dumps(metrics), flush=True)
                self.writer.flush()
            if self.step % self.save_every == 0 or self.step == int(self.cfg.online.target_updates):
                self.save()
        self.reply = dict(self.packet(), trained_samples=n)

    def save(self):
        self.check_invariants()
        saved = {k: v for k, v in self.payload.items() if k != "model"}
        saved.update(model=self.expert.state_dict(), step=self.step,
                     online_sft={"step": self.step, "source_checkpoint": str(self.checkpoint),
                                 "source_step": self.payload["step"], "lr": self.lr,
                                 "mode": "pure-online", "uses_offline_replay": False,
                                 "selective_correction_threshold_mps": self.threshold,
                                 "correction_tail_weight": self.tail_weight, "weight_decay": 0.,
                                 "failure_window_chunks": self.failure_chunks,
                                 "feedback_used_only_for_label_selection": bool(self.failure_chunks),
                                 "teacher_corrected_samples_total": self.teacher_samples,
                                 "nonzero_gradient_updates": self.nonzero_updates,
                                 "one_simulator_per_gpu": True, "world_size": self.world,
                                 "protected_logic_sha256": self.invariants},
                     online_sft_optimizer=self.optimizer.state_dict())
        _atomic_save(saved, self.out/f"sft_step_{self.step}.pt")
        _atomic_save(saved, self.out/"latest.pt")

    def close(self):
        pass  # run_refill closes its protocol before the final shared barrier.

    def finalize(self, metadata):
        if self.failure_labels is not None:
            metadata.update(delayed_states_seen=self.failure_labels.seen,
                            delayed_states_consumed_once=self.failure_labels.emitted,
                            delayed_states_discarded=self.failure_labels.remaining)
        digest = hashlib.sha256()
        for p in self.parameters:
            digest.update(p.detach().float().cpu().contiguous().numpy().tobytes())
        hashes = [None]*self.world
        dist.all_gather_object(hashes, digest.hexdigest())
        if len(set(hashes)) != 1:
            raise RuntimeError("Workers ended with different trainable weights")
        metadata["trainable_weights_sha256"] = hashes[self.rank]
        metadata["all_ranks_synchronized"] = True
        atomic_json(self.out/f"actor_rank_{self.rank}.json", metadata)
        dist.barrier()
        if self.rank == 0:
            self.save()
            atomic_json(self.out/"training_finished.json", {"step": self.step, "success": True})
            self.writer.close()
        dist.barrier()
        dist.destroy_process_group()
