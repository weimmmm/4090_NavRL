"""Synchronous four-GPU learner: one fresh online update per committed chunk.

Isaac sends CURRENT normalized student inputs every ten physics steps via a
private Unix-domain socket. No episode-sized collection or replay step comes
between observing the student state and updating its parameters. The learner
returns only the trainable tail; the actor applies it before generating its
next complete ten-command chunk. All other weights remain frozen. Default
mode is PURE ONLINE: no old-data cache is opened and only action zero has loss.
Optional horizon=10 consumes the just-executed student chunk once, supervising
PPO correction labels on all valid positions. No future observation is input.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from multiprocessing.connection import Listener, Client

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from lidar_wam.runner.action_expert_closed_loop_sft import (
    DirectPolicy, build_model, draw, supervised_loss, validate)
from lidar_wam.runner.super_action_expert import _atomic_save, _autocast
from lidar_wam.runner.online_supervision import (
    TARGET_MODES, current_state_anchor_loss, masked_online_tail_loss,
    selective_online_correction_loss, student_feasible_target)
from lidar_wam.models.flow_online_sft import DifferentiableFlowPolicy


def policy_wrapper(expert, project_output=True):
    return (DifferentiableFlowPolicy(expert, project_output=project_output)
            if getattr(expert, "flow_only", False)
            else DirectPolicy(expert, project_output=project_output))


def tail(model):
    return [(name, value) for name, value in model.named_parameters() if value.requires_grad]


def weight_packet(model, step):
    entries = tail(model)
    return {"step": step,
            "teacher_target_mode": getattr(model, "online_teacher_target_mode", "raw"),
            "supervision_horizon": getattr(model, "online_supervision_horizon", 1),
            "freeze_policy_output_head": getattr(model, "online_freeze_policy_output_head", False),
            "names": [name for name, _ in entries],
            "weights": torch.cat([value.detach().flatten().float().cpu()
                                  for _, value in entries]).numpy()}


def first_step_loss(prediction, target, world_size, total_samples):
    # DDP averages rank gradients. Compensate so each fresh state has exactly
    # the same weight even when the active-drone count is not divisible by 4.
    if total_samples <= 0:
        return prediction.sum()*0.
    return F.smooth_l1_loss(prediction[:, 0].float(), target.float(),
                            reduction="none", beta=.25).mean(-1).sum()*world_size/total_samples


def serve(args):
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if args.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed+rank)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload.pop("optimizer", None)
    expert = build_model(payload, freeze_policy_output_head=args.freeze_policy_output_head).to(device)
    expert.online_freeze_policy_output_head = args.freeze_policy_output_head
    expert.online_teacher_target_mode = args.teacher_target_mode
    expert.online_supervision_horizon = args.supervision_horizon
    flow_only = bool(getattr(expert, "flow_only", False))
    if flow_only and (args.mode != "pure-online" or args.supervision_horizon != 1
                      or args.teacher_target_mode != "student-box"):
        raise ValueError("Flow SFT requires pure-online first-action labels in the per-axis box")
    model = policy_wrapper(expert, project_output=not args.unclipped_training_output)
    if world > 1:
        model = DDP(model, device_ids=[local], broadcast_buffers=False)
    parameters = [value for _, value in tail(expert)]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01, betas=(.9, .95))
    if "online_sft_optimizer" in payload:
        optimizer.load_state_dict(payload["online_sft_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    pure = args.mode == "pure-online"
    selective = args.selective_correction_threshold_mps > 0
    if not 0 <= args.selective_correction_threshold_mps < float("inf"):
        raise ValueError("Selective correction threshold must be finite and nonnegative")
    if selective and (not pure or args.supervision_horizon != 1
                      or args.anchor_weight != 0 or args.unclipped_training_output
                      or args.teacher_target_mode != ("student-box" if flow_only else "student-norm")):
        raise ValueError("Selective correction requires pure-online feasible first-action labels, "
                         "normal deployment projection, and no simultaneous anchor/tail ablation")
    if args.supervision_horizon == 10 and not pure:
        raise ValueError("Complete online chunk supervision does not mix offline data")
    reference = None
    if args.anchor_weight > 0 or selective:
        if not pure:
            raise ValueError("Current-state anchor pilot is pure-online only")
        source = torch.load(args.anchor_checkpoint or args.checkpoint,
                            map_location="cpu", weights_only=False)
        if source.get("online_sft") or source.get("closed_loop_sft"):
            raise ValueError("Anchor must be the original expert, not an SFT checkpoint")
        if source["vae"] != payload["vae"] or source["condition_stats"] != payload["condition_stats"]:
            raise ValueError("Original anchor inputs/normalization differ from student")
        if source.get("format") != payload.get("format"):
            raise ValueError("Reference and student formats differ")
        reference = policy_wrapper(build_model(source).to(device))
        reference.eval().requires_grad_(False)
        del source
    cache = None
    if not pure:
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
        if abs(cache["metadata"]["scaling_factor"]-payload["vae"]["scaling_factor"]) > 1e-6:
            raise ValueError("Cache VAE differs from student")
    generator = torch.Generator().manual_seed(args.seed+rank*100003)
    writer = listener = connection = None
    step = int(payload.get("online_sft", {}).get("step", 0))
    if step != args.start_step:
        raise ValueError(f"Checkpoint step {step} differs from requested start {args.start_step}")
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        args.out.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.out/"tensorboard")
        config = {**vars(args), "world_size": world,
                  "trainable_parameters": sum(p.numel() for p in parameters),
                  "teacher_query_interval_physics_steps": 10 if args.supervision_horizon == 1 else 1,
                  "online_action_index": 0 if args.supervision_horizon == 1 else "0:9",
                  "selective_correction_threshold_mps": args.selective_correction_threshold_mps,
                  "selective_supervision": ("PPO first action on significant original/PPO disagreement; "
                                            "otherwise original full chunk on SAME online state"
                                            if selective else None),
                  "uses_offline_replay": not pure,
                  "refill_enabled": args.refill_enabled,
                  "batch_semantics": ("just-executed online chunk starts; once-only, masked terminal tails"
                                      if args.supervision_horizon == 10 else
                                      "fresh current states only; reset/warmup/braking rows excluded" if pure else
                                      "fresh states plus offline anchors"),
                  "maximum_global_batch_size": args.per_gpu_batch_size*world,
                  "student_update_before_chunk_generation": True,
                  "flow_only": flow_only,
                  "sampling_steps": expert.sample_steps if flow_only else 0,
                  "differentiates_through_all_euler_steps": flow_only,
                  "shared_student_reference_noise": flow_only and reference is not None,
                  "trainable_names": [name for name, _ in tail(expert)]}
        (args.out/"config.json").write_text(json.dumps(config, indent=2, default=str)+"\n")
        if cache is not None:
            metrics = validate(model.module if world > 1 else model, cache["val"], device)
            for key, value in metrics.items():
                writer.add_scalar("validation/"+key, value, step)
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
        writer.add_text("config/training_mode", args.mode, step)
        print(json.dumps({"mode": args.mode, "uses_offline_replay": not pure,
                          "maximum_per_gpu_batch": args.per_gpu_batch_size}), flush=True)
        if args.socket.exists():
            raise FileExistsError(args.socket)
        listener = Listener(str(args.socket), family="AF_UNIX")
        os.chmod(args.socket, 0o600)
        print(f"[online learner] ready: {args.socket} world_size={world}", flush=True)
        writer.flush()
    model.train()
    started = time.time()

    def save(current_step):
        saved = {key: value for key, value in payload.items() if key != "model"}
        saved.update(model=expert.state_dict(), step=current_step,
                     architecture={**payload["architecture"], "training_action_diffusion": flow_only},
                     online_sft={"step": current_step, "source_checkpoint": str(args.checkpoint),
                                 "lr": args.lr, "online_weight": args.online_weight,
                                 "mode": args.mode, "uses_offline_replay": not pure,
                                 "teacher_target_mode": args.teacher_target_mode,
                                 "supervision_horizon": args.supervision_horizon,
                                 "unclipped_training_output": args.unclipped_training_output,
                                 "deployment_output_projection": True,
                                 "online_tail_weight": args.online_tail_weight,
                                 "target_rollout": ("PPO corrections at student-visited states; not a counterfactual PPO rollout"
                                                    if args.supervision_horizon == 10 else "current-state PPO first action"),
                                 "label_latency_physics_steps": 10 if args.supervision_horizon == 10 else 0,
                                 "anchor_weight": args.anchor_weight,
                                 "freeze_policy_output_head": args.freeze_policy_output_head,
                                 "selective_correction_threshold_mps": args.selective_correction_threshold_mps,
                                 "selective_reference_checkpoint": str(args.anchor_checkpoint or args.checkpoint)
                                 if selective else None,
                                 "selective_supervision": ("one detached supervision source per fresh state; "
                                                           "feasible PPO first command OR original full chunk"
                                                           if selective else None),
                                 "anchor_checkpoint": str(args.anchor_checkpoint or args.checkpoint)
                                 if args.anchor_weight > 0 else None,
                                 "anchor_input": "same fresh student state; no offline replay"
                                 if args.anchor_weight > 0 else None,
                                 "per_gpu_total_batch_size": args.per_gpu_batch_size,
                                 "refill_enabled": args.refill_enabled,
                                 "query_interval_physics_steps": 10 if args.supervision_horizon == 1 else 1,
                                 "optimizer_update_interval_steps": 10,
                                 "action_index": 0 if args.supervision_horizon == 1 else "0:9",
                                 "update_before_chunk_generation": True},
                     online_sft_optimizer=optimizer.state_dict())
        _atomic_save(saved, args.out/f"sft_step_{current_step}.pt")
        _atomic_save(saved, args.out/"latest.pt")

    def progress(current_step, metrics=None):
        value = {"step": current_step, "updated_at": time.time(),
                 "learner_pid": os.getpid(), **(metrics or {})}
        temporary = args.out/"progress.json.tmp"
        temporary.write_text(json.dumps(value)+"\n")
        os.replace(temporary, args.out/"progress.json")

    try:
        if rank == 0:
            # Publish the resume model for an INITIAL fixed-validation point.
            # All snapshot paths are atomic and immutable for the async workers.
            save(step)
            progress(step)
        while True:
            message = None
            if rank == 0:
                if connection is None:
                    connection = listener.accept()
                    connection.send(weight_packet(expert, step))
                try:
                    message = connection.recv()
                except EOFError:
                    message = {"command": "abort", "reason": "Isaac actor disconnected unexpectedly"}
            items = [message]
            if world > 1:
                dist.broadcast_object_list(items, src=0, device=device)
            message = items[0]
            command = message["command"]
            if command == "abort":
                raise RuntimeError(message["reason"])
            if command in ("session_done", "finish"):
                if rank == 0:
                    writer.add_scalar("rollout/completed_episodes", message.get("episodes", 0), step)
                    connection.send({"step": step, "ack": command})
                    connection.close()
                    connection = None
                if command == "finish":
                    break
                continue
            if command != "update" or step >= args.max_updates:
                raise RuntimeError(f"Invalid request {command}, step={step}")
            if int(message["student_version"]) != step:
                raise RuntimeError("Actor is not using the latest synchronized student weights")
            fresh = message["batch"]
            n = len(fresh["target"])
            if n <= 0:
                raise ValueError("Empty fresh batch must not trigger an update")
            target_shape = (n, 3) if args.supervision_horizon == 1 else (n, 10, 3)
            if (np.asarray(fresh["target"]).shape != target_shape
                    or not all(np.isfinite(value).all() for value in fresh.values())):
                raise ValueError("Malformed fresh chunk-start supervision")
            tail_samples = 0
            if args.supervision_horizon == 10:
                mask = np.asarray(fresh.get("target_mask"))
                if (mask.shape != (n, 10) or not np.all((mask == 0) | (mask == 1))
                        or not mask[:, 0].all()):
                    raise ValueError("Invalid online chunk supervision mask")
                tail_samples = int(mask[:, 1:].any(axis=1).sum())
            if (args.teacher_target_mode == "student-norm"
                    and np.any(np.linalg.norm(fresh["target"], axis=-1)>1.0001)):
                raise ValueError("Online labels exceed the student's deployed velocity range")
            if (args.teacher_target_mode == "student-box"
                    and np.any(np.abs(fresh["target"]) > 1.0001)):
                raise ValueError("Online labels exceed the per-axis velocity range")
            indices = np.arange(rank, n, world)
            real_count = len(indices)
            if pure:
                if real_count > args.per_gpu_batch_size:
                    raise ValueError("Fresh states exceed the configured per-GPU batch capacity")
                offline_count = global_offline_count = 0
                a = None
            else:
                offline_count = args.per_gpu_batch_size-max(real_count, 1)
                global_offline_count = args.per_gpu_batch_size*world-max(n, world)
                if offline_count <= 0:
                    raise ValueError("Per-GPU batch must leave space for offline supervision")
                a = draw(cache["train"], offline_count, generator, device)
            # Empty ranks still execute identical DDP graph/collectives.
            if not real_count:
                indices = np.asarray([0])
            b = {key: torch.from_numpy(value[indices]).to(device).float() for key, value in fresh.items()}
            optimizer.zero_grad(set_to_none=True)
            # Identical noise prevents stochastic sample differences from
            # being mistaken for original-versus-PPO policy disagreement.
            sample_noise = (torch.randn(len(b["latent"]), 10, 3, device=device)
                            if flow_only else None)
            with _autocast(device, "bf16"):
                inputs = tuple(b[key] if pure else torch.cat((a[key], b[key]))
                               for key in ("latent", "goal", "proprio"))
                prediction = model(*inputs, sample_noise) if flow_only else model(*inputs)
                p = prediction[offline_count:]
                # Global offline mean remains correctly weighted when active
                # online states divide unevenly between the four GPUs.
                offline_loss = prediction.new_zeros(())
                if not pure:
                    offline_loss = supervised_loss(prediction[:offline_count], a["target"])
                    offline_loss *= offline_count*world/global_offline_count
                first_target = b["target"] if args.supervision_horizon == 1 else b["target"][:, 0]
                online_loss = first_step_loss(p, first_target, world, n) if real_count else p.sum()*0.
                tail_loss = prediction.new_zeros(())
                if args.supervision_horizon == 10:
                    tail_loss = masked_online_tail_loss(p, b["target"], b["target_mask"], world,
                                                        tail_samples) if real_count else p.sum()*0.
                online_total = online_loss + args.online_tail_weight*tail_loss
                loss = online_total if pure else offline_loss + args.online_weight*online_total
                anchor_loss = prediction.new_zeros(())
                anchor_mae = prediction.new_zeros(())
                selective_metrics = torch.zeros(4, device=device)
                if reference is not None:
                    with torch.no_grad():
                        ref_inputs = (b["latent"], b["goal"], b["proprio"])
                        original = (reference(*ref_inputs, sample_noise) if flow_only
                                    else reference(*ref_inputs))
                    if selective:
                        loss, selective_metrics = selective_online_correction_loss(
                            p, first_target, original, args.selective_correction_threshold_mps,
                            world, n if real_count else 0)
                    elif real_count:
                        anchor_loss = current_state_anchor_loss(p, original, world, n)
                        anchor_mae = (p.float()-original.float()).abs().mean((1, 2)).sum().detach()*world/n*2.
                    else:
                        anchor_loss = p.sum()*0.
                    loss = loss + args.anchor_weight*anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite online SFT loss")
            loss.backward()
            grad = nn.utils.clip_grad_norm_(parameters, 1.)
            if not bool(torch.isfinite(grad)):
                raise FloatingPointError("Non-finite online SFT gradient; optimizer update rejected")
            optimizer.step()
            step += 1
            deployed_p = student_feasible_target(p) if args.unclipped_training_output else p
            mae = (deployed_p[:, 0].float()-first_target).abs().mean(-1).sum().detach()*world/n*2
            if not real_count:
                mae.zero_()
            values = torch.stack((loss.detach(), offline_loss.detach(), online_loss.detach(), mae,
                                  anchor_loss.detach(), anchor_mae.detach(), tail_loss.detach(),
                                  *selective_metrics))
            if world > 1:
                dist.all_reduce(values)
                values /= world
            if rank == 0:
                metrics = dict(zip(("loss", "offline_bc", "online_first_bc", "online_first_mae_mps",
                                    "current_state_anchor_bc", "current_state_anchor_mae_mps",
                                    "online_tail_bc", "selective_correction_fraction",
                                    "selective_original_teacher_gap_mps", "selective_teacher_bc",
                                    "selective_preservation_bc"), values.tolist()))
                metrics.update(fresh_states=n, grad_norm=float(grad),
                               per_gpu_total_batch=real_count if pure else args.per_gpu_batch_size,
                               global_total_batch=n if pure else args.per_gpu_batch_size*world,
                               global_offline_batch=global_offline_count,
                               global_valid_batch=global_offline_count+n,
                               global_supervised_actions=int(fresh["target_mask"].sum())
                               if args.supervision_horizon == 10 else n,
                               rank0_offline_batch=offline_count, rank0_online_batch=real_count,
                               physics_step=int(message["physics_step"]), episode=int(message["episode"]),
                               teacher_speed_over_2_fraction=float((np.linalg.norm(fresh["target"], axis=-1)>1.0001).mean()),
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               elapsed_s=time.time()-started)
                metrics.update(message.get("diagnostics", {}))
                progress(step, metrics)
                for key, value in metrics.items():
                    writer.add_scalar("train/"+key, value, step)
                if step == args.start_step+1 or step % 10 == 0:
                    print(json.dumps({"step": step, **metrics}), flush=True)
                    writer.flush()
                if step % args.save_every == 0 or step == args.max_updates:
                    save(step)
                packet = weight_packet(expert, step)
                packet["metrics"] = metrics
                connection.send(packet)
    finally:
        if rank == 0:
            if step:
                save(step)
            if connection is not None:
                connection.close()
            if listener is not None:
                listener.close()
            writer.close()
        if world > 1:
            dist.destroy_process_group()


def smoke_client(args):
    """Same IPC/collective protocol as Isaac, including fewer states than GPUs."""
    connection = Client(str(args.socket), family="AF_UNIX")
    packet = connection.recv()
    generator = np.random.default_rng(123)
    for index, n in enumerate((5, 2)):
        batch = {"latent": generator.normal(size=(n, 4, 27, 5)).astype(np.float32),
                 "goal": generator.normal(size=(n, 4)).astype(np.float32),
                 "proprio": generator.normal(size=(n, 10)).astype(np.float32),
                 "target": np.zeros((n, 3), np.float32)}
        connection.send({"command": "update", "batch": batch, "student_version": packet["step"],
                         "physics_step": (index+1)*10, "episode": 0})
        previous = packet["weights"]
        packet = connection.recv()
        assert np.isfinite(packet["weights"]).all()
        assert not np.array_equal(previous, packet["weights"])
        print(f"IPC+DDP update={packet['step']} fresh_states={n}", flush=True)
    connection.send({"command": "finish"})
    print(connection.recv(), flush=True)
    connection.close()


def batch_benchmark(args):
    """Measure configured-batch forward/backward/Adam memory, not estimates."""
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if args.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload.pop("optimizer", None)
    expert = build_model(payload, freeze_policy_output_head=args.freeze_policy_output_head).to(device)
    model = policy_wrapper(expert)
    optimizer = torch.optim.AdamW([p for _, p in tail(expert)], lr=args.lr)
    pure = args.mode == "pure-online"
    if pure:
        count = args.per_gpu_batch_size
        batch = {"latent": torch.randn(count, 4, 27, 5, device=device),
                 "goal": torch.randn(count, 4, device=device),
                 "proprio": torch.randn(count, 10, device=device),
                 "target": torch.zeros(count, 3, device=device)}
    else:
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)["train"]
        generator = torch.Generator().manual_seed(123)
        batch = draw(cache, args.per_gpu_batch_size, generator, device)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, "bf16"):
            prediction = model(batch["latent"], batch["goal"], batch["proprio"])
            if pure:
                loss = first_step_loss(prediction, batch["target"], 1, count)
            else:
                loss = supervised_loss(prediction[:-32], batch["target"][:-32])
                loss += args.online_weight*first_step_loss(prediction[-32:], batch["target"][-32:, 0], 1, 32)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({"per_gpu_batch_size": args.per_gpu_batch_size,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                      "three_updates_s": time.time()-started, "loss": float(loss)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "smoke-client", "loss-test", "batch-benchmark"))
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--refill-enabled", action="store_true")
    parser.add_argument("--per-gpu-batch-size", type=int, default=512)
    parser.add_argument("--mode", choices=("pure-online", "mixed"), default="pure-online")
    parser.add_argument("--teacher-target-mode", choices=TARGET_MODES, default="raw")
    parser.add_argument("--anchor-weight", type=float, default=0.)
    parser.add_argument("--selective-correction-threshold-mps", type=float, default=0.,
                        help="Opt-in conditional online supervision; zero retains existing SFT")
    parser.add_argument("--freeze-policy-output-head", action="store_true",
                        help="Keep final norm/linear unchanged; optimize only policy blocks 7-8")
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--supervision-horizon", type=int, choices=(1, 10), default=1)
    parser.add_argument("--online-tail-weight", type=float, default=1.)
    parser.add_argument("--unclipped-training-output", action="store_true",
                        help="Regress pre-projection commands; deployment stays bounded at 2 m/s")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--cuda-memory-fraction", type=float, default=None)
    parser.add_argument("--online-weight", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction must be in (0,1]")
    if not np.isfinite(args.anchor_weight) or args.anchor_weight < 0:
        parser.error("Anchor weight must be finite and nonnegative")
    if not np.isfinite(args.online_tail_weight) or args.online_tail_weight < 0:
        parser.error("Tail weight must be finite and nonnegative")
    if args.command == "loss-test":
        prediction = torch.randn(5, 10, 3, requires_grad=True)
        first_step_loss(prediction, torch.zeros(5, 3), 1, 5).backward()
        assert prediction.grad[:, 0].abs().sum() > 0
        assert prediction.grad[:, 1:].abs().sum() == 0
        print("First-action-only loss gradient test passed", flush=True)
    elif args.command == "serve":
        serve(args)
    elif args.command == "batch-benchmark":
        batch_benchmark(args)
    else:
        smoke_client(args)


if __name__ == "__main__":
    main()
