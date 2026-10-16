"""Teacher-only diagnostic commands; never imported by the student adapter."""
import torch


def teacher_ego_chunk(world_velocity, quaternion_wxyz, mode, physics_step,
                      bootstrap_steps=10, bootstrap_speed=1., horizon=10):
    if (world_velocity.ndim != 2 or world_velocity.shape[-1] != 3
            or quaternion_wxyz.shape != (len(world_velocity), 4)
            or mode not in ("native", "student-norm", "student-box") or horizon != 10
            or min(physics_step, bootstrap_steps) < 0):
        raise ValueError("Invalid teacher benchmark command contract")
    if not bool(torch.isfinite(world_velocity).all() & torch.isfinite(quaternion_wxyz).all()):
        raise ValueError("Non-finite teacher input")
    norm = quaternion_wxyz.norm(dim=-1, keepdim=True)
    if bool((norm < 1e-6).any()):
        raise ValueError("Zero teacher quaternion")
    w, x, y, z = (quaternion_wxyz/norm).unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = yaw.cos(), yaw.sin()
    ego = torch.stack((c*world_velocity[:, 0]+s*world_velocity[:, 1],
                       -s*world_velocity[:, 0]+c*world_velocity[:, 1],
                       world_velocity[:, 2]), dim=-1)
    if mode == "student-norm":
        ego = ego/(ego.norm(dim=-1, keepdim=True)/2.).clamp_min(1.)
    elif mode == "student-box":
        ego = ego.clamp(-2., 2.)
    if physics_step < bootstrap_steps:
        ego = torch.zeros_like(ego)
        ego[:, 0] = bootstrap_speed
    # For K=10 the teacher holds this instantaneous ego command under the
    # SAME current-yaw adapter as the student. This is not a PPO future rollout.
    return ego[:, None].expand(-1, horizon, -1).clone()
