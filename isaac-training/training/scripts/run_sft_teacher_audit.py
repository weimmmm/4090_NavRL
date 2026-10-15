"""Matched teacher-ceiling audit followed by the SFT candidate's reproduction.

Run on the HOST after the 16-route wiring job succeeds. No learner, collection
dataset, replay or student-control modification is launched by this script.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from online_sft_jobs import atomic_json, docker_command, read_json, run_logged
from run_action_dagger_sft import HOST, ROOT, PROJECT, TRAIN, DEPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-pid', type=int, required=True)
    args = parser.parse_args()
    out = PROJECT/'outputs/action_sft_teacher_audit_20261015'
    host_out = HOST/'training/lidar_WAM/outputs/action_sft_teacher_audit_20261015'
    smoke = host_out/'smoke_feasible_k1/metrics.json'
    deadline = time.monotonic()+900
    while not smoke.is_file():
        os.kill(args.smoke_pid, 0)  # Confirmed handle, not an inferred live status.
        if time.monotonic() > deadline:
            raise TimeoutError('Teacher wiring job did not complete; no formal audit started')
        time.sleep(1)
    wiring = read_json(smoke)
    if (wiring.get('num_envs') != 16
            or wiring.get('policy') != 'privileged_ppo_online_sft_teacher'
            or wiring.get('teacher_query_interval_physics_steps') != 1
            or wiring.get('teacher_velocity_domain') != 'student-norm'):
        raise RuntimeError('Teacher wiring output does not match the requested adapter')
    # A completed metric is followed by os._exit in the evaluator. The next
    # process will not share the physical card with a still-running smoke app.
    for _ in range(120):
        try:
            os.kill(args.smoke_pid, 0)
        except ProcessLookupError:
            break
        stat = Path(f'/proc/{args.smoke_pid}/stat')
        if stat.is_file() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            break
        time.sleep(.5)
    else:
        raise RuntimeError('Completed smoke app has not exited; no competing GPU job started')
    source = PROJECT/'outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt'
    candidate = PROJECT/'outputs/action_sft_original_selective_headfreeze_20261015/best_closed_loop.pt'
    jobs = [('teacher_native_k1', 'native', 1, source),
            ('teacher_feasible_k1', 'student-norm', 1, source),
            ('teacher_feasible_k10', 'student-norm', 10, source),
            ('student_best_repeat', None, 10, candidate)]
    environment = {'ISAAC_ACTIVE_GPU': '0', 'PYTHONUNBUFFERED': '1',
                   'PYTHONPATH': f'{PROJECT}:{DEPS}',
                   'NAVRL_ASSETS_ROOT': 'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0'}
    results = {}
    try:
        for name, mode, interval, checkpoint in jobs:
            folder, job = host_out/name, out/name
            folder.mkdir(exist_ok=False)
            atomic_json(host_out/'status.json', {'stage':'running', 'host_pid':os.getpid(),
                        'current_job':name, 'completed':list(results), 'gpu':4,
                        'student_training_started':False, 'new_training_data_collected':False})
            entry = 'evaluate_online_sft_teacher.py' if mode else 'wam_closed_loop_eval.py'
            command = ['/opt/navrl/bin/isaac-python', f'training/scripts/{entry}',
                       'headless=true', 'device=cuda:0', 'seed=20261014', 'env.num_envs=128',
                       'env.max_episode_length=2500', 'env.num_obstacles=350', 'env_dyn.num_obstacles=0',
                       'env.scene_path=training/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt',
                       f'wam_eval.checkpoint={checkpoint}', 'wam_eval.bootstrap_forward_speed_mps=1.0',
                       'wam_eval.bootstrap_action_chunks=1', 'wam_eval.goal_stop_adapter.enabled=true',
                       'wam_eval.goal_stop_adapter.enter_distance_m=0.5',
                       'wam_eval.goal_stop_adapter.enter_speed_mps=0.2', 'wam_eval.safety_filter.enabled=false',
                       f'wam_eval.metrics_path={job}/metrics.json', f'wam_eval.trajectory_path={job}/trajectories.npz']
            if mode:
                command += [f'+wam_eval.teacher_checkpoint={PROJECT}/outputs/privileged_ppo_checkpoint_27000.pt',
                            f'+wam_eval.teacher_velocity_domain={mode}',
                            f'+wam_eval.teacher_query_interval_physics_steps={interval}']
            run_logged(docker_command('navrl-collect-gpu4',
                       ['timeout','--signal=TERM','--kill-after=15s','900', *command],
                       environment, ROOT), folder/'worker.log', timeout=960)
            metrics = read_json(folder/'metrics.json')
            if (metrics.get('num_envs') != 128 or metrics.get('max_physics_steps') != 2500
                    or metrics.get('physics_dt_s') != .016
                    or metrics.get('map_sha256') != '2f97c4649531c19ebd7402b57408210e60743401da82e13c6f498e46bcc74753'
                    or metrics.get('replan_interval_steps') != interval):
                raise RuntimeError(f'{name}: unmatched geometry/horizon/action timing')
            results[name] = {'counts':metrics['counts'], 'rates':metrics['rates'],
                             'checkpoint':metrics['checkpoint'], 'checkpoint_step':metrics['checkpoint_step']}
            atomic_json(host_out/'results.json', results)
        atomic_json(host_out/'status.json', {'stage':'complete','results':results,'gpu':4})
    except BaseException as error:
        atomic_json(host_out/'status.json', {'stage':'failed','results':results,
                                           'error':f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
