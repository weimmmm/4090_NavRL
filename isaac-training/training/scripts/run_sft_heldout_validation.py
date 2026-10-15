"""One acceptance test on saved routes 128:256, never used for SFT selection."""
import hashlib
import os
from pathlib import Path

from online_sft_jobs import atomic_json, docker_command, read_json, run_logged
from run_action_dagger_sft import HOST, ROOT, PROJECT, TRAIN, PY, DEPS


def main():
    out = PROJECT/'outputs/action_sft_heldout_routes_20261015'
    host_out = HOST/'training/lidar_WAM/outputs/action_sft_heldout_routes_20261015'
    host_out.mkdir(exist_ok=False)
    original = PROJECT/'outputs/action_dit_final_lee_ego_history1_5m_v3/best_closed_loop.pt'
    candidate = PROJECT/'outputs/action_sft_original_selective_headfreeze_20261015/best_closed_loop.pt'
    host_candidate = HOST/candidate.relative_to(ROOT)
    candidate_sha = hashlib.sha256(host_candidate.read_bytes()).hexdigest()
    atomic_json(host_out/'candidate_locked_before_validation.json', {
        'checkpoint':str(candidate), 'sha256':candidate_sha,
        'selection_routes':'0:128', 'acceptance_routes':'128:256',
        'candidate_selection_must_not_use_acceptance_scores':True})
    environment = {'ISAAC_ACTIVE_GPU':'0', 'PYTHONUNBUFFERED':'1',
        'PYTHONPATH':f'{PROJECT}:{DEPS}',
        'NAVRL_ASSETS_ROOT':'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0'}
    results = {}
    try:
        for name, checkpoint in (('original',original), ('sft_best_locked',candidate)):
            folder, job = host_out/name, out/name
            folder.mkdir()
            atomic_json(host_out/'status.json', {'stage':'running','host_pid':os.getpid(),
                                                'current_job':name,'completed':list(results),'gpu':4})
            command = ['/opt/navrl/bin/isaac-python','training/scripts/wam_closed_loop_eval.py',
                'headless=true','device=cuda:0','seed=20261014','env.num_envs=128',
                'env.max_episode_length=2500','env.num_obstacles=350','env_dyn.num_obstacles=0',
                'env.fixed_route_start_index=128',
                'env.scene_path=training/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt',
                f'wam_eval.checkpoint={checkpoint}','wam_eval.bootstrap_forward_speed_mps=1.0',
                'wam_eval.bootstrap_action_chunks=1','wam_eval.goal_stop_adapter.enabled=true',
                'wam_eval.goal_stop_adapter.enter_distance_m=0.5',
                'wam_eval.goal_stop_adapter.enter_speed_mps=0.2','wam_eval.safety_filter.enabled=false',
                f'wam_eval.metrics_path={job}/metrics.json',f'wam_eval.trajectory_path={job}/trajectories.npz']
            run_logged(docker_command('navrl-collect-gpu4',
                ['timeout','--signal=TERM','--kill-after=15s','900',*command], environment,ROOT),
                folder/'worker.log',timeout=960)
            metrics=read_json(folder/'metrics.json')
            if (metrics.get('fixed_route_start_index')!=128 or metrics.get('num_envs')!=128
                    or metrics.get('max_physics_steps')!=2500 or metrics.get('replan_interval_steps')!=10):
                raise RuntimeError('Acceptance route range or execution settings changed')
            results[name]={'counts':metrics['counts'],'rates':metrics['rates'],
                           'checkpoint_step':metrics['checkpoint_step']}
            atomic_json(host_out/'results.json',results)
        if hashlib.sha256(host_candidate.read_bytes()).hexdigest()!=candidate_sha:
            raise RuntimeError('Candidate changed during held-out validation')
        # Inspect the ACTUAL saved initial states, routes and quaternion rows,
        # not merely the requested CLI offset. GPU execution is unnecessary.
        check = r'''
import json,numpy as np,torch,sys
from pathlib import Path
sys.path.insert(0,'/workspace/NavRL/isaac-training/training/scripts')
from summarize_sft_target_ablation import load_evaluation,paired_changes
r=Path(sys.argv[1]);scene=torch.load('/workspace/NavRL/isaac-training/training/datasets/wam_40x40_obs200_350_settled_5m/environments/map40_obs0350.pt',map_location='cpu',weights_only=False)
for name in ['original','sft_best_locked']:
 z=np.load(r/name/'trajectories.npz',allow_pickle=True)
 assert np.array_equal(z['source_route_index'],np.arange(128,256))
 assert np.array_equal(z['start_position'],scene['start_position'][128:256].numpy())
 assert np.array_equal(z['target_position'],scene['goal_position'][128:256].numpy())
 np.testing.assert_allclose(z['root_state_wxyz'][0,:,:3],z['start_position'],atol=1e-5,rtol=0)
 q=z['root_state_wxyz'][0,:,3:7];expected=scene['initial_orientation_wxyz'][128:256].numpy()
 np.testing.assert_allclose(abs((q*expected).sum(-1)),np.ones(128),atol=1e-5,rtol=0)
changes=paired_changes(load_evaluation(r/'original'),load_evaluation(r/'sft_best_locked'))
(r/'paired_acceptance.json').write_text(json.dumps(changes,indent=2)+'\n')
print(json.dumps(changes))
'''
        run_logged(docker_command(TRAIN,[PY,'-c',check,out],{},ROOT),host_out/'route_audit.log',timeout=120)
        atomic_json(host_out/'status.json', {'stage':'complete','results':results,
                                            'paired':read_json(host_out/'paired_acceptance.json')})
    except BaseException as error:
        atomic_json(host_out/'status.json', {'stage':'failed','results':results,
                                            'error':f'{type(error).__name__}: {error}'})
        raise


if __name__=='__main__':main()
