#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/wam_trining/trining/lidar_WAM
task_python=/mnt/workspace/LaGen/.venv-ppu/bin/python
task_data=/mnt/workspace/wam_trining/data/navrl_static_100k/lagen_cache
task_run=outputs/world_circular_causal_8h
task_baseline=outputs/world_circular_causal_full
task_reports=outputs/representative_baseline

if [[ ${1:-} != --eval-only ]]; then
  mkdir -p "$task_run"
  if [[ ! -e "$task_run/latest.pt" ]]; then
    cp "$task_baseline/latest.pt" "$task_run/latest.pt"
    ln -s ../world_circular_causal_full/best.pt "$task_run/best.pt"
    cp "$task_baseline/best_metrics.json" "$task_run/best_metrics.json"
    cp "$task_baseline/history.json" "$task_run/history.json"
    cp "$task_baseline/config.json" "$task_run/config.json"
  fi

  "$task_python" scripts/train_unet_8h.py train-world \
    --resume --steps 1000000 --max-hours 8 \
    --batch-size 1664 --lr 3e-5 --eval-every 200 --log-every 25 \
    --data-root "$task_data"
fi

for task_checkpoint in best latest; do
  "$task_python" scripts/evaluate_representative.py evaluate \
    --split val --method lagen_unet --unet-run world_circular_causal_8h \
    --checkpoint-name "${task_checkpoint}.pt" --report-tag "8h_${task_checkpoint}" \
    --data-root "$task_data"
done

task_selected=$("$task_python" - <<'PY'
import json
from pathlib import Path
root = Path('outputs/representative_baseline')
scores = {name: json.loads((root / f'val_lagen_unet_8h_{name}.json').read_text())['summary']['all']['chamfer_m']
          for name in ('best', 'latest')}
print(min(scores, key=scores.get))
PY
)

"$task_python" scripts/evaluate_representative.py evaluate \
  --split test --method lagen_unet --unet-run world_circular_causal_8h \
  --checkpoint-name "${task_selected}.pt" --report-tag 8h_selected \
  --data-root "$task_data"

"$task_python" - "$task_selected" <<'PY'
import json
import sys
from pathlib import Path
root = Path('outputs/representative_baseline')
selected = sys.argv[1]
def report(name):
    return json.loads((root / name).read_text())
old_val = report('val_lagen_unet.json')['summary']['all']
old_test = report('test_lagen_unet.json')['summary']['all']
new_val = report(f'val_lagen_unet_8h_{selected}.json')['summary']['all']
new_test = report('test_lagen_unet_8h_selected.json')['summary']['all']
velocity_val = report('val_velocity_pose.json')['summary']['all']
velocity_test = report('test_velocity_pose.json')['summary']['all']
result = {
    'selected_by_validation_chamfer': selected,
    'baseline_5000': {'val': old_val, 'test': old_test},
    'trained_8h': {'val': new_val, 'test': new_test},
    'velocity_pose': {'val': velocity_val, 'test': velocity_test},
    'test_chamfer_change_vs_5000': new_test['chamfer_m'] / old_test['chamfer_m'] - 1,
    'test_chamfer_change_vs_velocity_pose': new_test['chamfer_m'] / velocity_test['chamfer_m'] - 1,
    'best_metrics': report('../world_circular_causal_8h/best_metrics.json'),
}
path = root / 'unet_8h_final_comparison.json'
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'final_report': str(path), **result}), flush=True)
PY
