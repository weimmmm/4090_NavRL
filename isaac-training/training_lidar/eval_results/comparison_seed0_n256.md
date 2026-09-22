# Common-scene checkpoint comparison

Both checkpoints were evaluated with deterministic policy actions on terrain seed 0,
using 256 parallel routes, 350 static obstacles, 80 dynamic obstacles, and a
2,200-step horizon. Both ray casters intersect `/World/ground`.

| Policy | Success | Collision | Out of bounds | Timeout | Mean successful steps | Mean successful path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original NavRL PPO | 0/256 (0.00%) | 214/256 (83.59%) | 3/256 (1.17%) | 39/256 (15.23%) | N/A | N/A |
| Range image + circular CNN | 137/256 (53.52%) | 117/256 (45.70%) | 1/256 (0.39%) | 1/256 (0.39%) | 1806.82 | 52.48 m |

Each checkpoint uses its native observation shape: the baseline uses the original
4 m, 36 x 4 distance array and ordinary CNN; the new policy uses the 10 m raw
108 x 18 scan, a 36 x 6 range image, and horizontal circular convolutions.

The baseline was trained while its ray caster targeted only
`/World/defaultGroundPlane`, so static terrain obstacles were absent from its
training observations. Evaluating it against `/World/ground` is intentionally a
distribution shift, but it is required for a physically meaningful comparison in
which both policies can observe the obstacles they may collide with.

Raw outputs:

- `baseline_seed0_n256.json`
- `range_image_seed0_n256.json`

## Static-only evaluation

The range-image checkpoint was also evaluated on the same 256 fixed routes with
350 static obstacles and no dynamic obstacles. It achieved 193/256 successes
(75.39%), 58/256 collisions (22.66%), no out-of-bounds failures, and 5/256
timeouts (1.95%). The mean successful trajectory took 1834.48 steps and covered
53.77 m.

Compared with the 350-static plus 80-dynamic evaluation, removing dynamic
obstacles increased success by 21.88 percentage points and reduced collision by
23.05 percentage points.

Additional raw output:

- `range_image_static350_seed0_n256.json`
