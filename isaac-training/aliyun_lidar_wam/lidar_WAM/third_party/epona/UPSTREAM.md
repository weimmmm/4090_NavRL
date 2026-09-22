Source: https://github.com/Kevin-thu/Epona (models/stt.py, utils/rope_2d.py,
utils/embeddings.py), retrieved 2026-09-17. Licensed under the MIT license in
LICENSE. The upstream temporal and spatial attention blocks are used directly
by `lidar_wam/runner/epona_history.py`.

Local compatibility changes in `stt.py`: package-relative utility imports and
moving 2D rotary frequencies to the query device at use time. In `rope_2d.py`,
frequency construction stays on CPU rather than calling `.cuda()` during model
initialization. This supports the PPU environment and CPU smoke tests. Epona's
camera/trajectory-specific `SpatialTemporalTransformer` is retained for
reference but is not instantiated for NavRL LiDAR.
