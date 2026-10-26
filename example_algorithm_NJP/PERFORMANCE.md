# NJP training throughput

The training loop uses batched pairwise distances, skips unused Critic parameter gradients during the Actor update, and batches CUDA target-network updates. The target update retains the original multiply/multiply/add operations and rounding. The native distance function remains available as a regression oracle.

CUDA Graph is optional and disabled by default. To enable it on a CUDA device with a PyTorch version that supports fused Adam, run from the repository root:

```bash
python example_algorithm_NJP/train.py --fused_adam --cuda_graph --render_interval 0
```

`--no_cuda_graph` selects eager execution. The CPU default continues to omit the newer Adam constructor keywords. Environment configuration, rewards, network sizes, sampling order, and update frequency are unchanged. The example still defaults to 3 predators, 10 prey, and the `[-1, 1] × [-1, 1]` map.

## How CUDA Graph affects training

Each species captures one DDPG update, including both optimizer steps and target updates. Sampling, exploration noise, action inference, and environment steps stay outside the graph. Preparation restores networks, Adam state, and random states after its warmup and capture, so preparation does not count as training. A shared capture stream avoids accumulating cuBLAS workspaces on recapture.

Loading policies or training state invalidates the graph; changing batch shape rebuilds it. Checkpoints contain ordinary training state and work in either eager or graph mode. Graph buffers are not serialized. If preparation fails and rollback succeeds, training warns and falls back to eager execution. Replay failures and unsuccessful rollback propagate instead of attempting a potentially duplicate update. Checkpoint metadata and the final log report whether graphs are active, pending, disabled, or have fallen back.

## Measurements

Measured on 2026-10-26 with an NVIDIA GeForce RTX 3080, Intel Core i9-10900K, NVIDIA driver 580.178.04, Python 3.8.20, PyTorch 2.4.1+cu121, CUDA runtime 12.1, and NumPy 1.24.4. PyTorch/OMP/MKL used one CPU thread. Competing training was paused during measurement and resumed afterward.

The baseline is `main` at `bb9d46f64ee85f242e67ef11016798ecdbae8868`, which already includes batched action inference and fused Adam support. **All three modes use fused Adam**, so the comparison measures only this change.

| Scenario | Mode | Steps/s, median (min–max) | Relative to main |
| --- | --- | ---: | ---: |
| 3 predators / 10 prey, L=1 | main, fused Adam | 143.8 (143.3–147.6) | 1.00× |
| 3 predators / 10 prey, L=1 | Optimized eager, fused Adam | 166.2 (165.4–169.4) | 1.16× |
| 3 predators / 10 prey, L=1 | Optimized + CUDA Graph | 433.6 (424.6–434.9) | 3.02× |
| 12 predators / 40 prey, L=4 | main, fused Adam | 121.0 (120.0–122.0) | 1.00× |
| 12 predators / 40 prey, L=4 | Optimized eager, fused Adam | 158.5 (156.0–163.2) | 1.31× |
| 12 predators / 40 prey, L=4 | Optimized + CUDA Graph | 355.7 (353.8–359.5) | 2.94× |

These are full training-step measurements: action inference and noise, environment stepping and resets, rewards/statistics, replay insertion and sampling, and one network update per species per timed step. Rendering, console output, checkpoint writes, and startup are outside the timed interval. The harness starts fresh models and empty replay buffers from matching seeds, learns once each replay has 256 samples, warms up for 100 environment steps, then times 1,000 steps with CUDA synchronization at both boundaries. It runs three repetitions (seeds 171–173), rotating mode order and using a fresh process for every trial.

Both scenarios use batch size 256, hidden dimension 128, replay capacity 500,000 per species, gamma 0.95, tau 0.01, and exploration noise 0.05. The default scenario uses the repository's unchanged environment configuration and 120-step episodes. The larger scenario overrides only `L=4`, `n_p=12`, `n_e=40`, and both turn limits to 0.05 in memory, with 1,000-step episodes. These scenario comparisons are not a controlled sweep of population alone: map size and episode length also differ.

Graph preparation for both agents takes a median of 0.171 s (default) / 0.173 s (larger scenario), included in warmup and excluded from steady-state throughput. Median peak PyTorch allocated GPU memory during the timed interval is about 19.5 MiB for baseline/eager and 27.2 MiB for graphs. This allocator metric excludes CUDA context and other non-PyTorch allocations; it is not total process VRAM.

Population changes the balance between pairwise environment work, batched inference, replay insertion, and the fixed-batch network updates. Gains therefore depend on scenario and hardware. The measurements establish throughput and numerical equivalence for these runs, not faster convergence or improved policy quality.

### Individual trials (steps/s)

| Scenario | Mode | Seed 171 | Seed 172 | Seed 173 |
| --- | --- | ---: | ---: | ---: |
| default | baseline | 143.3 | 143.8 | 147.6 |
| default | eager | 169.4 | 165.4 | 166.2 |
| default | graph | 433.6 | 424.6 | 434.9 |
| swarm | baseline | 121.0 | 122.0 | 120.0 |
| swarm | eager | 158.5 | 163.2 | 156.0 |
| swarm | graph | 353.8 | 355.7 | 359.5 |

## Reproduce

With the repository and its Python 3.8/native dependencies installed, run from the repository root on an otherwise idle CUDA GPU:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_njp.py \
  --baseline-ref bb9d46f64ee85f242e67ef11016798ecdbae8868 \
  --scenario default --output /tmp/njp-default.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_njp.py \
  --baseline-ref bb9d46f64ee85f242e67ef11016798ecdbae8868 \
  --scenario swarm --output /tmp/njp-swarm.json
```

The script reads the baseline Agent and environment implementations directly from Git. It generates its own training data, needs no downloaded checkpoint, and writes per-trial timing, graph preparation, memory, configuration, and state hashes to JSON. It does not edit configuration or manage other processes.

## Correctness checks

For each seed and scenario, all three modes produced identical final state hashes after 1,100 steps, including episode resets. The hash covers networks, targets, Adam state, replay data, observations, physical state, reward statistics, update counts, and NumPy/PyTorch/CUDA/Python/environment random states; only the intentional Adam backend flags are excluded.

The focused suite passed **144 tests, with 1 expected skip** (CUDA Graph on CPU). It checks each of 100 consecutive updates against the original arithmetic, CPU/legacy Adam compatibility, exact contact and periodic-boundary distances, 100-step environment trajectories with different maps/dynamics/obstacles, shape recapture, rollback failures, and checkpoint resume between eager and graph execution.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_njp_performance.py tests/test_njp_finetune.py \
  tests/test_njp_optimization.py tests/test_pps_observation.py \
  tests/test_pps_collision.py tests/test_pps_distance.py
```
