# Geometry Dash RL

A lightweight numerical Gymnasium environment and PPO training project inspired by Geometry Dash. The simulator runs without screenshots or a game window.

## Requirements

Python 3.11 or 3.12. Create a virtual environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Training

```powershell
.\.venv\Scripts\python.exe train.py --n-envs 8 --total-timesteps 1000000
.\.venv\Scripts\python.exe train.py --n-envs 8 --resume gd_sim_ppo.zip
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs/sim
```

Training uses CPU PPO with MlpPolicy, SubprocVecEnv and VecMonitor. Compare `--vec-env dummy` with `--vec-env subproc`; multiprocessing overhead can dominate a small simulation. Use Ctrl+C for a graceful stop and save. Forced process termination can bypass saving.

Checkpoints are written every 50,000 combined environment steps; the last five are retained. The latest model is saved as gd_sim_ppo.zip. Logs are stored under runs/sim. Resume restores the policy and optimizer; unfinished rollouts and environment state are not restored.

## Environment

`gd_env.py` implements automatic horizontal movement, gravity, jumping, triangular spikes, blocks, landing and randomly generated easy levels. Actions are release (0) and hold jump (1). Observations contain 20 normalized numbers describing the cube and the nearest three obstacles. Rewards combine horizontal progress, a collision penalty and a completion bonus.

Custom obstacle maps use rows `[x, width, height, kind]`, where kind 0 is a spike and kind 1 is a block. Pass an array through the `obstacles` constructor argument. Seeded resets are reproducible.

This is simplified physics, not a recreation of the commercial game's levels or engine. Simulator policies do not automatically transfer to the real game.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_simulator -v
```

The simulator tests cover the Gymnasium contract, physics, collisions, deterministic seeds, 10,000 simulation steps and subprocess vectorization.

## Optional live-game integration

`gd_live_env.py` and `train_live.py` contain the experimental Windows screen-capture integration. It requires a separately installed game, windowed mode, Auto-Retry enabled and Practice disabled. CUDA PyTorch is optional and must be installed separately for GPU training. See `train_live.py --help` for options. Calibration is machine-specific:

```powershell
.\.venv\Scripts\python.exe train_live.py --calibrate
.\.venv\Scripts\python.exe train_live.py --diagnose
.\.venv\Scripts\python.exe train_live.py --device cpu
```

Hold F9 to stop live training. Do not change window geometry after calibration. Live death detection is heuristic and does not recognize every possible game state. `test_offline.py` contains additional live-integration mocks and CUDA checks requiring a compatible GPU.

## Repository hygiene

Virtual environments, IDE settings, screenshots, calibration coordinates, model weights and training logs are excluded. requirements-lock.txt records the original Windows CUDA environment; requirements.txt is the portable starting point. No game assets are distributed.
