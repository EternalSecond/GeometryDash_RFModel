"""CPU PPO training for the numerical simulator. Windows multiprocessing safe."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import signal
import time
from functools import partial
from pathlib import Path

# Each worker is tiny: avoid competing BLAS/OpenMP thread pools.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from gd_env import GeometryDashEnv


def make_env(level_length, max_episode_steps):
    return GeometryDashEnv(level_length=level_length, max_episode_steps=max_episode_steps)


def atomic_save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.zip')
    try:
        model.save(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    torch.set_num_threads(args.torch_threads)
    run = Path(args.log_dir) / time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    factories = [partial(make_env, args.level_length, args.max_episode_steps)
                 for _ in range(args.n_envs)]
    vec = model = None

    class Checkpoints(BaseCallback):
        def _on_step(self):
            if self.n_calls % math.ceil(args.checkpoint_steps / args.n_envs) == 0:
                atomic_save(self.model, run / f'checkpoint_{self.num_timesteps}.zip')
                atomic_save(self.model, args.model_path)
                paths = sorted(run.glob('checkpoint_*.zip'), key=lambda p: p.stat().st_mtime_ns)
                for old in paths[:-args.keep_checkpoints]:
                    old.unlink(missing_ok=True)
            return True

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    old_signal = signal.signal(signal.SIGTERM, stop)
    try:
        # spawn is explicit and works on Windows. Factories create independent envs.
        vec = (SubprocVecEnv(factories, start_method='spawn') if args.vec_env == 'subproc'
               else DummyVecEnv(factories))
        vec = VecMonitor(vec, filename=str(run / 'episodes.csv'),
                         info_keywords=('is_success', 'progress'))
        vec.seed(args.seed)  # SB3 assigns seed + rank, applied at the next reset.
        if args.resume:
            model = PPO.load(args.resume, env=vec, device='cpu', tensorboard_log=str(run))
        else:
            model = PPO('MlpPolicy', vec, device='cpu', verbose=1, seed=args.seed,
                        n_steps=args.n_steps, batch_size=args.batch_size,
                        learning_rate=3e-4, n_epochs=10, gamma=0.99,
                        ent_coef=0.01, target_kl=0.03,
                        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256])),
                        tensorboard_log=str(run))
        print(f'{args.n_envs} environments, {args.vec_env}, CPU, 20 numeric observations', flush=True)
        model.learn(total_timesteps=args.total_timesteps, callback=Checkpoints(),
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('Stopped; saving model.', flush=True)
    finally:
        try:
            if model is not None:
                atomic_save(model, args.model_path)
                print(f'Saved: {args.model_path}', flush=True)
        finally:
            signal.signal(signal.SIGTERM, old_signal)
            if vec is not None:
                vec.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--vec-env', choices=('subproc','dummy'), default='subproc')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--n-steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--level-length', type=float, default=200)
    parser.add_argument('--max-episode-steps', type=int, default=3000)
    parser.add_argument('--checkpoint-steps', type=int, default=50_000)
    parser.add_argument('--keep-checkpoints', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume')
    parser.add_argument('--model-path', default='gd_sim_ppo.zip')
    parser.add_argument('--log-dir', default='runs/sim')
    args = parser.parse_args()
    if min(args.n_envs, args.total_timesteps, args.n_steps, args.torch_threads,
           args.max_episode_steps, args.checkpoint_steps, args.keep_checkpoints) < 1:
        parser.error('Counts must be positive.')
    if args.batch_size < 2 or args.n_envs*args.n_steps < 2:
        parser.error('Batch and rollout size must be at least 2.')
    if args.n_envs*args.n_steps % args.batch_size:
        parser.error('batch-size must divide n-envs * n-steps.')
    if not math.isfinite(args.level_length) or args.level_length < 20:
        parser.error('level-length must be finite and >= 20.')
    if Path(args.model_path).suffix != '.zip':
        parser.error('model-path must end in .zip.')
    return args


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train(parse_args())
