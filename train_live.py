"""Calibration, diagnostics and PPO. Hold F9 to stop and save."""
from __future__ import annotations
import argparse
import json
import os
import signal
import time
from pathlib import Path
import cv2
import numpy as np
from gd_live_env import GameFocusLost, GeometryDashEnv, windows_api, foreground_title, client_region, crop


def countdown():
    print('Switch to Geometry Dash. Starting in 5 seconds.', flush=True)
    for n in range(5, 0, -1):
        print(n, flush=True)
        time.sleep(1)


def calibrate(args):
    import mss
    api = windows_api()
    countdown()
    print('F8 when ATTEMPT is fully visible; F9 cancels.', flush=True)
    with mss.mss() as sct:
        deadline = time.perf_counter()+60
        while time.perf_counter() < deadline:
            if api.GetAsyncKeyState(0x78) & 0x8000:
                return
            if api.GetAsyncKeyState(0x77) & 0x8000:
                if args.window_title.lower() not in foreground_title(api).lower():
                    raise RuntimeError('Focus the game before pressing F8.')
                region = args.region or client_region(api)
                left,top,width,height = region
                frame = np.asarray(sct.grab(dict(left=left,top=top,width=width,height=height))).copy()
                break
            time.sleep(1/120)
        else:
            raise TimeoutError('F8 was not pressed within 60 seconds.')
    roi = args.attempt_roi
    if roi is None:
        print('Select only ATTEMPT (without the number); Enter confirms, C cancels.', flush=True)
        try:
            roi = tuple(map(int,cv2.selectROI('Select ATTEMPT',frame[:,:,:3],showCrosshair=True)))
        finally:
            cv2.destroyAllWindows()
        if min(roi[2:]) <= 0:
            print('Calibration cancelled; previous configuration preserved.')
            return
    gray = cv2.cvtColor(crop(frame,roi),cv2.COLOR_BGRA2GRAY)
    if np.count_nonzero(cv2.Canny(gray,80,160)) < 20:
        raise ValueError('Empty ROI; select the ATTEMPT word.')
    path = Path(args.template)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),gray) or not cv2.imwrite(str(path.with_suffix('.screen.png')),frame):
        raise OSError('Failed to save calibration images.')
    Path(args.config).write_text(json.dumps(dict(region=list(region),attempt_roi=list(roi),template=str(path),window_title=args.window_title,stall_seconds=args.stall_seconds,reset_timeout=args.reset_timeout,attempt_absence_seconds=args.attempt_absence_seconds,max_episode_steps=args.max_episode_steps),indent=2),encoding='utf-8')
    print(f'Saved {args.config}, {path} and full screenshot. Run GD Diagnose next.')


def atomic_save(model,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.stem+'.tmp.zip')
    try:
        model.save(str(temporary))
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def make_env(args):
    return GeometryDashEnv(region=tuple(args.region),attempt_roi=tuple(args.attempt_roi),template_path=args.template,control_hz=args.control_hz,capture_hz=args.capture_hz,match_threshold=args.match_threshold,death_pixel=tuple(args.death_pixel) if args.death_pixel else None,window_title=args.window_title,stall_seconds=args.stall_seconds,reset_timeout=args.reset_timeout,attempt_absence_seconds=args.attempt_absence_seconds,max_episode_steps=args.max_episode_steps)


def diagnose(args):
    env = make_env(args)
    try:
        countdown()
        env.reset()
        last_log = 0
        while True:
            _,reward,terminated,truncated,info = env.step(0)
            done = terminated or truncated
            if done or time.perf_counter()-last_log > 1:
                print(f"score={info['attempt_score']:.3f}, motion={info['motion']:.2f}, reward={reward}, death={info['death_reason'] or '-'}",flush=True)
                last_log = time.perf_counter()
            if done:
                env.reset()
    except KeyboardInterrupt:
        print('Diagnostics stopped.')
    finally:
        env.close()


def policy_settings():
    """NatureCNN plus separate deep 512-unit actor and critic heads."""
    return dict(normalize_images=True, net_arch=dict(pi=[512, 512], vf=[512, 512]))


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack,VecTransposeImage
    torch.set_num_threads(1)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Select cpu or install a CUDA PyTorch build.')
    print(f'Device: {device}',flush=True)
    run = Path(args.log_dir)/time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True,exist_ok=True)
    (run/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    raw = make_env(args)
    vec = model = None

    last_save_time = time.monotonic()
    last_save_step = 0

    def save_checkpoint(pause=True):
        nonlocal last_save_time, last_save_step
        if model is None:
            return
        was_paused = raw._paused
        if pause:
            raw.pause()
        try:
            checkpoint = run/f'checkpoint_{model.num_timesteps}.zip'
            atomic_save(model, checkpoint)
            atomic_save(model, args.model_path)
            last_save_time = time.monotonic()
            last_save_step = model.num_timesteps
            snapshots = sorted(run.glob('checkpoint_*.zip'), key=lambda p: p.stat().st_mtime_ns)
            for old in snapshots[:-args.keep_checkpoints]:
                old.unlink(missing_ok=True)
            print(f'Checkpoint saved at step {last_save_step}', flush=True)
        finally:
            if pause and not was_paused:
                raw.resume()

    def save_while_waiting():
        # Input is released during reset. Do not toggle Esc in an unknown menu.
        if time.monotonic()-last_save_time >= args.checkpoint_seconds:
            save_checkpoint(pause=False)

    raw.on_wait = save_while_waiting

    class GameCallback(BaseCallback):
        episodes = 0
        def _on_rollout_start(self):
            raw.resume()
        def _on_rollout_end(self):
            raw.pause()
        def _on_step(self):
            info = self.locals['infos'][0]
            self.logger.record('game/attempt_score',info.get('attempt_score',0))
            self.logger.record_mean('game/step_seconds',info.get('step_seconds',0))
            episode_due = False
            if self.locals['dones'][0]:
                self.episodes += 1
                episode_due = self.episodes % args.checkpoint_episodes == 0
            if (episode_due or self.num_timesteps-last_save_step >= args.checkpoint_steps
                    or time.monotonic()-last_save_time >= args.checkpoint_seconds):
                save_checkpoint()
            return True

    def stop_signal(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    previous_signal = signal.signal(signal.SIGTERM, stop_signal)
    try:
        vec = DummyVecEnv([lambda: Monitor(raw,str(run/'train.monitor.csv'))])
        vec = VecTransposeImage(VecFrameStack(vec,n_stack=4,channels_order='last'))
        if args.resume:
            model = PPO.load(args.resume,env=vec,device=device,tensorboard_log=str(run))
        else:
            model = PPO('CnnPolicy',vec,device=device,verbose=1,seed=args.seed,learning_rate=0.00025,n_steps=args.n_steps,batch_size=64,gamma=0.99,gae_lambda=0.95,n_epochs=4,clip_range=0.2,ent_coef=0.01,vf_coef=0.5,max_grad_norm=0.5,target_kl=0.03,tensorboard_log=str(run),policy_kwargs=policy_settings())
        last_save_step = model.num_timesteps
        print(f'Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}', flush=True)
        if args.resume:
            print('Resumed checkpoint architecture and optimizer (new architecture applies to fresh training).', flush=True)
        countdown()
        model.learn(total_timesteps=args.total_timesteps,callback=GameCallback(),reset_num_timesteps=not bool(args.resume),tb_log_name='PPO')
    except (KeyboardInterrupt, GameFocusLost) as error:
        print(f'Training stopped: {error}')
    finally:
        try:
            try:
                raw.release_keys()
            except Exception as error:
                print(f'Key release failed: {error}')
            try:
                raw.pause(check_stop=False)
            except (Exception,KeyboardInterrupt) as error:
                print(f'Could not pause game: {error}')
            if model is not None:
                try:
                    atomic_save(model,args.model_path)
                    print(f'Model saved: {args.model_path}')
                except OSError as error:
                    fallback = run/'emergency.zip'
                    atomic_save(model, fallback)
                    print(f'Primary save failed ({error}); model saved to {fallback}')
        finally:
            signal.signal(signal.SIGTERM, previous_signal)
            (vec if vec is not None else raw).close()


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--calibrate',action='store_true')
    mode.add_argument('--diagnose',action='store_true')
    parser.add_argument('--config',default='game_config.json')
    parser.add_argument('--region',type=int,nargs=4)
    parser.add_argument('--attempt-roi',type=int,nargs=4)
    parser.add_argument('--template',default='attempt.png')
    parser.add_argument('--window-title',default='Geometry Dash')
    parser.add_argument('--death-pixel',type=int,nargs=5)
    parser.add_argument('--match-threshold',type=float,default=0.85)
    parser.add_argument('--control-hz',type=float,default=30)
    parser.add_argument('--capture-hz',type=float,default=60)
    parser.add_argument('--total-timesteps',type=int,default=1000000)
    parser.add_argument('--n-steps',type=int,choices=(512,1024),default=512)
    parser.add_argument('--checkpoint-episodes',type=int,default=25)
    parser.add_argument('--checkpoint-steps',type=int,default=2048)
    parser.add_argument('--checkpoint-seconds',type=float,default=120)
    parser.add_argument('--keep-checkpoints',type=int,default=5)
    parser.add_argument('--stall-seconds',type=float,default=8)
    parser.add_argument('--reset-timeout',type=float,default=30)
    parser.add_argument('--attempt-absence-seconds',type=float,default=0.05)
    parser.add_argument('--max-episode-steps',type=int,default=18000)
    parser.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume')
    parser.add_argument('--model-path',default='gd_ppo_agent.zip')
    parser.add_argument('--log-dir',default='runs')
    args = parser.parse_args()
    if min(args.total_timesteps,args.checkpoint_episodes,args.checkpoint_steps,args.keep_checkpoints,args.max_episode_steps) < 1 or min(args.checkpoint_seconds,args.stall_seconds,args.reset_timeout,args.attempt_absence_seconds) <= 0:
        parser.error('Timesteps and checkpoint interval must be positive.')
    if Path(args.model_path).suffix != '.zip' or Path(args.template).suffix.lower() != '.png':
        parser.error('Model must end in .zip and template in .png.')
    if args.calibrate:
        calibrate(args)
        return
    if Path(args.config).is_file():
        settings = json.loads(Path(args.config).read_text(encoding='utf-8'))
        for name in ('region','attempt_roi'):
            if getattr(args,name) is None:
                setattr(args,name,settings[name])
        # Explicit command-line options override calibration defaults.
        import sys
        for name in ('template','window_title'):
            flag = '--'+name.replace('_','-')
            if not any(x == flag or x.startswith(flag+'=') for x in sys.argv[1:]):
                setattr(args,name,settings.get(name,getattr(args,name)))
    if args.region is None or args.attempt_roi is None:
        parser.error('Run GD Calibrate first, or supply --region and --attempt-roi.')
    if not Path(args.template).is_file():
        parser.error('Template missing. Run GD Calibrate first.')
    (diagnose if args.diagnose else train)(args)


if __name__ == '__main__':
    main()

