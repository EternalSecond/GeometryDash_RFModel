"""Offline checks: no keyboard input and no live game required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gd_live_env import GeometryDashEnv, crop
from train_live import atomic_save, policy_settings


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.env = GeometryDashEnv.__new__(GeometryDashEnv)
        e = self.env
        e.roi = (50,30,120,35)
        word = np.zeros((35,120),np.uint8)
        cv2.putText(word,'ATTEMPT',(2,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,255,2)
        e.template = cv2.Canny(word,80,160)
        e.threshold = 0.85
        e.last_score = 0.0
        e.capture_period = 1/60
        e.attempt_absence_seconds = 0.05
        e._pixel_ready = False
        e._armed = False
        e._absent_since = None
        e._pixel_hits = 0
        e.death_pixel = None
        e.pixel_tolerance = 12
        self.blank = np.zeros((140,260,4),np.uint8)
        self.visible = self.blank.copy()
        self.visible[30:65,50:170,:3] = word[...,None]

    def test_retry_is_single_event_after_absence(self):
        e = self.env
        self.assertFalse(e._is_dead(self.visible,0)[0])
        self.assertFalse(e._is_dead(self.blank,1)[0])
        self.assertFalse(e._is_dead(self.blank,1.3)[0])
        self.assertEqual(e._is_dead(self.visible,1.4),(True,'auto_retry'))
        e._paused, e._needs_reset, e._restart_obs = False,True,e._observation(self.visible)
        e.release_keys = Mock()
        obs,_ = e.reset()
        self.assertEqual(obs.shape,(84,84,1))
        self.assertFalse(e._is_dead(self.visible,1.5)[0])

    def test_shifted_word(self):
        shifted = np.roll(self.visible,20,axis=1)
        self.assertTrue(self.env._attempt_visible(shifted))

    def test_pixel_requires_two_frames_and_rgb_conversion(self):
        e = self.env
        e._armed = True
        e.death_pixel = (0,0,10,20,30)
        e._is_dead(self.blank, -0.1)
        frame = self.blank.copy()
        frame[0,0,:3] = [30,20,10]
        self.assertFalse(e._is_dead(frame,0)[0])
        self.assertEqual(e._is_dead(frame,0.1),(True,'death_pixel'))

    def test_crop_rejects_outside(self):
        with self.assertRaises(ValueError):
            crop(self.blank,(250,0,20,20))

    def test_focus_loss_releases_keys(self):
        e = self.env
        e._closed = False
        e._api = Mock()
        e._api.GetAsyncKeyState.return_value = 0
        e.window_title = 'geometry dash'
        e.release_keys = Mock()
        from unittest.mock import patch
        with patch('gd_live_env.foreground_title',return_value='Other app'):
            with self.assertRaises(RuntimeError):
                e._guard()
        e.release_keys.assert_called_once()

    def test_short_attempt_and_flicker(self):
        e = self.env
        e._is_dead(self.blank,0)
        self.assertFalse(e._is_dead(self.visible,0.02)[0])
        e._is_dead(self.blank,0.1)
        e._is_dead(self.blank,0.16)
        self.assertEqual(e._is_dead(self.visible,0.18),(True,'auto_retry'))

    def test_stall_truncates_then_waits_for_real_restart(self):
        e = self.env
        e._needs_reset = e._paused = e._held = False
        e._guard = Mock()
        e.release_keys = Mock()
        e.action_space = spaces.Discrete(2)
        e._sample = Mock(return_value=self.blank)
        e._previous = e._observation(self.blank)
        e._last_motion = 0
        e.motion_threshold = 0.5
        e.stall_seconds = 8
        e.period = 0.03
        e._episode_steps = 0
        e.max_episode_steps = 100
        e._restart_obs = None
        with patch('gd_live_env.time.perf_counter',side_effect=[10,10.04]):
            _,reward,terminated,truncated,info = e.step(0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertLess(reward,0)
        self.assertEqual(info['reset_reason'],'stall')
        e.reset_timeout = 0.001
        e.on_wait = Mock()
        e._sample = Mock(side_effect=[self.blank,self.blank,self.visible])
        with patch('gd_live_env.time.perf_counter',side_effect=[20,21,21,22,22,23]):
            obs,_ = e.reset()
        self.assertEqual(e.on_wait.call_count,3)
        self.assertEqual(obs.shape,(84,84,1))
        self.assertFalse(e._needs_reset)

    def test_ten_thousand_steps_with_deaths_and_stalls(self):
        e = self.env
        e._needs_reset = True
        e._paused = e._held = False
        e._guard = Mock()
        e.release_keys = Mock()
        e.action_space = spaces.Discrete(2)
        e._sample = Mock(return_value=self.visible)
        e._attempt_visible = Mock(return_value=True)
        observation = e._observation(self.blank)
        e._observation = Mock(return_value=observation)
        e._restart_obs = None
        e.motion_threshold = 0.5
        e.stall_seconds = 0.1
        e.reset_timeout = 1
        e.max_episode_steps = 100
        e.period = 0.01
        e.on_wait = None
        clock = [0.0]
        def tick():
            clock[0] += 0.02
            return clock[0]
        deaths = stalls = 0
        with patch('gd_live_env.time.perf_counter',side_effect=tick):
            e.reset()
            for step in range(10000):
                e._is_dead = Mock(return_value=(step % 7 == 0,'auto_retry' if step % 7 == 0 else ''))
                _,reward,terminated,truncated,_ = e.step(0)
                self.assertTrue(np.isfinite(reward))
                deaths += int(terminated)
                stalls += int(truncated)
                if terminated or truncated:
                    e.reset()
        self.assertGreater(deaths,1000)
        self.assertGreater(stalls,1000)

    def test_save_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            path = Path(tmp)/'agent.zip'
            path.write_bytes(b'previous checkpoint')
            model = Mock()
            def fail(target):
                Path(target).write_bytes(b'partial')
                raise OSError('disk error')
            model.save.side_effect = fail
            with self.assertRaises(OSError):
                atomic_save(model,path)
            self.assertEqual(path.read_bytes(),b'previous checkpoint')
            self.assertFalse(path.with_name('agent.tmp.zip').exists())


class FakeGame(gym.Env):
    observation_space = spaces.Box(0,255,(84,84,1),np.uint8)
    action_space = spaces.Discrete(2)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.np_random.integers(0,256,(84,84,1),dtype=np.uint8),{}
    def step(self,action):
        self.steps += 1
        return self.np_random.integers(0,256,(84,84,1),dtype=np.uint8),0.1,self.steps >= 8,False,{}


class PPOSmokeTest(unittest.TestCase):
    def test_cuda_update_save_load(self):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack,VecTransposeImage
        from stable_baselines3.common.monitor import Monitor
        torch.set_num_threads(1)
        self.assertTrue(torch.cuda.is_available(),'CUDA unavailable')
        device = 'cuda'
        vec = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: Monitor(FakeGame())]),4,channels_order='last'))
        try:
            model = PPO('CnnPolicy',vec,n_steps=64,batch_size=32,n_epochs=1,device=device,policy_kwargs=policy_settings())
            before = next(model.policy.parameters()).detach().clone()
            model.learn(64)
            self.assertFalse(torch.equal(before,next(model.policy.parameters())))
            with tempfile.TemporaryDirectory(dir='.') as tmp:
                path = Path(tmp)/'model.zip'
                atomic_save(model,path)
                loaded = PPO.load(path,env=vec,device=device)
                obs = vec.reset()
                self.assertTrue(np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0]))
                self.assertFalse(path.with_name('model.tmp.zip').exists())
                loaded.learn(64, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps,128)
        finally:
            vec.close()


class TrainingExitTests(unittest.TestCase):
    def test_focus_and_manual_stop_save_before_close(self):
        from types import SimpleNamespace
        from gd_live_env import GameFocusLost
        from train_live import train
        from stable_baselines3 import PPO
        for exception in (GameFocusLost('test focus'),KeyboardInterrupt('test F9')):
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory(dir='.') as tmp:
                path = Path(tmp)/'final.zip'
                class StoppingGame(FakeGame):
                    _paused = False
                    total = 0
                    def step(self,action):
                        self.total += 1
                        if self.total == 12:
                            raise exception
                        return super().step(action)
                    def pause(self,check_stop=True):
                        self._paused = True
                    def resume(self):
                        self._paused = False
                    def release_keys(self):
                        pass
                    def close(inner):
                        self.assertTrue(path.exists(),'Must save before closing')
                raw = StoppingGame()
                args = SimpleNamespace(device='cuda',log_dir=tmp,model_path=str(path),resume=None,seed=42,n_steps=64,total_timesteps=64,checkpoint_steps=4,checkpoint_episodes=25,checkpoint_seconds=120,keep_checkpoints=2)
                with patch('train_live.make_env',return_value=raw),patch('train_live.countdown'):
                    train(args)
                loaded = PPO.load(path,device='cuda')
                self.assertEqual(loaded.num_timesteps,11)
                self.assertEqual(len(list(Path(tmp).glob('*/checkpoint_*.zip'))),2)


if __name__ == '__main__':
    unittest.main(verbosity=2)



