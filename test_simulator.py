import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from gd_env import GeometryDashEnv


class SimulatorTests(unittest.TestCase):
    def test_gym_contract(self):
        check_env(GeometryDashEnv(), skip_render_check=True)

    def test_seed_and_map(self):
        a, b = GeometryDashEnv(), GeometryDashEnv()
        a.reset(seed=42)
        b.reset(seed=42)
        np.testing.assert_array_equal(a.obstacles, b.obstacles)
        previous = a.obstacles.copy()
        a.reset()
        self.assertFalse(np.array_equal(previous, a.obstacles))

    def test_jump_and_ground(self):
        env = GeometryDashEnv(obstacles=[])
        env.reset()
        env.step(1)
        self.assertGreater(env.y, 0)
        self.assertFalse(env.grounded)
        for _ in range(30):
            env.step(0)
        self.assertEqual(env.y, 0)
        self.assertTrue(env.grounded)

    def test_spike_death_and_block_side(self):
        for kind in (0,1):
            env = GeometryDashEnv(obstacles=[[3,1,1,kind]])
            env.reset()
            for _ in range(30):
                _, reward, done, truncated, info = env.step(0)
                if done:
                    break
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertLess(reward, 0)
            self.assertEqual(info['death_reason'], 'spike' if kind == 0 else 'block')
            with self.assertRaises(RuntimeError):
                env.step(0)

    def test_block_landing_and_walk_off(self):
        env = GeometryDashEnv(obstacles=[[3,2,1,1]])
        env.reset()
        env.x, env.y, env.vy, env.grounded = 3.1,1.05,-2,False
        _,_,done,_,_ = env.step(0)
        self.assertFalse(done)
        self.assertEqual(env.y,1)
        self.assertTrue(env.grounded)
        for _ in range(20):
            env.step(0)
        self.assertEqual(env.y,0)

    def test_triangle_not_bounding_box(self):
        env = GeometryDashEnv(obstacles=[[3,2,1,0]])
        env.reset()
        # Cube just entering low left flank, above triangle but below its apex.
        env.x,env.y,env.vy,env.grounded = 2.01,0.9,0,False
        self.assertFalse(env.step(0)[2])

    def test_completion_and_timeout(self):
        env = GeometryDashEnv(level_length=20, obstacles=[])
        env.reset()
        for _ in range(110):
            obs,reward,done,truncated,info = env.step(0)
            self.assertTrue(env.observation_space.contains(obs))
            if done:
                break
        self.assertTrue(info['is_success'])
        self.assertGreater(reward,1)
        self.assertFalse(truncated)
        env = GeometryDashEnv(max_episode_steps=1)
        env.reset()
        self.assertEqual(env.step(0)[2:4],(False,True))

    def test_ten_thousand_steps(self):
        env = GeometryDashEnv()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        deaths = 0
        for _ in range(10000):
            obs,reward,terminated,truncated,_ = env.step(int(rng.integers(2)))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            if terminated or truncated:
                deaths += 1
                env.reset()
        self.assertGreater(deaths,10)

    def test_subprocesses_and_monitor(self):
        from functools import partial
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        vec = VecMonitor(SubprocVecEnv([partial(GeometryDashEnv,max_episode_steps=2) for _ in range(4)],start_method='spawn'))
        try:
            vec.seed(123)
            self.assertEqual(vec.reset().shape,(4,20))
            vec.step(np.zeros(4,dtype=int))
            obs, rewards, dones, infos = vec.step(np.zeros(4,dtype=int))
            self.assertTrue(dones.all())
            self.assertTrue(all('episode' in info and 'terminal_observation' in info for info in infos))
            self.assertEqual(obs.shape,(4,20))
        finally:
            vec.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)

