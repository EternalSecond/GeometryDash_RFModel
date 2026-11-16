"""Single real Windows Geometry Dash environment. F9 stops input."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def windows_api():
    if sys.platform != 'win32':
        raise RuntimeError('Real game input requires Windows.')
    from ctypes import wintypes
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetForegroundWindow.restype = wintypes.HWND
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.GetAsyncKeyState.argtypes = [ctypes.c_int]
    api.GetAsyncKeyState.restype = ctypes.c_short
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    try:
        api.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        api.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        api.SetProcessDPIAware()
    return api


def foreground_title(api):
    title = ctypes.create_unicode_buffer(512)
    api.GetWindowTextW(api.GetForegroundWindow(), title, len(title))
    return title.value


def client_region(api):
    from ctypes import wintypes
    rect, point = wintypes.RECT(), wintypes.POINT()
    hwnd = api.GetForegroundWindow()
    if not api.GetClientRect(hwnd, ctypes.byref(rect)) or not api.ClientToScreen(hwnd, ctypes.byref(point)):
        raise OSError('Cannot read game window coordinates.')
    return (point.x, point.y, rect.right, rect.bottom)


def crop(frame, roi):
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('ROI is outside the captured frame.')
    return frame[y:y+h, x:x+w]


class GameFocusLost(RuntimeError):
    """Stop input and let the trainer persist its state."""


class GeometryDashEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, region, attempt_roi, template_path, control_hz=30,
                 capture_hz=60, match_threshold=0.85, motion_threshold=0.5,
                 stall_seconds=8, reset_timeout=30, death_pixel=None,
                 pixel_tolerance=12, window_title='Geometry Dash',
                 attempt_absence_seconds=0.05, max_episode_steps=18000):
        super().__init__()
        if not 0 < control_hz <= capture_hz or not 0 < match_threshold <= 1:
            raise ValueError('Require 0 < control_hz <= capture_hz and 0 < threshold <= 1.')
        if motion_threshold < 0 or min(stall_seconds, reset_timeout) <= 0:
            raise ValueError('Invalid motion threshold or timeout.')
        left, top, width, height = region
        if min(width, height) <= 0 or not window_title.strip():
            raise ValueError('Invalid region or window title.')
        crop(np.empty((height, width), np.uint8), attempt_roi)
        if death_pixel is not None:
            x, y, r, g, b = death_pixel
            if not (0 <= x < width and 0 <= y < height and all(0 <= c <= 255 for c in (r,g,b))):
                raise ValueError('Invalid death pixel.')
        if not 0 <= pixel_tolerance <= 255:
            raise ValueError('Invalid pixel tolerance.')
        template = cv2.imread(str(Path(template_path)), cv2.IMREAD_GRAYSCALE)
        if template is None or template.shape != (attempt_roi[3], attempt_roi[2]):
            raise ValueError('Calibrate first: template PNG must match attempt ROI dimensions.')
        self.template = cv2.Canny(template, 80, 160)
        if np.count_nonzero(self.template) < 20:
            raise ValueError('Template is empty; select the contrasting ATTEMPT word.')
        self.region = dict(left=left, top=top, width=width, height=height)
        self.roi, self.death_pixel = attempt_roi, death_pixel
        self.pixel_tolerance, self.window_title = pixel_tolerance, window_title.lower()
        self.threshold, self.motion_threshold = match_threshold, motion_threshold
        self.period, self.capture_period = 1/control_hz, 1/capture_hz
        self.stall_seconds, self.reset_timeout = stall_seconds, reset_timeout
        if attempt_absence_seconds <= 0 or max_episode_steps < 1:
            raise ValueError('Detector debounce and episode limit must be positive.')
        self.attempt_absence_seconds = attempt_absence_seconds
        self.max_episode_steps = max_episode_steps
        self.on_wait = None
        self._episode_steps = 0
        self._pixel_ready = False
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 255, (84, 84, 1), np.uint8)
        self._api = windows_api()
        import mss
        import pydirectinput
        self._keys = pydirectinput
        self._keys.PAUSE = 0
        self._sct = mss.mss()
        self._held = self._paused = self._closed = False
        self._needs_reset = True
        self._restart_obs = self._previous = self._absent_since = None
        self._armed = False
        self._pixel_hits = 0
        self._last_motion = self._next_capture = time.perf_counter()
        self.last_score = 0.0

    def _guard(self, check_stop=True):
        if self._closed:
            raise RuntimeError('Environment is closed.')
        if check_stop and self._api.GetAsyncKeyState(0x78) & 0x8000:
            self.release_keys()
            raise KeyboardInterrupt('F9')
        if self.window_title not in foreground_title(self._api).lower():
            self.release_keys()
            raise GameFocusLost('Game lost focus; saving and stopping.')

    def release_keys(self):
        if self._held:
            old = self._keys.FAILSAFE
            try:
                self._keys.FAILSAFE = False
                self._keys.keyUp('space')
                self._held = False
            finally:
                self._keys.FAILSAFE = old

    def _grab(self):
        self._guard()
        return np.asarray(self._sct.grab(self.region))

    def _sample(self):
        time.sleep(max(0, self._next_capture-time.perf_counter()))
        frame = self._grab()
        self._next_capture = time.perf_counter()+self.capture_period
        return frame

    @staticmethod
    def _observation(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)[...,None]

    def _attempt_visible(self, frame):
        x,y,w,h = self.roi
        search = frame[max(0,y-8):min(frame.shape[0],y+h+8), max(0,x-32):min(frame.shape[1],x+w+32)]
        edges = cv2.Canny(cv2.cvtColor(search, cv2.COLOR_BGRA2GRAY),80,160)
        self.last_score = float(cv2.matchTemplate(edges,self.template,cv2.TM_CCOEFF_NORMED).max())
        return self.last_score >= self.threshold

    def _is_dead(self, frame, now):
        visible = self._attempt_visible(frame)
        if visible:
            self._absent_since = None
        else:
            if self._absent_since is None:
                self._absent_since = now
            if now-self._absent_since >= self.attempt_absence_seconds:
                self._armed = True
        if self._armed and visible:
            return True, 'auto_retry'
        if self.death_pixel is not None:
            x,y,r,g,b = self.death_pixel
            hit = np.max(np.abs(frame[y,x,:3].astype(np.int16)-np.array([b,g,r],np.int16))) <= self.pixel_tolerance
            if not hit:
                self._pixel_ready = True
            self._pixel_hits = self._pixel_hits+1 if hit else 0
            if self._pixel_ready and self._pixel_hits >= 2:
                return True, 'death_pixel'
        return False, ''

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._paused or not self._needs_reset:
            raise RuntimeError('Reset requires initial state or death and an unpaused game.')
        self.release_keys()
        obs, self._restart_obs = self._restart_obs, None
        if obs is None:
            next_notice = time.perf_counter()+self.reset_timeout
            # No fabricated reset observation: wait for an actual attempt.
            # F9/focus checks remain active, and the trainer can save while waiting.
            while obs is None:
                frame = self._sample()
                if self.on_wait is not None:
                    self.on_wait()
                if self._attempt_visible(frame):
                    obs = self._observation(frame)
                elif time.perf_counter() >= next_notice:
                    print('Waiting for ATTEMPT: check level, Auto-Retry and calibration. F9 stops.', flush=True)
                    next_notice = time.perf_counter()+self.reset_timeout
        self._armed = self._needs_reset = False
        self._absent_since, self._pixel_hits = None, 0
        self._pixel_ready = False
        self._episode_steps = 0
        self._previous = obs.copy()
        self._last_motion = time.perf_counter()
        self._next_capture = self._last_motion+self.capture_period
        return obs, {'attempt_score':self.last_score}

    def step(self, action):
        if self._needs_reset or self._paused:
            raise RuntimeError('Reset/resume before step.')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action!r}')
        try:
            self._guard()
            started = time.perf_counter()
            if int(action) == 1 and not self._held:
                self._held = True
                self._keys.keyDown('space')
            elif int(action) == 0:
                self.release_keys()
            while True:
                frame = self._sample()
                now = time.perf_counter()
                dead, reason = self._is_dead(frame, now)
                if dead or now-started >= self.period:
                    break
            obs = self._observation(frame)
            motion = float(cv2.absdiff(obs[16:76],self._previous[16:76]).mean())
            self._previous = obs.copy()
            if motion >= self.motion_threshold:
                self._last_motion = now
            self._episode_steps += 1
            info = dict(motion=motion, attempt_score=self.last_score, death_reason=reason, step_seconds=now-started)
            if dead:
                self.release_keys()
                self._needs_reset = True
                if reason == 'auto_retry':
                    self._restart_obs = obs.copy()
                return obs, -1.0, True, False, info
            if now-self._last_motion > self.stall_seconds:
                self.release_keys()
                self._needs_reset = True
                info['reset_reason'] = 'stall'
                return obs, -0.1, False, True, info
            if self._episode_steps >= self.max_episode_steps:
                self.release_keys()
                self._needs_reset = True
                self._restart_obs = obs.copy()  # Logical time limit; game continues.
                info['reset_reason'] = 'time_limit'
                return obs, 0.0, False, True, info
            # Bounded time-scaled reward; a frozen screen earns no survival bonus.
            reward = min(now-started, 2*self.period) * (0.1 if motion >= self.motion_threshold else -0.02)
            return obs, reward, False, False, info
        except BaseException:
            self.release_keys()
            raise

    def pause(self, check_stop=True):
        if not self._paused:
            self._guard(check_stop)
            self.release_keys()
            self._keys.press('esc')
            self._paused = True
            self._pause_started = time.perf_counter()
            time.sleep(0.15)

    def resume(self):
        if self._paused:
            self._guard()
            self._keys.press('esc')
            self._paused = False
            now = time.perf_counter()
            paused_seconds = now-self._pause_started
            self._last_motion += paused_seconds
            if self._absent_since is not None:
                self._absent_since += paused_seconds
            self._next_capture = now+self.capture_period

    def close(self):
        if not self._closed:
            try:
                self.release_keys()
            finally:
                self._sct.close()
                self._closed = True
