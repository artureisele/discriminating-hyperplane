"""
Gymnasium version of cart-pole environment of this repo
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class CartPole(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End
    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500

    ## Arguments

    Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    from gymnasium.envs.registration import register
    register(
        id="customEnvs/CartPoleEnv-v0",
        entry_point="envs/cartpole_pret_gymnasium:CartPoleEnv",
    )
    >>> import gymnasium as gym
    >>> env = gym.make("customEnvs/CartPoleEnv-v0", render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self, render_mode: Optional[str] = None, focus:float=0.0, rewardtype:int = 0, debug_hyperplanes_render:bool = False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # try to stay ariound position focus
        self.focus = focus

        # 0 = Reward for learning the barriers
        # 1 = Reward for coming close to focus point
        self.rewardtype= rewardtype

        # Angle at which to fail the episode
        self.theta_threshold_radians = 24 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.debug_hyperplanes_render = debug_hyperplanes_render
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high)

        #a should be between -1 and 1 -> but actually vector with norm 1
        #b should be between -X and X where X is the maximum action magnitude of the real actions
        action_low = np.array([-1])
        action_high = np.array([1])
        #a,b of a hyperplane
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(1,)
        )

        self.render_mode = render_mode

        self.screen_width = 700
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def set_values_for_debug_render(self, next_real_action,next_threshold,to_right_is_dangerous, next_desired_action):
        self.next_real_action = next_real_action
        self.next_threshold = next_threshold
        self.to_right_is_dangerous = to_right_is_dangerous
        self.next_desired_action = next_desired_action

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        bonus = 1 if -1<action and action<1 else 0
        action = np.clip(action, [-1], [1])
        action = action.astype(np.float32)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            if self.rewardtype == 0:
                reward = 1.0 + bonus
            elif self.rewardtype == 1:
                penalty = (math.sqrt((x-self.focus)**2)/(2*self.x_threshold))
                assert penalty<=1 and penalty>=0
                reward = 1 - penalty
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = -100
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward=-1
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {"bonus":bonus}
    

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        For options, tuple bounds or (4)np array state are avaible
        """
        super().reset(seed=seed)
        if options is None:
            options = {}

        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = options.get("bounds",(-0.05,0.05))
        self.state = options.get("state", self.np_random.uniform(low=low, high=high, size=(4,)))

        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}
        
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2 +1.2
        scale = self.screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0
        if self.state is None:
            return None
        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        gfxdraw.vline(self.surf,int(self.x_threshold * scale + self.screen_width / 2.0), 0, self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(-self.x_threshold * scale + self.screen_width / 2.0),0,self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(self.focus * scale + self.screen_width / 2.0),0,self.screen_height, (0, 255, 255))

        if self.debug_hyperplanes_render:
            next_desired_action = self.next_desired_action
            next_real_action = self.next_real_action
            next_threshold = self.next_threshold
            percentage_of_left_line = (next_threshold+1)/2.0
            percentage_of_real_action = (next_real_action+1)/2.0
            percentage_of_desired_action = (next_desired_action+1)/2.0
            to_right_is_dangerous = self.to_right_is_dangerous
            left_color = (255,0,0) if not to_right_is_dangerous else (0,255,0)
            right_color = (255,0,0) if to_right_is_dangerous else (0,255,0)
            gfxdraw.hline(self.surf, int(self.screen_width / 4.0), int(self.screen_width / 4.0) + int ( (self.screen_width / 4.0 * 2.0)*percentage_of_left_line), 50, left_color)
            gfxdraw.hline(self.surf, int(self.screen_width / 4.0) + int ( (self.screen_width / 4.0 * 2.0)*percentage_of_left_line), int(self.screen_width / 4.0*3.0) , 50, right_color)
            gfxdraw.vline(self.surf,int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_real_action) ,25,75,(0,0,0))
            gfxdraw.vline(self.surf,int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_desired_action) ,35,65,(0,0,255))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False