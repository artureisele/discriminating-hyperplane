import gym
import numpy as np
import math
from gym import spaces, logger
from gym.utils import seeding
from sklearn import svm
from sklearn.svm import LinearSVC
import time
import copy

class Cartpole(gym.Env):
    def __init__(self, inv_set, disc_steps, focus=0.0):
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

        # Angle at which to fail the episode
        self.theta_threshold_radians = 24 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.inv_set = inv_set
        self.disc_steps = disc_steps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        bonus = int(action >= -1 and action <= 1)
        action = np.clip(action, -1, 1)
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians\
            or theta > self.theta_threshold_radians
        #done = False

        if not done:
            if x > -self.x_threshold \
            and x < self.x_threshold \
            and theta > -self.theta_threshold_radians \
            and theta < self.theta_threshold_radians:
                reward = 1 + bonus
            else:
                reward = -1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -100
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {'cost' : self.cost()}
    
    def cost(self):
        x, x_dot, theta, theta_dot = self.state
        return int(x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians)
        # if action < -1 or action > 1:
        #     return 1

    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
        
    def run_system(self, x_0, num_steps, controller):
        x_hist = []
        x_t = self.reset(x_0)
        term = False
        step = 0
        while not (term) and step <= num_steps:
            u = controller.forward(x_t, step)
            if u >= 1:
                u = [1]
            elif u <= -1:
                u = [-1]
            u = np.float32(u)
            x_t, rew, term, info = self.step(u)
            x_hist.append(x_t)
            step += 1
        return np.array(x_hist)
    
    def next_states(self, states, inputs):
        # loop, vectorized gym, remake source code to be vectorized???
        num_states = states.shape[0]
        num_inputs = inputs.shape[1]
        states = np.repeat(states, num_inputs, axis=0).reshape(num_states, num_inputs, 4)
        for step in range(self.disc_steps):
            next_states = np.zeros((num_states, num_inputs, 4))
            for i in range(num_states):
                for j in range(num_inputs):
                    state = states[i, j]
                    state = self.reset(state)
                    input = np.float32(inputs[i, j])
                    next_state, _, _, _ = self.step(input)
                    next_states[i, j] = next_state
            states = next_states
        return next_states

    def sample_states(self, num_states):
        pos = np.random.uniform(-4.8, 4.8, num_states)
        vel = np.random.uniform(-5, 5, num_states)
        theta = np.random.uniform(-.418, .418, num_states)
        theta_dot = np.random.uniform(-1, 1, num_states)
        states = np.column_stack((pos, vel, theta, theta_dot))
        states = states[self.inv_set.forward(states) > 0]
        return states

    def sample_inputs(self, states, num_inputs):
        # sample ints or cont values????
        num_states = states.shape[0]
        inputs = np.random.uniform(-1, 1, num_states*num_inputs).reshape((num_states, num_inputs, 1))
        return inputs
    
    def label_inputs(self, states, inputs):
        num_states = states.shape[0]
        num_inputs = inputs.shape[1]
        env_copy = copy.deepcopy(self)
        next_states = env_copy.next_states(states, inputs)
        labels = np.sign(self.inv_set.forward(next_states.reshape(-1, 4))).reshape(num_states, num_inputs)
        return labels
    
    def label_state(self, inputs, input_labels):
        if np.all(input_labels == 1.0):
            a = np.array([1])
            b = np.array([-1])
        elif np.all(input_labels == -1.0):
            a = np.array([1])
            b = np.array([1])
        else: 
            lsvc = LinearSVC(verbose=0, class_weight={-1:2, 1:1})
            lsvc.fit(inputs.reshape(-1, 1), np.ravel(input_labels.reshape(-1, 1)))
            a=lsvc.coef_[0]
            b=-lsvc.intercept_
        return a, b
    def sample_data_real_trajectory(self, num_steps):
        x_hist = []
        a_hist = []
        x_t = self.reset()
        x_hist.append(x_t)
        term = False
        step = 0
        while not (term) and step < num_steps:
            a = self.action_space.sample()
            a = np.float32(a)
            a_hist.append(a)
            x_t, rew, term, info = self.step(a)
            x_hist.append(x_t)
            step += 1
        return np.array(x_hist), np.array(a_hist), term
    def sample_data_real_trajectory2(self, num_steps, num_actions):
        x_hist = []
        a_hist = []
        done_hist = []
        pos = np.random.uniform(-3, 3, 1)
        vel = np.random.uniform(-5, 5, 1)
        theta = np.random.uniform(-.418, .418, 1)
        theta_dot = np.random.uniform(-1, 1, 1)
        states = (pos, vel, theta, theta_dot)
        inital_state = self.reset(states)
        x_hist.append(inital_state)
        for j in range(num_actions):
            self.reset(inital_state)
            term = False
            step = 0
            a = self.action_space.sample()
            a = np.float32(a)
            a_hist.append(a)
            while step < num_steps:
                x_t, rew, done, info = self.step(a)
                if done:
                    term = done[0]
                step += 1
            done_hist.append(term)
        return np.array(x_hist), np.array(a_hist), np.array(done_hist)
    def sample_data(self, num_states, num_inputs):
        states = self.sample_states(num_states)
        while len(states) <= num_states:
            states = np.vstack((states, self.sample_states(num_states)))
        inputs = self.sample_inputs(states, num_inputs)
        labels = self.label_inputs(states, inputs)
        return states[:num_states], inputs[:num_states], labels[:num_states]
    
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()