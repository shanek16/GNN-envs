import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv_v2(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv_v2, self).__init__()
        self.n_leaders = 4 #2
        self.mask = np.ones((self.n_agents,), dtype = int)
        self.mask[0:self.n_leaders] = 0
        # self.xor_mask = self.mask ^ 1
        self.quiver = None
        # self.half_leaders = int(self.n_leaders / 2.0)

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv_v2, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,), dtype = int)
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u
        self.n_timesteps += 1

        # uninformed bees
        # x, y position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5 * self.mask
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5 * self.mask
        # x, y velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt * self.mask
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt * self.mask

        # leader bees
        # x, y position
        t = np.pi/4 * self.n_timesteps * self.dt
        self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leader, 2] * self.dt
        self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leader, 3] * self.dt
        # x, y velocity
        self.x[0:self.n_leader, 2] = -self.v_max * np.sin(t - np.pi/2)
        self.x[0:self.n_leader, 3] = self.v_max * np.cos(t - np.pi/2)
        # self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5 * self.xor_mask
        # self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5 * self.xor_mask
        # # x, y velocity
        # self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt * self.xor_mask
        # self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt * self.xor_mask

        if self.n_timesteps > 199:
            self.done = True
            # x in nest?
            cond1 = self.x[:,0] >= self.v_max - self.nest_R
            cond2 = self.x[:,0] <= self.v_max + self.nest_R
            cond3 = self.x[:,1] >= self.v_max
            self.x_in_nest = cond1 & cond2 & cond3
            self.S_in_nest += sum(self.x_in_nest)
            print('n_agents in nest: ',sum(self.x_in_nest))   

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), self.done, {}

    def reset(self):
        super(FlockingLeaderEnv_v2, self).reset()
        # self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
        #                                                                                  high=self.v_max, size=(1, 1))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
                                                                                                                                                                          
        return (self.state_values, self.state_network)

    def render(self, mode='human'):
        super(FlockingLeaderEnv_v2, self).render(mode)
        # self.ax.plot([self.goal_x,self.goal_x],[-self.nest_R,self.nest_R])
        self.ax.plot([self.v_max - self.nest_R, self.v_max + self.nest_R],[self.v_max,self.v_max])

        X = self.x[0:self.n_leaders, 0]
        Y = self.x[0:self.n_leaders, 1]
        U = self.x[0:self.n_leaders, 2]
        V = self.x[0:self.n_leaders, 3]

        if self.quiver == None:
            self.quiver = self.ax.quiver(X, Y, U, V, color='r')
        else:
            self.quiver.set_offsets(self.x[0:self.n_leaders, 0:2])
            self.quiver.set_UVC(U, V)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        # print('\nself.average_timesteps: ',self.S_timesteps/self.n_test_episodes)
        print('average_# of_agents in nest: ',self.S_in_nest/self.n_test_episodes)
        pass