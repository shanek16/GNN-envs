import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv_v4(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv_v4, self).__init__()
        self.n_leaders = 4 #2
        self.mask = np.ones((self.n_agents,), dtype = int)
        self.mask[0:self.n_leaders] = 0
        # self.xor_mask = self.mask ^ 1
        self.quiver = None

    def params_from_cfg(self, args, m):
        super(FlockingLeaderEnv_v4, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,), dtype = int)
        self.mask[0:self.n_leaders] = 0
        
        self.m = int(m)
        print('m = ',self.m)
        self.cx = np.sqrt(self.v_max**2/(self.m**2 + 1))
        self.v_hist = np.zeros((210,2))
        self.v_hist[:100,:] = [self.v_max, 0]
        x = np.ones((2,110))*[[self.cx], [self.m * self.cx]]
        self.v_hist[100:210, :] = x.T
        self.Rx_final = sum(self.v_hist[:,0]) * self.dt
        self.Ry_final = sum(self.v_hist[:,1]) * self.dt
        print('Rx: {}'.format(self.Rx_final))
        print('Ry: {}'.format(self.Ry_final))

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u
        self.n_timesteps += 1

        # uninformed bees
        # x, y position
        self.x[self.n_leaders:, 0] = self.x[self.n_leaders:, 0] + self.x[self.n_leaders:, 2] * self.dt + self.u[self.n_leaders:, 0] * self.dt * self.dt * 0.5
        self.x[self.n_leaders:, 1] = self.x[self.n_leaders:, 1] + self.x[self.n_leaders:, 3] * self.dt + self.u[self.n_leaders:, 1] * self.dt * self.dt * 0.5
        # x, y velocity
        self.x[self.n_leaders:, 2] = self.x[self.n_leaders:, 2] + self.u[self.n_leaders:, 0] * self.dt
        self.x[self.n_leaders:, 3] = self.x[self.n_leaders:, 3] + self.u[self.n_leaders:, 1] * self.dt

        # leader bees
        # x, y position
        self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
        self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
        if self.n_timesteps < 100:
            pass
        elif self.n_timesteps <= 209:
            # x, y velocity
            self.x[0:self.n_leaders, 2] = self.cx
            self.x[0:self.n_leaders, 3] = self.m * self.cx

        else:#if self.n_timesteps > 209:
            self.done = True
            # x in nest?
            self.x_in_nest = np.square(self.x[:,0]-self.Rx_final)+np.square(self.x[:,1]-self.Ry_final) <= np.square(self.nest_R)
            self.S_in_nest += sum(self.x_in_nest)
            # print('n_agents in nest: ',sum(self.x_in_nest))

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), self.done, {}

    def reset(self):
        super(FlockingLeaderEnv_v4, self).reset()
        # self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
        #                                                                                  high=self.v_max, size=(1, 1))
        
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
                                                                                                                                                                          
        return (self.state_values, self.state_network)

    # def render(self, index, n_test_episodes):
    def render(self, mode='human'):
        super(FlockingLeaderEnv_v4, self).render(mode)#index, n_test_episodes)

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