import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv_v3(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv_v3, self).__init__()
        self.n_leaders = 4 #2
        self.mask = np.ones((self.n_agents,), dtype = int)
        self.mask[0:self.n_leaders] = 0
        # self.xor_mask = self.mask ^ 1
        self.quiver = None
        self.vx = 4
        self.timestep = 800
        self.c = 50

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv_v3, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,), dtype = int)
        self.mask[0:self.n_leaders] = 0
        self.v_hist = np.zeros((800,2))
        self.v_hist[0,:] = [self.vx, 0]
        t = np.arange(1,800) * self.dt
        x = np.zeros((2,self.timestep - 1))
        x[0, :] = self.vx * np.ones(self.timestep - 1)
        x[1, :] = np.square(self.vx)/50 * t
        self.v_hist[1:800, :] = x.T
        self.Rx_final = sum(self.v_hist[:,0]) * self.dt
        self.Ry_final = sum(self.v_hist[:,1]) * self.dt
        print('Rx: {}'.format(self.Rx_final))
        print('Ry: {}'.format(self.Ry_final))

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
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
        t = self.n_timesteps * self.dt
        self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
        self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
        # x, y velocity
        self.x[0:self.n_leaders, 2] = self.vx
        self.x[0:self.n_leaders, 3] = (self.vx^2)/50 * t

        if self.n_timesteps > self.timestep - 1:
            self.done = True
            # x in nest?
            cond1 = self.x[:,0] >= self.Rx_final - self.nest_R
            cond2 = self.x[:,0] <= self.Rx_final + self.nest_R
            cond3 = self.x[:,1] >= self.Ry_final
            self.x_in_nest = cond1 & cond2 & cond3
            self.S_in_nest += sum(self.x_in_nest)
            print('n_agents in nest: ',sum(self.x_in_nest))   

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), self.done, {}

    def reset(self):
        super(FlockingLeaderEnv_v3, self).reset()        
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.vx, 0]]
                                                                                                                                                                          
        return (self.state_values, self.state_network)

    # def render(self, index, n_test_episodes):
    def render(self, mode='human'):
        super(FlockingLeaderEnv_v3, self).render(mode)#index, n_test_episodes)

        # self.ax.plot([self.goal_x,self.goal_x],[-self.nest_R,self.nest_R])
        self.ax.plot([self.Rx_final - self.nest_R, self.Rx_final + self.nest_R],[self.Ry_final,self.Ry_final])

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