import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv2(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv2, self).__init__()
        self.n_leaders = 4 #2
        self.leader_mode = 1 #'streaker'
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None
        self.half_leaders = int(self.n_leaders / 2.0)
        self.v_mean = 6.7 #mean velocity of uninformed bees from paper

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv2, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        leader_dict = {1: "streaker", 0: "passive_leader"}
        assert u.shape == (self.n_agents, self.nu)
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u

        #uninformed bees
        # x, y position
        self.x[self.n_leaders:, 0] = self.x[self.n_leaders:, 0] + self.x[self.n_leaders:, 2] * self.dt + self.u[self.n_leaders:, 0] * self.dt * self.dt * 0.5
        self.x[self.n_leaders:, 1] = self.x[self.n_leaders:, 1] + self.x[self.n_leaders:, 3] * self.dt + self.u[self.n_leaders:, 1] * self.dt * self.dt * 0.5
        # x, y velocity
        self.x[self.n_leaders:, 2] = self.x[self.n_leaders:, 2] + self.u[self.n_leaders:, 0] * self.dt
        self.x[self.n_leaders:, 3] = self.x[self.n_leaders:, 3] + self.u[self.n_leaders:, 1] * self.dt

        #informed bees
        #sol1) 
        #if max(leader_position) > max(x(position)): mode=passive and leader_velocity=-v_mean
        if max(self.x[0:self.n_leaders,0]) > max(self.x[self.n_leaders:,0]):
            self.leader_mode = 0 #'passive_leader'
            self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
            self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
            self.x[0:self.n_leaders,2] = -self.v_mean #x vel=-v_mean
            self.x[0:self.n_leaders,3] = 0 #y vel=0 ->>>>>>>>>>>>>>>>>>>idle?

        #else if  passive mode   and min(leader_position) < min(x(position)): mode=active and leader_vel=max_v
        elif self.leader_mode==0 and min(self.x[0:self.n_leaders,0]) < min(self.x[self.n_leaders:,0]):
            self.leader_mode = 1 #active leader
            self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
            self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
            self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
        else:
            # x, y position
            self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
            self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
        # print('leader_mode: ',leader_dict[self.leader_mode])
        
        #sol2) if leader> front 10% of flock, x(velocity)==0

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingLeaderEnv2, self).reset()
        self.leader_mode = 1 #active leader
        # self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
        #                                                                                  high=self.v_max, size=(1, 1))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
                                                                                                                                                                          
        return (self.state_values, self.state_network)

    def render(self, mode='human'):
        super(FlockingLeaderEnv2, self).render(mode)

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
