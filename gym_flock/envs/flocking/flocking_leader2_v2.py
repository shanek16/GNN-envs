import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv2_v2(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv2_v2, self).__init__()
        self.n_leaders = 4 #2
        self.leader_mode = 1 #'streaker'
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None
        self.v_mean = 6.7 #mean velocity of uninformed bees from paper
        self.theta_r = np.sqrt(self.r_max)/self.v_max
        self.m_timesteps = 0

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv2_v2, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        leader_dict = {1: "streaker", 0: "passive_leader"}
        assert u.shape == (self.n_agents, self.nu)
        self.n_timesteps += 1
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u
        theta = np.pi/400 * self.m_timesteps

        #uninformed bees
        # x, y position
        self.x[self.n_leaders:, 0] = self.x[self.n_leaders:, 0] + self.x[self.n_leaders:, 2] * self.dt + self.u[self.n_leaders:, 0] * self.dt * self.dt * 0.5
        self.x[self.n_leaders:, 1] = self.x[self.n_leaders:, 1] + self.x[self.n_leaders:, 3] * self.dt + self.u[self.n_leaders:, 1] * self.dt * self.dt * 0.5
        # x, y velocity
        self.x[self.n_leaders:, 2] = self.x[self.n_leaders:, 2] + self.u[self.n_leaders:, 0] * self.dt
        self.x[self.n_leaders:, 3] = self.x[self.n_leaders:, 3] + self.u[self.n_leaders:, 1] * self.dt

        #informed bees
        # x, y position
        self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
        self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
        #(1,2)if leader_mode == 1 and max((y-v_max)/x) < cot(theta + theta_r): leader_velocity = continue going
        if self.leader_mode==1 and max((self.x[0:self.n_leaders,1]-self.v_max)/self.x[0:self.n_leaders,0]) < 1/np.tan(theta + self.theta_r):
            self.m_timesteps += 1
            t = np.pi/4 * self.m_timesteps * self.dt
            self.x[0:self.n_leaders, 2] = np.ones((self.n_leaders,)) * -self.v_max * np.sin(t - np.pi/2)
            self.x[0:self.n_leaders, 3] = np.ones((self.n_leaders,)) * self.v_max * np.cos(t - np.pi/2)
        #(3)elif leader_mode == 1 and max((y-v_max)/x) > cot(theta + theta_r): mode=passive and leader_vel = - (1)
        elif self.leader_mode==1 and max((self.x[0:self.n_leaders,1]-self.v_max)/self.x[0:self.n_leaders,0]) > 1/np.tan(theta + self.theta_r):
            self.leader_mode = 0 #'passive_leader'
            self.m_timesteps -= 1
            t = np.pi/4 * self.m_timesteps * self.dt
            self.x[0:self.n_leaders, 2] = np.ones((self.n_leaders,)) * self.v_max * np.sin(t - np.pi/2)
            self.x[0:self.n_leaders, 3] = np.ones((self.n_leaders,)) * -self.v_max * np.cos(t - np.pi/2)
        #(4)elif leader_mode == 0 and min((y-v_max)/x) < cot(theta - theta_r): mode=active and leader_vel = (1)
        elif self.leader_mode==0 and min((self.x[0:self.n_leaders,1]-self.v_max)/self.x[0:self.n_leaders,0]) < 1/np.tan(theta - self.theta_r):
            self.leader_mode = 1 #'active_leader'
            self.m_timesteps += 1
            t = np.pi/4 * self.m_timesteps * self.dt
            self.x[0:self.n_leaders, 2] = np.ones((self.n_leaders,)) * -self.v_max * np.sin(t - np.pi/2)
            self.x[0:self.n_leaders, 3] = np.ones((self.n_leaders,)) * self.v_max * np.cos(t - np.pi/2)
        #(5,6)
        else:
            self.m_timesteps -= 1
            t = np.pi/4 * self.m_timesteps * self.dt
            self.x[0:self.n_leaders, 2] = np.ones((self.n_leaders,)) * self.v_max * np.sin(t - np.pi/2)
            self.x[0:self.n_leaders, 3] = np.ones((self.n_leaders,)) * -self.v_max * np.cos(t - np.pi/2)
        #sol2) if leader> front 10% of flock, x(velocity)==0
        
        if self.m_timesteps > 199:
            self.done = True
            # x in nest?
            cond1 = self.x[:,0] >= self.v_max - self.nest_R
            cond2 = self.x[:,0] <= self.v_max + self.nest_R
            cond3 = self.x[:,1] >= self.v_max
            self.x_in_nest = cond1 & cond2 & cond3
            self.S_in_nest += sum(self.x_in_nest)
            self.S_timesteps += self.n_timesteps
            print('\nn_timesteps: ',self.n_timesteps)
            print('n_agents in nest: ',sum(self.x_in_nest))

        self.compute_helpers(self.leader_mode)
        return (self.state_values, self.state_network), self.instant_cost(self.leader_mode), self.done, {}

    def reset(self):
        super(FlockingLeaderEnv2_v2, self).reset()
        self.leader_mode = 1 #active leader
        # self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
        #                                                                                  high=self.v_max, size=(1, 1))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
                                                                                                                                                                          
        return (self.state_values, self.state_network)

    def render(self, mode='human'):
        super(FlockingLeaderEnv2_v2, self).render(mode)
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
        print('\nself.average_timesteps: ',self.S_timesteps/self.n_test_episodes)
        print('average_# of_agents in nest: ',self.S_in_nest/self.n_test_episodes)
        pass