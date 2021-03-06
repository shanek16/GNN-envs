import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv1(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv1, self).__init__()
        self.n_leaders = 4 #2
        self.leader_mode = 1 #'streaker'
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None
        self.half_leaders = int(self.n_leaders / 2.0)

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv1, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        self.n_timesteps += 1
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u
        leader_dict = {1: "streaker", 0: "passive_leader"}

        #uninformed bees
        # x, y position
        self.x[self.n_leaders:, 0] = self.x[self.n_leaders:, 0] + self.x[self.n_leaders:, 2] * self.dt + self.u[self.n_leaders:, 0] * self.dt * self.dt * 0.5
        self.x[self.n_leaders:, 1] = self.x[self.n_leaders:, 1] + self.x[self.n_leaders:, 3] * self.dt + self.u[self.n_leaders:, 1] * self.dt * self.dt * 0.5
        # x, y velocity
        self.x[self.n_leaders:, 2] = self.x[self.n_leaders:, 2] + self.u[self.n_leaders:, 0] * self.dt
        self.x[self.n_leaders:, 3] = self.x[self.n_leaders:, 3] + self.u[self.n_leaders:, 1] * self.dt

        #informed bees
        #sol1) 
        #if max(leader_position) > max(x(position)): mode=passive and leader_velocity==0
        if max(self.x[0:self.n_leaders,0]) > max(self.x[self.n_leaders:,0]):
            self.leader_mode = 0 #'passive_leader'
            self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
            self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
            self.x[0:self.n_leaders,2] = 0 #x vel=0
            self.x[0:self.n_leaders,3] = 0 #y vel=0
        #else if     passive mode   and min(leader_position) < min(x(position)): mode=active and leader_vel=max_v
        elif self.leader_mode==0 and min(self.x[0:self.n_leaders,0]) < min(self.x[self.n_leaders:,0]):
            self.leader_mode = 1 #active leader
            self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
            self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
            self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
        else:
            # x, y position
            self.x[0:self.n_leaders, 0] = self.x[0:self.n_leaders, 0] + self.x[0:self.n_leaders, 2] * self.dt
            self.x[0:self.n_leaders, 1] = self.x[0:self.n_leaders, 1] + self.x[0:self.n_leaders, 3] * self.dt
            # # x, y velocity ->>>>>>>>>>>>>>>>>>>idle?
            # self.x[:, 2] = self.x[:, 2]
            # self.x[:, 3] = self.x[:, 3]

        #sol2) if leader> front 10% of flock, x(velocity)==0

        # x in nest? using enumerate
        '''for i,v in enumerate(self.x_in_nest):
            # print(x[i,0:2])
            if v > 0: continue
            if self.x[i,0] >= self.goal_x and self.x[i,1] >= -self.nest_R and self.x[i,1] < self.nest_R + 1:
                self.x_in_nest[i] = 1
                print('n_x_in_nest: ',sum(self.x_in_nest),'in step')'''
        
        if min(self.x[:,0]) > self.goal_x:
            # x in nest?
            # cond1 = self.x[:,0] >= self.goal_x
            cond2 = self.x[:,1]>= -self.nest_R
            cond3 = self.x[:,1] < self.nest_R + 1
            self.x_in_nest = cond2 & cond3 #& cond1

            self.done = True
            self.S_timesteps += self.n_timesteps
            self.S_in_nest += sum(self.x_in_nest)
            print('\nn_timesteps: ',self.n_timesteps)
            print('n_agents in nest: ',sum(self.x_in_nest))

        self.compute_helpers(self.leader_mode)
        return (self.state_values, self.state_network), self.instant_cost(self.leader_mode), self.done, {}

    def reset(self):
        super(FlockingLeaderEnv1, self).reset()
        self.leader_mode = 1 #active
        # self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
        #                                                                                  high=self.v_max, size=(1, 1))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]
                                                                                                                                                                          
        return (self.state_values, self.state_network)

    def render(self, mode='human'):
        super(FlockingLeaderEnv1, self).render(mode)
        self.ax.plot([self.goal_x,self.goal_x],[-self.nest_R,self.nest_R])

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