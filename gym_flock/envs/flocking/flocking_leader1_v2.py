import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingLeaderEnv1_v2(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv1_v2, self).__init__()
        self.n_leaders = 4 #2
        self.leader_mode = 1 #'streaker'
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None
        # self.half_leaders = int(self.n_leaders / 2.0)
        self.m_timesteps = 1

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv1_v2, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.v_hist = np.zeros((210,2))
        self.v_hist[0,:] = [self.v_max, 0]
        t = np.pi/4 * np.arange(1,210) * self.dt
        x = np.ones((2,210-1))*[-self.v_max * np.sin(t - np.pi/2), self.v_max * np.cos(t - np.pi/2)]
        self.v_hist[1:210, :] = x.T
        self.Rx_final = sum(self.v_hist[:,0]) * self.dt
        self.Ry_final = sum(self.v_hist[:,1]) * self.dt
        print('Rx: {}'.format(self.Rx_final))
        print('Ry: {}'.format(self.Ry_final))
        self.theta_r = np.sqrt(self.r_max)/self.Ry_final

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        self.n_timesteps += 1
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u
        self.theta = np.pi/400 * self.m_timesteps

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
        # if self.leader_mode==1 and max((self.x[self.n_leaders:,1]-self.Ry_final)/self.x[self.n_leaders:,0]) > -1/np.tan(self.theta + self.theta_r):
        if self.leader_mode==1 and \
            max(self.x[self.n_leaders:,0]/(self.Ry_final - self.x[self.n_leaders:,1])) > max(self.x[0:self.n_leaders,0]/(self.Ry_final - self.x[0:self.n_leaders,1])):
            
            self.m_timesteps += 1
            t = np.pi/4 * self.m_timesteps * self.dt
            self.x[0:self.n_leaders, 2] = -self.v_max * np.sin(t - np.pi/2)
            self.x[0:self.n_leaders, 3] = self.v_max * np.cos(t - np.pi/2)

        #(3)elif leader_mode == 1 and max((y-v_max)/x) > cot(theta + theta_r): mode=passive and leader_vel = - (1)
        # elif self.leader_mode==1 and max((self.x[self.n_leaders:,1]-self.Ry_final)/self.x[self.n_leaders:,0]) <= -1/np.tan(self.theta + self.theta_r):
        elif self.leader_mode==1 and \
            max(self.x[self.n_leaders:,0]/(self.Ry_final - self.x[self.n_leaders:,1])) <= max(self.x[0:self.n_leaders,0]/(self.Ry_final - self.x[0:self.n_leaders,1])):
            
            self.leader_mode = 0 #'passive_leader'
            self.x[0:self.n_leaders,2:4] = 0 #x,y vel=0

        #(4)elif leader_mode == 0 and min((y-v_max)/x) < cot(theta - theta_r): mode=active and leader_vel = (1)
        # elif self.leader_mode==0 and min((self.x[self.n_leaders:,1]-self.Ry_final)/self.x[self.n_leaders:,0]) > -1/np.tan(self.theta - self.theta_r):
        elif self.leader_mode==0 and \
            np.mean(self.x[self.n_leaders:,0]/(self.Ry_final - self.x[self.n_leaders:,1])) > np.mean(self.x[0:self.n_leaders,0]/(self.Ry_final - self.x[0:self.n_leaders,1])):
            
            self.leader_mode = 1 #'active_leader'
            self.m_timesteps += 1
            t = np.pi/4 * self.m_timesteps * self.dt
            self.x[0:self.n_leaders, 2] = -self.v_max * np.sin(t - np.pi/2)
            self.x[0:self.n_leaders, 3] = self.v_max * np.cos(t - np.pi/2)
        #(5,6)
        else:
            self.x[0:self.n_leaders,2:4] = 0 #x,y vel=0

        #sol2) if leader> front 10% of flock, x(velocity)==0

        # x in nest? using enumerate
        '''for i,v in enumerate(self.x_in_nest):
            # print(x[i,0:2])
            if v > 0: continue
            if self.x[i,0] >= self.goal_x and self.x[i,1] >= -self.nest_R and self.x[i,1] < self.nest_R + 1:
                self.x_in_nest[i] = 1
                print('n_x_in_nest: ',sum(self.x_in_nest),'in step')'''
        
        if self.m_timesteps > 210-1:
            self.done = True
            # x in nest?
            self.x_in_nest = np.square(self.x[:,0]-self.Rx_final)+np.square(self.x[:,1]-self.Ry_final) <= np.square(self.nest_R)
            self.S_in_nest += sum(self.x_in_nest)
            self.S_timesteps += self.n_timesteps
            print('n_timesteps: ',self.n_timesteps)
            print('n_agents in nest: ',sum(self.x_in_nest))

        if self.n_timesteps > 300 -1:
            self.done = True
            # x in nest?
            self.x_in_nest = np.square(self.x[:,0]-self.Rx_final)+np.square(self.x[:,1]-self.Ry_final) <= np.square(self.nest_R)
            self.S_in_nest += sum(self.x_in_nest)
            self.S_timesteps += self.n_timesteps
            print('n_timesteps: ',self.n_timesteps)
            print('n_agents in nest: ',sum(self.x_in_nest))

        self.compute_helpers(self.leader_mode)
        return (self.state_values, self.state_network), self.instant_cost(self.leader_mode), self.done, {}

    def reset(self):
        super(FlockingLeaderEnv1_v2, self).reset()
        self.leader_mode = 1 #active
        self.m_timesteps = 1
        # self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
        #                                                                                  high=self.v_max, size=(1, 1))
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * [[self.v_max, 0]]                                                                                                    
        return (self.state_values, self.state_network)

    def render(self, mode='human'):
        super(FlockingLeaderEnv1_v2, self).render(mode)
        # self.ax.plot([self.goal_x,self.goal_x],[-self.nest_R,self.nest_R])
        # self.ax.plot([self.Rx_final - self.nest_R, self.Rx_final + self.nest_R],[self.Ry_final, self.Ry_final])

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