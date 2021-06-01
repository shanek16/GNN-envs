import numpy as np

r_max = 3
n_agents=10
goal_x = 10
nest_R = 5/4*np.sqrt(r_max * np.sqrt(n_agents))
x_in_nest = np.zeros(n_agents)

x = np.zeros((n_agents, 2))

x[1,:]=[11,0]

for i in range(n_agents):
    print(i)
    if x[i,0] >= goal_x and x[i,1] >= -nest_R and x[i,1] < nest_R + 1:
        x_in_nest[i] = 1
print('x_in_nest:\n',x_in_nest)
print(sum(x_in_nest))