import numpy as np

r_max = 3
n_agents=10
goal_x = 10
nest_R = 5/4*np.sqrt(r_max * np.sqrt(n_agents))
x_in_nest = np.zeros(n_agents)

x = np.zeros((n_agents, 2))

x[1,:]=[11,0]
x[3,:]=[11,0]
x[5,:]=[11,0]
x_in_nest[1] = 1
print('x[:,0]:\n',x[:,0])
print('x_in_nest\n:',x_in_nest)

#using enumerate --> working but still slow
'''for i,v in enumerate(x_in_nest):
    if v > 0: continue
    if x[i,0] >= goal_x and x[i,1] >= -nest_R and x[i,1] < nest_R + 1:
        x_in_nest[i] = 1
    print(i)'''

#idea from matlab --> need improvement
# if x[:,0]>= goal_x and x[:,1]>= -nest_R and x[i,1] < nest_R + 1:
#     x_in_nest[:]=1

#using np.where
'''np.where(x[:,0] >= goal_x,1,x_in_nest)
print('x[:,0]:',x[:,0])'''

#using np.all()
'''print(x[:,0] >=  goal_x )
print(type(x[:,0] >=  goal_x )) #is class numpy.ndarray
print((x[:,0] >= goal_x ).all()) #False--> one result'''

#using np.logical_and()
cond1 = x[:,0] >= goal_x
cond2 = x[:,1]>= -nest_R
cond3 = x[:,1] < nest_R + 1
print(cond1 & cond2 & cond3)
x_in_nest = cond1 & cond2 & cond3
print('x_in_nest:\n',x_in_nest)
print(sum(x_in_nest))