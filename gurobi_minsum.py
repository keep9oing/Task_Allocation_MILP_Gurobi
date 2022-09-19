import gurobipy as gp
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.spatial import distance_matrix
from gurobipy import GRB
from itertools import combinations


TIME_LIMIT = None # if you want time limit for solver
LOG_TO_CONSOLE = True # False if you don't want to print log on the console

# robot number
r_num = 2
# task number
t_num = 13

# index for robot node
V_R = [ i for i in range(r_num)]
# index for task node
V_T = [ i for i in range(t_num)]
# index for decision vairable
V_ID = V_T + [ i + t_num for i in range(r_num)]

# random robt & task location
R_loc = [[random.uniform(0,1),random.uniform(0,1)] for i in range(r_num)]
T_loc = [[random.uniform(0,1),random.uniform(0,1)] for i in range(t_num)]
# location stack for distance matrix
loc =  T_loc + R_loc
loc = np.array(loc)

# travel time matrix (velocity of robot is 1.0)
travel_time_mat = distance_matrix(loc, loc)
travel_time = dict()

# dictionary for gurobi
for r in V_ID:
    for c in range(len(V_T)):
        travel_time[r,c] = travel_time_mat[r,c]

# create model
m = gp.Model("ta_minsum")

# decision variable
x = m.addVars(V_ID, V_T, vtype=GRB.BINARY, name="x")

# every task should be allocated
m.addConstrs(x.sum('*',j) == 1 for j in V_T)
# each source node can have at most 1 target node
m.addConstrs(x.sum(i,'*') <= 1 for i in V_ID)
# prevent select same node for target node
m.addConstrs((x[i,i] == 0 for i in V_T), name="removeSame")

# subtour elimination
for degree in range(2, len(V_T)+1):
    if degree > 2:
        for comb in list(combinations(V_T, degree)):
            m.addConstr(gp.quicksum(x[i,j] + x[j,i] for i, j in combinations(comb, 2)) <= degree - 1)
    else:
        m.addConstrs(gp.quicksum([x[i,j],x[j,i]]) <= degree - 1 for i, j in combinations(V_T,2))

# minsum objective
m.setObjective(x.prod(travel_time), GRB.MINIMIZE)
if TIME_LIMIT is not None:
    m.Params.timeLimit = TIME_LIMIT
m.Params.LogToConsole = LOG_TO_CONSOLE
m.optimize()

# get solution from model
solution = np.zeros((len(V_ID), len(V_T)))
for i in V_ID:
    for j in V_T:
        solution[i,j] = x[i,j].X

# construct solution list from solution matrix
robot = [[] for _ in range(r_num)]

for r in V_R:
    n = r+t_num
    while sum(solution[n,:])>0.5:
        n = np.where(solution[n,:]>0.5)[0][0]
        robot[r].append(n)

# Plot solution
r_loc = np.array(R_loc)
t_loc = np.array(T_loc)
plt.plot(r_loc[:,0],r_loc[:,1],'bo', label="robot")
plt.plot(t_loc[:,0],t_loc[:,1],'ro', label="task")

for r_id, route in enumerate(robot):
    if len(route) > 0:
        plt.plot( [r_loc[r_id,0],t_loc[route[0],0]], [r_loc[r_id,1],t_loc[route[0],1]],'g-')
        plt.plot(t_loc[route,0],t_loc[route,1], 'g-')


plt.title("MinSum")
plt.xlim((0,1))
plt.ylim((0,1))
plt.axis('equal')
plt.legend()
plt.show()
