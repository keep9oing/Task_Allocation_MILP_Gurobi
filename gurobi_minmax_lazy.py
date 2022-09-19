import gurobipy as gp
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.spatial import distance_matrix
from gurobipy import GRB
from itertools import combinations, permutations

import networkx as nx


# robot and task number
r_num = 2
t_num = 13

# indices for robot and task node
V_R = [ i for i in range(r_num)]
V_T = [ i for i in range(t_num)]

# random robot and task location
R_loc = [[random.uniform(0,1),random.uniform(0,1)] for i in range(r_num)]
T_loc = [[random.uniform(0,1),random.uniform(0,1)] for i in range(t_num)]

# Special Case for Test (Uncomment below code block to test)
# r_num = 2
# t_num = 3
# T_loc = [[0,2],[1,2],[1,2.5]]
# R_loc = [[0,0],[1,0]]
# V_R = [ i for i in range(r_num)]
# V_T = [ i for i in range(t_num)]

# calcuate travel time matrix for each robot
travel_time_mat = np.zeros((t_num+1,t_num+1,r_num))
for r in range(r_num):
    loc = T_loc + [R_loc[r]]
    travel_time_mat[:,:,r] = distance_matrix(loc,loc)

# travel time dictionary for gurobi
travel_time = dict()
for k in V_R:
    for r in range(t_num+1):
        for c in range(t_num):
            travel_time[r,c,k] = travel_time_mat[r,c,k]
for k in V_R:
    for r in range(t_num+1):
        travel_time[r,t_num+1,k] = 0

# index for decision variable

V_ID = V_T+[len(V_T)]
V_C = V_T + [t_num]

# create model
m = gp.Model("mrta_minmax")

# decision variable
x = m.addVars(V_ID, V_C, V_R, vtype=GRB.BINARY, name="x")
# variable for minmax objective
z = m.addVar(vtype=GRB.CONTINUOUS, name='z')

# every task should be assgined
m.addConstrs(gp.quicksum(x.sum('*',j,k) for k in V_R) == 1 for j in V_T)

# every source node need one target (include no target index)
m.addConstrs(gp.quicksum(x.sum(i,'*',k) for k in V_R) == 1  for i in V_T)

# robot can be unused
m.addConstrs(x.sum(t_num,'*',k) <= 1  for k in V_R)

# the number of finish taget should be same as the number of robot
m.addConstr(gp.quicksum(x.sum('*',t_num,k) for k in V_R) == r_num)

# in&out constraint
m.addConstrs(x.sum('*', i, k) - x.sum(i,'*', k) == 0 for i in V_T for k in V_R)

# prevent select same node
m.addConstr(gp.quicksum(x.sum(i,i,'*') for i in V_T) == 0)

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in permutations(V_T, 2) if sum([vals[i,j,k] for k in range(r_num)]) >= 0.5)
        # find the shortest cycle in the selected edge list
        # print("selected:",selected)
        tour = subtour(selected)
        # add subtour elimination constr. for every pair of cities in tour
        # print("tour:",tour)
        if tour:
            model.cbLazy(gp.quicksum(model._vars.sum(i, j,'*')+model._vars.sum(j,i,'*') for i, j in combinations(tour, 2)) <= len(tour)-1)

# Given a tuplelist of edges, find the shortest subtour
def subtour(edges):
    cycle_list = []
    G = nx.DiGraph()
    for i, j in edges:
        G.add_edge(i, j)
    for cycle in nx.simple_cycles(G):
        cycle_list.append(cycle)

    if cycle_list:
        return min(cycle_list,key=len)
    else:
        return cycle_list

# constraints for minmax objective
for k in V_R:
    m.addConstr(gp.quicksum(gp.quicksum(x[i,j,k] * travel_time[i,j,k] for j in V_T) for i in V_ID) <= z)

# minmax opjective
m.setObjective(z, GRB.MINIMIZE)

m._vars = x
m.Params.lazyConstraints = 1
m.optimize(subtourelim)

# get solution matrix from model
solution = np.zeros((t_num+1, t_num+1, r_num))
for k in V_R:
    for i in V_ID:
        for j in V_C:
            solution[i,j,k] = x[i,j,k].X

# construct solution for each robot
robot = [[] for _ in range(r_num)]
for r in V_R:
    n = t_num
    while sum(solution[n,:,r])>0.5 and solution[n,-1,r] < 0.1:
        n = np.where(solution[n,:,r]>0.5)[0][0]
        robot[r].append(n)

# plot result
r_loc = np.array(R_loc)
t_loc = np.array(T_loc)
plt.plot(r_loc[:,0],r_loc[:,1],'bo',label="robot")
plt.plot(t_loc[:,0],t_loc[:,1],'ro',label="task")

for r_id, route in enumerate(robot):
    if len(route) > 0:
        plt.plot( [r_loc[r_id,0],t_loc[route[0],0]], [r_loc[r_id,1],t_loc[route[0],1]],'g-')
        plt.plot(t_loc[route,0],t_loc[route,1], 'g-')

plt.title("MinMax({:.2f})".format(m.ObjVal))
plt.xlim((0,1))
plt.ylim((0,1))
plt.axis('equal')
plt.legend()
plt.show()
