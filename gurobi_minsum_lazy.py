import gurobipy as gp
from matplotlib import pyplot as plt
import numpy as np
import random
from scipy.spatial import distance_matrix
from gurobipy import GRB
from itertools import combinations, permutations

import networkx as nx

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

# T_loc = [[0,2],[1,2],[1,2.5]]
# R_loc = [[0,0],[1,0]]

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

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in permutations(V_T, 2) if vals[i, j] >= 0.5)
        # find the shortest cycle in the selected edge list
        # print("selected:",selected)
        tour = subtour(selected)
        # add subtour elimination constr. for every pair of cities in tour
        # print("tour:",tour)
        if tour:
            model.cbLazy(gp.quicksum(model._vars[i, j]+model._vars[j, i] for i, j in combinations(tour, 2)) <= len(tour)-1)

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

# minsum objective
m.setObjective(x.prod(travel_time), GRB.MINIMIZE)

m._vars = x
m.Params.lazyConstraints = 1
m.optimize(subtourelim)

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
