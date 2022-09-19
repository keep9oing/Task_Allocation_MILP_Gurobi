# Task_Allocation_MILP_Gurobi
Devloping with the Gurobi (Academic License). <br>
Euclidean distance cost based multi robot task allocation with MILP(Mixed Intger Linear Programming) modeling.

> gurobi_minsum.py is 2D decision variable version. <br>
> gurobi_minsum_2.py and gurobi_minmax.py are 3D decision variable version.

### Update (22.09.19)
Since the naive implementation of subtour elimination constraints requires too much cost(memory, computation time), lazy constraint callback version for each algorithm is implemented.

![minsum](https://user-images.githubusercontent.com/31655488/190587459-21621a07-48dd-4bfa-8c54-17801cdfa082.png)
![minmax](https://user-images.githubusercontent.com/31655488/190587477-54abbe4c-653b-49ac-9217-76239737c76e.png)
