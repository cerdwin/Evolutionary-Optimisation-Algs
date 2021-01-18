Application of genetic algorithm to the problem of Travelling Salesman

The code has been implemented in the file new_salesman.cpp, which has been compiled with 'g++ -g new_salesman.cpp '.
As input, (.in) files were used, which first line contains the number of cities available (n) and subsequent n lines contain
three numbers: each city's identification number and two subsequent coordinates in the xy plane. This structure has been chosen
to fit with TSP problems in TSPLIB(http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/). Please see the file 'pub01.in' attached for reference of format of input files.

Modifications of configurations of the algorithm hare done mostly by changing macros, as described further in the report. For instance, changing the macro ALGORITHM
as per comment adjacent to its definition, can make the script run for example Simulated Annealing, memetic Genetic Algorithm or a greedy algorithm of nearest neighbour search
to name a few. Similarly, parameters ranging from population sizes, SA temperature cooling schedule parameters or perturbations used can be modified this way.


