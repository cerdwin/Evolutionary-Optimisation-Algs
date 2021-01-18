import random
import math
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_application, convert_xor
transformations = standard_transformations + (implicit_application, convert_xor)

pi = math.pi
best_so_far = None
def cos(x):
    return math.cos(x)

def sin(x):
    return math.sin(x)

def tan(x):
    return math.tan(x)

def population_generator(pop_size,x_min, x_max, y_min, y_max):
    """
    generates new x-y coordinate pairs making up the first generation
    :param pop_size: usually 100
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return: list of coordinate tuples
    """
    ret = []
    pi = math.pi
    while len(ret)<pop_size:
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(y_min, y_max)
        if (x1, x2) not in ret:
            ret.append((x1, x2))
    return ret


def evaluate(population, function):
    ret = {}
    for solution in population:
        pi = math.pi
        x1 = solution[0]
        x2 = solution[1]
        ret[solution] = eval(function)

    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))

def to_swap(first, second, first_constraint, second_constraint):
    x1 = first[0]
    x2 = first[1]
    first_breach = eval(first_constraint)
    second_breach = eval(second_constraint)
    tmp = 0
    if first_breach>0:
        tmp+=first_breach
    if second_breach>0:
        tmp+=second_breach
    feasible_first = False

    x1 = second[0]
    x2 = second[1]
    first_breach = eval(first_constraint)
    second_breach = eval(second_constraint)
    tmp2 = 0
    if first_breach>0:
        tmp2+=first_breach
    if second_breach>0:
        tmp2+=second_breach
    if feasible_first is False and first_breach<=0 and second_breach<=0:
        #print("something feasibble at:", first)
        return True
    if feasible_first and (first_breach>0 or second_breach>0):
        #print("something other feasibble at:", first)
        return False
    if tmp2 < tmp:
        return True
    return False

def both_feasible(first, second, first_constraint, second_constraint):
    candidates = [first, second]
    for value in candidates:
        x1 = value[0]
        x2 = value[1]
        first_breach = eval(first_constraint)
        second_breach = 0
        if second_constraint:
            second_breach = eval(second_constraint)
        if first_breach>0 or second_breach>0:
            return False
    #print("both feasible")
    return True

def selection(population, first_constraint, second_constraint):
    """
    A 'selection' part of the GA, selecting 40 candidates to sire new generation of offsprings selected through a tournament, beforehand performing stochastic ranking
    :param population: a dictionary of a tuple representing a point in 2d space and a score assigned to it by our objective function
    :return: A population selected to make the new generation
    """
    pre_sorted = list(population.keys())
##### bubble-sort like mechanism placing the "best" on the left
    for i in range(len(pre_sorted)):
        for j in range(1, len(pre_sorted)-1):
            die = random.uniform(0,1)
            if die < 0.46 or both_feasible(pre_sorted[j], pre_sorted[j+1], first_constraint, second_constraint): # comparing based off on the objective function alone
                if population[pre_sorted[j]]>population[pre_sorted[j+1]]:
                    tmp = pre_sorted[j]
                    pre_sorted[j] = pre_sorted[j + 1]
                    pre_sorted[j + 1] = tmp
            else:
                if to_swap(pre_sorted[j], pre_sorted[j+1], first_constraint, second_constraint):
                    tmp = pre_sorted[j]
                    pre_sorted[j] = pre_sorted[j+1]
                    pre_sorted[j+1] = tmp
    sorting_key = dict(zip(pre_sorted, list(range(len(pre_sorted)))))
    to_select = 40
    tournament_size = 6
    ret = {}

    #####
    #print("///////////////////////////////////////////////////////////")
    #for item in list(sorting_key.keys()):
    #    x1 = item[0]
    #    x2 = item[1]
        #print("item:", item, "first constraint:", eval(first_constraint), "second constraint", eval(second_constraint), "function:", eval(function))
    #####
   # print("///////////////////////////////////////////////////////////")

    while len(ret) != to_select:
        tmp = {}
        while len(tmp)< tournament_size:
            selected = random.choice(list(population.items()))
            tmp[selected[0]] = sorting_key[selected[0]]
        tournament_winner = min(tmp, key=tmp.get)
        ret[tournament_winner] = population[tournament_winner]
    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))


def offsprings(population, first_constraint, second_constraint):
    ret = []
    counter = 0
    while len(ret) != 100:
        mother = random.choice(list(population.items()))
        mother_x = mother[0][0]
        mother_y = mother[0][1]
        father = random.choice(list(population.items()))
        father_x = father[0][0]
        father_y = father[0][1]
        heritability = random.random()
        x1 = heritability*mother_x+(1-heritability)*father_x
        x2 = heritability*mother_y+(1-heritability)*father_y
        if [x1, x2] not in ret:
            ret.append((x1, x2))
            counter += 1
        x1 = heritability * father_x + (1 - heritability) * mother_x
        x2 = heritability * father_y + (1 - heritability) * mother_y

        if [x1, x2] not in ret:
            ret.append((x1, x2))
            counter += 1

    return ret


def mutate(population, x1_mutation_range, x2_mutation_range, first_constraint, second_constraint, function, x1_min, x1_max, x2_min, x2_max):
    #print('started process of mutation')
    ret = {}
    counter = 0
    for child in population:
        possible_versions = []
        while len(possible_versions)<5: ## we want to create 5 possible versions and pick the best
            x1 = random.uniform(child[0]-x1_mutation_range/2, child[0]+x1_mutation_range/2)
            x2 = random.uniform(child[1]-x2_mutation_range/2, child[1]+x2_mutation_range/2)
            if x1 < x1_min or x1> x1_max or x2<x2_min or x2 > x2_max or (x1, x2) in ret.keys(): #or eval(first_constraint)>0 or eval(second_constraint)>0 :

                continue
            possible_versions.append([x1, x2])
        best = None
        minimum = None
        for version in possible_versions:
            x1 = version[0]
            x2 = version[1]
            r1 = eval(function)
            if best is None or minimum is None or r1 < minimum:
                best = [x1, x2]
                minimum = r1
        ret[(best[0], best[1])] = minimum
    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))

def validate(item, first_constraint, second_constraint, function):
    x1 = item[0]
    x2 = item[1]
    if eval(first_constraint)>0:
        return 99999999999
    if second_constraint:
        if eval(second_constraint)>0:
            return 99999999999
    return eval(function)


def ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint, second_constraint):
    initial_population = population_generator(100, x_min, x_max, y_min, y_max)
    evaluated_population = evaluate(initial_population, function)
    best_so_far = None
    best_result = 999999999
    for i in range(generations):
        print("generation", i , "out of ", generations, "generations")
        selected_population = selection(evaluated_population, first_constraint, second_constraint)
        for item in selected_population:
            if best_so_far is None or validate(item, first_constraint, second_constraint, function)<best_result:
                best_so_far = item
                best_result = validate(item, first_constraint, second_constraint, function)
        new_gen_before = offsprings(selected_population, first_constraint, second_constraint)
        mutated = mutate(new_gen_before, (x_max - x_min) * mutation_range, (y_max - y_min) * mutation_range,
                         first_constraint, second_constraint, function, x_min, x_max, y_min, y_max)
        evaluated_population = mutated
        mutation_range = mutation_range * 0.99


    print("best result is:", best_so_far, "with function value of:", best_result)




if __name__ == '__main__':
    model = 'g06'
    generations = 100
    mutation_range = 0.05  # parameter used for uniform mutation employed
    if model == 'g06': # ok pf = 0.4
        ## g06
        x_min = 13
        x_max = 100
        y_min = 0
        y_max = 100
        function = '(x1-10)**3+(x2-20)**3'
        first_constraint = '-(x1-5)**2-(x2-5)**2+100'
        second_constraint = '(x1-6)**2+(x2-5)**2-82.81'

        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint, second_constraint)

    elif model == 'g08': #pf is 0.475
        x_min = 0
        x_max = 10
        y_min = 0
        y_max = 10
        function = '-((sin(2 * pi * x1) ** 3) * (sin(2 * pi * x2))) / ((x1 ** 3) * (x1 + x2))'
        first_constraint = 'x1**2-x2+1'
        second_constraint = '1-x1+(x2-4)**2'
        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
           second_constraint)

    elif model == 'g11': # ok
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        function = 'x1**2+(x2-1)**2'
        first_constraint = 'x2-x1**2'
        second_constraint = '-1'
        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
           second_constraint)
    elif model == 'g24':
        x_min = 0
        x_max = 3
        y_min = 0
        y_max = 4
        function = '-x1-x2'
        first_constraint = '-2*x1**4+8*x1**3-8*x1**2-2'
        second_constraint = '-4*x1**4+32*x1**3-88*x1**2+96*x1+x2-36'
        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
           second_constraint)






