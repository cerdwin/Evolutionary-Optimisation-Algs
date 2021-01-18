import random
import math
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_application, convert_xor
transformations = standard_transformations + (implicit_application, convert_xor)

pi = math.pi

def cos(x):
    return math.cos(x)

def sin(x):
    return math.sin(x)

def tan(x):
    return math.tan(x)



def population_generator(pop_size, x_min, x_max, y_min, y_max, first_constraint, second_constraint, function):
    '''
    A function specific for the NSGA-II algorithm, takes in constraints along with pop. size and domain and range limits
    and outputs a list where first position is occupied by a dictionary of keys representing generated points and their values
    values w.r.t. the function we are trying to minimise and second is a dict with the same keys and sum of breach of
    constraints provided.
    :param pop_size:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param first_constraint:
    :param second_constraint:
    :return:
    '''
    first_dict = {}
    second_dict = {}
    counter = 0

    while len(first_dict.keys()) < pop_size:
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(y_min, y_max)
        r0 = eval(function)
        r1 = 0
        r2 = 0
        # if second constraint is None, we know we're dealing with g11, which is the only constraint g(x) = 0 as opposed to <=0

        if second_constraint:
            if eval(first_constraint) > 0:
                r1 = eval(first_constraint)
            if eval(second_constraint)>0:
                r2 = eval(second_constraint)
        else:
            r1 = abs(0-eval(first_constraint))

        if (x1, x2) not in first_dict :
            first_dict[(x1, x2)] = r0
            second_dict[(x1, x2)] = r1+r2
            counter += 1

    return first_dict, second_dict


def evaluate(population, function):
    ret = {}
    for solution in population:
        x1 = solution[0]
        x2 = solution[1]
        ret[solution] = eval(function)

    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))


def binary_tournament_operator(first, second):
    '''
    As part of the NSGA2 algorithm, the function gets two lists in the format [Frontier1, crowding distance] and returns true if the first is better
    :param first:
    :param second:
    :return:
    '''
    if first[0] < second[0] or (first[0] == second[0] and first[1] < second[1]):
        return True
    return False


def selection(population):
    """
    A 'selection' part of the GA, selecting 40 candidates to sire new generation of offsprings selected through a tournament
    :param population: a dictionary of a tuple representing a point in 2d space and a score assigned to it by our objective function
    :return: A population selected to make the new generation
    """
    to_select = 30
    tournament_size = 6
    ret = {}
    while len(ret) != to_select:
        tmp = {}
        pivot = random.choice(list(population.items()))
        for i in range(tournament_size):
            selected = random.choice(list(population.items()))
            if binary_tournament_operator(selected[1], pivot[1]):
                pivot = selected
            tmp[pivot[0]] = pivot[1]
        ret[pivot[0]] = pivot[1]
    return ret









def replacement_strategy(parents, children, populations):
    '''
    takes in a population of parents and kids, joins them, works out crowding distance and non-dominated fronts and returns a new population of the best of them
    :return:
    '''
    merged = {**parents, **children}
    evaluated = nd_front(merged, populations)
    to_select = 100  # let's say we need 100 new members of our population
    ret = {}  # a dictionary with a tuple symbolising a position for hey and a list with Frontier id and crowding distance as value
    best_key, best_value = random.choice(evaluated.items())
    while len(ret) != 100:
        for member in evaluated.keys():
            if binary_tournament_operator(evaluated[member], best_value):
                best_key = member
                best_value = evaluated[member]
        evaluated.pop(best_key)
        ret[best_key] = best_value
        best_key, best_value = random.choice(evaluated.items())


def criterion_breach_size(solution, criterion):
    x1 = solution[0]
    x2 = solution[1]
    res = eval(criterion)
    if res > 0:
        return res
    return 0


def sum_of_breaches(population, criteria):
    res = {}
    for item in population.keys():
        res[item] = 0
        for criterion in criteria:
            tmp = criterion_breach_size(population[item], criterion)
            res[item] = res[item] + tmp
    return res




def nd_front(population, secondary_population):
    '''
    pop
    :param population: a dictionary where keys are tuples in 2d plane and values are fitness
    :param secondary_population: a list of dictionaries of tuples and corresponding criteria
    :return:
    '''

    F1 = []
    set_values = [[] for key in population.keys()]
    ranks = {}
    np_values = {}
    Set_dict = dict(zip(population.keys(), set_values))

    for member in population.keys():
        np = 0
        for other_member in population.keys():
            if other_member == member:
                continue

            if (population[member] < population[other_member] and secondary_population[member] < secondary_population[
                other_member]):
                tmp = Set_dict[member]
                tmp.append(other_member)
                Set_dict[member] = tmp
            elif (population[member] > population[other_member] and secondary_population[member] > secondary_population[
                other_member]):
                np += 1
        np_values[member] = np
        if np == 0:
            ranks[member] = 1
            F1.append(member)
    i = 0
    Frontiers = []
    Frontiers.append(F1)
    while len(Frontiers[i]) != 0:
        Q = []
        for member in Frontiers[i]:
            for other_member in Set_dict[member]:
                np_values[other_member] = np_values[other_member] - 1
                if np_values[other_member] == 0:
                    ranks[other_member] = i + 2
                    Q.append(other_member)
        i += 1
        Frontiers.append(Q)
    Frontiers.pop()
    #print("Frontiers:", Frontiers)

    return crowding(Frontiers, [population, secondary_population])

def crowding(F, populations):
    max_no = 9999999999999.9
    distances = dict(zip([item for item in populations[0].keys()], [0 for item in populations[0].keys()]))
    for frontier in range(len(F)):
        for objective_function in populations:
            temp_dict = {}
            for key in F[frontier]:   # was: for key in distances.keys():
                temp_dict[key] = objective_function[key]
            tmp_sorted_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1], reverse=False))
            tmp_sorted_pre_adjustment = tmp_sorted_dict.keys()
            tmp_sorted = [ key for key in tmp_sorted_pre_adjustment]
            #print("tmp_sorted:", tmp_sorted)
            if len(F[frontier]) == 1:
                distances[tmp_sorted[0]] = max_no
                break
            distances[tmp_sorted[0]] = max_no
            distances[tmp_sorted[-1]] = max_no
            f_min = objective_function[tmp_sorted[0]]
            f_max = objective_function[tmp_sorted[-1]]
            f_diff = abs(f_max - f_min)+0.000000001
            for member in range(1, len(tmp_sorted_dict.keys()) - 1):
                distances[tmp_sorted[member]] = distances[tmp_sorted[member]] + (objective_function[tmp_sorted[member + 1]] - objective_function[tmp_sorted[member - 1]]) / f_diff
    # I want to return a dictionary, where key is the solution (a, b) and value a list with frontier number on 0th idx and distance on 1st
    ret = dict(zip([item for item in populations[0].keys()], [[] for item in populations[0].keys()]))
    for i in range(len(F)):
        for item in F[i]:
            ret[item] = [i + 1, distances[item]]  # frontier starts off from 1
    #print(dict(sorted(ret.items(), key=lambda item: item[1][0], reverse=False)))
    return dict(sorted(ret.items(), key=lambda item: item[1][0], reverse=False))

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
        x1 = heritability * mother_x + (1 - heritability) * father_x
        x2 = heritability * mother_y + (1 - heritability) * father_y

        if [x1, x2] not in ret:
            ret.append((x1, x2))
            counter += 1
        x1 = heritability * father_x + (1 - heritability) * mother_x
        x2 = heritability * father_x + (1 - heritability) * mother_x

        if [x1, x2] not in ret:
            ret.append((x1, x2))
            counter += 1

    return ret
def mutate(population, x1_mutation_range, x2_mutation_range, first_constraint, second_constraint, function, x1_min,
           x1_max, x2_min, x2_max):
    #print('started process of mutation')
    ret = {}
    counter = 0
    for child in population:
        possible_versions = []
        while len(possible_versions) < 5:  ## we want to create 5 possible versions and pick the best
            x1 = random.uniform(child[0] - x1_mutation_range / 2, child[0] + x1_mutation_range / 2)
            x2 = random.uniform(child[1] - x2_mutation_range / 2, child[1] + x2_mutation_range / 2)
            if x1 < x1_min or x1 > x1_max or x2 < x2_min or x2 > x2_max: #or eval(first_constraint) > 0 or eval(
                    #second_constraint) > 0 or (x1, x2) in ret.keys():
                counter += 1
                #print('missed the mark for the ', counter, 'time')
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
    return list(ret.keys())
    ##return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))


def consolidate(function, first_constraint, second_constraint, mutated, selected_population, population_size):
    '''

    :param function: objective function we're minimising
    :param first_constraint: first constraint function
    :param second_constraint: second constraint function
    :param mutated: a list containing "raw" points representing offspring generation
    :param selected_population: a dictionary where keys are points of the parent generation and value is a list with their front no. and crowding distance
    :param population_size: self-explanatory
    :return: A dictionary with keys as coordinates of the new population and values are a list with front number and crowding distance
    '''
    # 1. first we have to somehow evaluate fronts and crowd. distances for both children and parents together
    for offspring in mutated:
        x1 = offspring[0]
        x2 = offspring[1]
        if offspring not in selected_population.keys():
            r0 = eval(function)
            r1 = 0
            r2 = 0
            if second_constraint:
                if eval(first_constraint)>0:
                    r1 = eval(first_constraint)
                if eval(second_constraint)>0:
                    r2 = eval(second_constraint)
            else:
                r1 = abs(0-eval(first_constraint))
            selected_population[offspring] = [r0, r1+r2]
    first_dict = {}
    second_dict = {}
    for key, value in selected_population.items():
        first_dict[key] = value[0]
        second_dict[key] = value[1]
    return nd_front(first_dict, second_dict)
def validate(item, first_constraint, second_constraint, function):
    x1 = item[0]
    x2 = item[1]
    if eval(first_constraint)>0:
        return 99999999999
    if second_constraint:
        if eval(second_constraint)>0:
            return 99999999999
    return eval(function)

def ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint, second_constraint, initial_population_dict, population_size):
    evaluated_population = initial_population_dict
    best_so_far = None
    best_result = 999999999
    for i in range(generations):
        print('generation:', i, 'out of:', generations)
        selected_population = selection(evaluated_population)
        for item in selected_population:
            if best_so_far is None or validate(item, first_constraint, second_constraint, function) < best_result:
                best_so_far = item
                best_result = validate(item, first_constraint, second_constraint, function)
        new_gen_before = offsprings(selected_population, first_constraint, second_constraint)

        mutated = mutate(new_gen_before, (x_max - x_min) * mutation_range, (y_max - y_min) * mutation_range,
                         first_constraint, second_constraint, function, x_min, x_max, y_min, y_max)
        evaluated_population = consolidate(function, first_constraint, second_constraint, mutated, selected_population, population_size)
        mutation_range = mutation_range * 0.99999
    print("best result is:", best_so_far, "with function value of:", best_result)



if __name__ == '__main__':
    model = 'g06'
    generations = 100
    population_size = 100
    mutation_range = 0.08  # parameter used for uniform mutation employed
    if model == 'g06': # so usa 100 generacoes, pode nao funcionar quando um constraint esta offseting o outro  = depois o erro e mais ou menos zero
        ## g06
        x_min = 13
        x_max = 100
        y_min = 0
        y_max = 100
        function = '(x1-10)**3+(x2-20)**3'
        first_constraint = '-(x1-5)**2-(x2-5)**2+100'
        second_constraint = '(x1-6)**2+(x2-5)**2-82.81'
        first_rewritten = '(100-(x1-5)**2)**.5+5'
        second_rewritten = '(82.81-(x1-6)**2)**2+5'
        first_temp_dict = {}
        second_temp_dict = {}
        first_temp_dict, second_temp_dict = population_generator(population_size, x_min, x_max, y_min, y_max, first_constraint, second_constraint, function)
        temp_dict = nd_front(first_temp_dict, second_temp_dict)

        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
        second_constraint, temp_dict, population_size)

    elif model == 'g08': # nao presta, acho que isso e porque a funcao e assim muito baixa em comparacao com as penalisacoes que nao chegam
        x_min = 0
        x_max = 10
        y_min = 0
        y_max = 10
        function = '-((sin(2 * pi * x1) ** 3) * (sin(2 * pi * x2))) / ((x1 ** 3) * (x1 + x2))'
        first_constraint = 'x1**2-x2+1'
        second_constraint = '1-x1+(x2-4)**2'
        first_temp_dict = {}
        second_temp_dict = {}
        first_temp_dict, second_temp_dict = population_generator(population_size, x_min, x_max, y_min, y_max, first_constraint,
                                                                 second_constraint, function)
        temp_dict = nd_front(first_temp_dict, second_temp_dict)
        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
           second_constraint, temp_dict, population_size)
    elif model == 'g11': #oxala por causa do igual isto nao functiona assim tao bem
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        function = 'x1**2+(x2-1)**2'
        first_constraint = 'x2-x1**2'
        second_constraint = None
        first_temp_dict = {}
        second_temp_dict = {}
        first_temp_dict, second_temp_dict = population_generator(population_size, x_min, x_max, y_min, y_max, first_constraint,
                                                                 second_constraint, function)
        temp_dict = nd_front(first_temp_dict, second_temp_dict)
        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
           second_constraint, temp_dict, population_size)

    elif model == 'g24': # converges well
        x_min = 0
        x_max = 3
        y_min = 0
        y_max = 4
        function = '-x1-x2'
        first_constraint = '-2*x1**4+8*x1**3-8*x1**2-2'
        second_constraint = '-4*x1**4+32*x1**3-88*x1**2+96*x1+x2-36'
        first_temp_dict = {}
        second_temp_dict = {}
        first_temp_dict, second_temp_dict = population_generator(population_size, x_min, x_max, y_min, y_max, first_constraint,
                                                                 second_constraint, function)
        temp_dict = nd_front(first_temp_dict, second_temp_dict)
        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
           second_constraint, temp_dict, population_size)