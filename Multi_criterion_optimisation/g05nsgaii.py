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


def population_generator(population_size, x_min, x_max, y_min, y_max, first_constraint, second_constraint, function,
                         third_constraint, fourth_constraint, fifth_constraint, x3_min, x3_max, x4_min, x4_max):
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

    while len(first_dict.keys()) < population_size:
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(y_min, y_max)
        x3 = random.uniform(x3_min, x3_max)
        x4 = random.uniform(x4_min, x4_max)
        r0 = eval(function)
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        r5 = 0
        # if second constraint is None, we know we're dealing with g11, which is the only constraint g(x) = 0 as opposed to <=0

        if eval(first_constraint) > 0:
            r1 = eval(first_constraint)
        if eval(second_constraint) > 0:
            r2 = eval(second_constraint)
        r3 = abs(eval(third_constraint))
        r4 = abs(eval(fourth_constraint))
        r5 = abs(eval(fifth_constraint))

        if (x1, x2, x3, x4) not in first_dict:
            first_dict[(x1, x2, x3, x4)] = r0
            second_dict[(x1, x2, x3, x4)] = r1 + r2 + r3 + r4 + r5
            counter += 1

    return first_dict, second_dict



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
    return crowding(Frontiers, [population, secondary_population])


def crowding(F, populations):
    max_no = 9999999999999.9
    distances = dict(zip([item for item in populations[0].keys()], [0 for item in populations[0].keys()]))
    for frontier in range(len(F)):
        for objective_function in populations:
            temp_dict = {}
            for key in F[frontier]:  # was: for key in distances.keys():
                temp_dict[key] = objective_function[key]
            tmp_sorted_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1], reverse=False))
            tmp_sorted_pre_adjustment = tmp_sorted_dict.keys()
            tmp_sorted = [key for key in tmp_sorted_pre_adjustment]
            if len(F[frontier]) == 1:
                distances[tmp_sorted[0]] = max_no
                break
            distances[tmp_sorted[0]] = max_no
            distances[tmp_sorted[-1]] = max_no
            f_min = objective_function[tmp_sorted[0]]
            f_max = objective_function[tmp_sorted[-1]]
            f_diff = abs(f_max - f_min) + 0.000000001
            for member in range(1, len(tmp_sorted_dict.keys()) - 1):
                distances[tmp_sorted[member]] = distances[tmp_sorted[member]] + (
                            objective_function[tmp_sorted[member + 1]] - objective_function[
                        tmp_sorted[member - 1]]) / f_diff
    ret = dict(zip([item for item in populations[0].keys()], [[] for item in populations[0].keys()]))
    for i in range(len(F)):
        for item in F[i]:
            ret[item] = [i + 1, distances[item]]  # frontier starts off from 1
    return dict(sorted(ret.items(), key=lambda item: item[1][0], reverse=False))


def offsprings(population):
    ret = []
    counter = 0
    while len(ret) != 100:
        mother = random.choice(list(population.items()))
        mother_x = mother[0][0]
        mother_y = mother[0][1]
        mother_x3 = mother[0][2]
        mother_x4 = mother[0][3]
        father = random.choice(list(population.items()))
        father_x = father[0][0]
        father_y = father[0][1]
        father_x3 = father[0][2]
        father_x4 = father[0][3]

        heritability = random.random()
        x1 = heritability * mother_x + (1 - heritability) * father_x
        x2 = heritability * mother_y + (1 - heritability) * father_y
        x3 = heritability * mother_x3 + (1 - heritability) * father_x4
        x4 = heritability * mother_x4 + (1 - heritability) * father_x3

        if (x1, x2, x3, x4) not in ret:
            ret.append((x1, x2, x3, x4))
            counter += 1
        x1 = heritability * father_x + (1 - heritability) * mother_y
        x2 = heritability * father_y + (1 - heritability) * mother_x
        x3 = heritability * father_x3 + (1 - heritability) * father_x3
        x4 = heritability * father_x4 + (1 - heritability) * father_x4

        if (x1, x2, x3, x4) not in ret:
            ret.append((x1, x2, x3, x4))
            counter += 1

    return ret


def mutate(population, x1_mutation_range, x2_mutation_range,  function, x1_min,
           x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max, x3_mutation_range, x4_mutation_range):
    # print('started process of mutation')
    ret = {}
    counter = 0
    for child in population:
        possible_versions = []
        while len(possible_versions) < 5:  ## we want to create 5 possible versions and pick the best
            if child[0] - x1_mutation_range/2<x1_min:
                bottom_x1 = x1_min
            else:
                bottom_x1 = child[0] - x1_mutation_range/2
            if child[0] + x1_mutation_range / 2 > x1_max:
                top_x1 = x1_max
            else:
                top_x1 = child[0] + x1_mutation_range / 2
            if child[1] - x2_mutation_range /2 < y_min:
                bottom_x2 = y_min
            else:
                bottom_x2 = child[1] - x2_mutation_range / 2
            if child[1] + x2_mutation_range / 2>y_max:
                top_x2 = y_max
            else:
                top_x2 = child[1] + x2_mutation_range / 2
            if (child[2] - x1_mutation_range/2) < x3_min or (child[2] - x1_mutation_range/2)>x3_max:
                bottom_x3 = x3_min
                #print("x3 bottom bound is min:", bottom_x3)
            else:
                bottom_x3 = child[2] - x1_mutation_range / 2
                #print("x3 bottom bound is original:", bottom_x3, "which is bigger than the minimum:", x3_min)
            if child[2] + x3_mutation_range / 2>x3_max or child[2] + x3_mutation_range / 2< x3_min:
                top_x3 = x3_max
                #print("x3 top bound is max:", top_x3)
            else:
                top_x3 = child[2] + x3_mutation_range / 2
                #print("x3 top bound is original:", top_x3)
            if (child[3] - x1_mutation_range/2) < x4_min or (child[3] - x1_mutation_range/2)>x4_max:
                bottom_x4 = x4_min
                #print("x4 bottom bound is min:", bottom_x4)
            else:
                bottom_x4 = child[3] - x1_mutation_range / 2
                #print("x4 bottom bound is original:", bottom_x4)
            if child[3] + x4_mutation_range / 2 > x4_max:
                top_x4 = x4_max
                #print("x4 top bound is max:", top_x4)
            else:
                top_x4 = child[3] + x4_mutation_range / 2
                #print("x4 top bound is original:", top_x4)

            x1 = random.uniform(bottom_x1, top_x1)
            x2 = random.uniform(bottom_x2, top_x2)
            x3 = random.uniform(bottom_x3, top_x3)
            x4 = random.uniform(bottom_x4, top_x4)

            possible_versions.append([x1, x2, x3, x4])
        best = None
        minimum = None
        for version in possible_versions:
            x1 = version[0]
            x2 = version[1]
            x3 = version[2]
            x4 = version[3]
            r1 = eval(function)
            if best is None or minimum is None or r1 < minimum:
                best = [x1, x2, x3, x4]
                minimum = r1
        ret[(best[0], best[1], best[2], best[3])] = minimum
    return list(ret.keys())


def consolidate(function, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint, mutated, selected_population, population_size):
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
        x3 = offspring[2]
        x4 = offspring[3]
        if offspring not in selected_population.keys():
            r0 = eval(function)
            r1 = 0
            r2 = 0
            r3 = abs(eval(third_constraint))
            r4 = abs(eval(fourth_constraint))
            r5 = abs(eval(fifth_constraint))
            if eval(first_constraint) > 0:
                r1 = eval(first_constraint)
            if eval(second_constraint) > 0:
                r2 = eval(second_constraint)
            selected_population[offspring] = [r0, r1 + r2 + r3 + r4 + r5]
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

def ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
       second_constraint, initial_population_dict, population_size, third_constraint, fourth_constraint,
       fifth_constraint, x3_min, x3_max, x4_min, x4_max):
    evaluated_population = initial_population_dict


    for i in range(generations):
        print('generation:', i, 'out of:', generations)
        selected_population = selection(evaluated_population)

        new_gen_before = offsprings(selected_population)
        mutated = mutate(new_gen_before, (x_max - x_min) * mutation_range, (y_max - y_min) * mutation_range,
                         function, x_min, x_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max, (x3_max-x3_min)*mutation_range, (x4_max-x4_min)*mutation_range)
        evaluated_population = consolidate(function, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint, mutated, selected_population,
                                           population_size)
        mutation_range = mutation_range * 0.99999
    result = list(selected_population.keys())
    x1 = result[0][0]
    x2 = result[0][1]
    x3 = result[0][2]
    x4 = result[0][3]
    best_x1 = result[0][0]
    best_x2 = result[0][1]
    best_x3 = x3
    best_x4 = x4
    best = eval(function)

    for possible_result in result:
        x1 = possible_result[0]
        x2 = possible_result[1]
        x3 = possible_result[2]
        x4 = possible_result[3]
        temp_best = eval(function)


        if temp_best < best:
            best = temp_best
            best_x1 = x1
            best_x2 = x2
            best_x3 = x3
            best_x4 = x4
    x1 = best_x1
    x2 = best_x2
    x3 = best_x3
    x4 = best_x4
    print("best result is:", (best_x1, best_x2, best_x3, best_x4), "with the value of:", best, "first constraint:",
          eval(first_constraint), "second:", eval(second_constraint), "third:", eval(third_constraint), "fourth:", eval(fourth_constraint))


if __name__ == '__main__':
    model = 'g05'
    generations = 100
    population_size = 100
    mutation_range = 0.08  # parameter used for uniform mutation employed
    if model == 'g05':  # so usa 100 generacoes, pode nao funcionar quando um constraint esta offseting o outro  = depois o erro e mais ou menos zero
        x_min = 0
        x_max = 1200
        y_min = 0
        y_max = 1200
        x3_min = -0.55
        x3_max = 0.55
        x4_min = -0.55
        x4_max = 0.55

        function = '3*x1+0.000001*x1**3+2*x2+(0.000002/3)*x2**3'
        first_constraint = '-x4+x3-0.55'
        second_constraint = '-x3+x4-0.55'
        third_constraint = '1000*sin(-x3-0.25)+1000*sin(-x4-0.25)+894.8-x1'
        fourth_constraint = '1000*sin(x3-0.25)+1000*sin(x3-x4-0.25)+894.8-x2'
        fifth_constraint = '1000*sin(x4-0.25)+1000*sin(x4-x3-0.25)+1294.8'
        first_temp_dict = {}
        second_temp_dict = {}
        first_temp_dict, second_temp_dict = population_generator(population_size, x_min, x_max, y_min, y_max,
                                                                 first_constraint, second_constraint, function,
                                                                 third_constraint, fourth_constraint, fifth_constraint, x3_min, x3_max, x4_min, x4_max)
        temp_dict = nd_front(first_temp_dict, second_temp_dict)

        ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint,
             second_constraint, temp_dict, population_size, third_constraint, fourth_constraint, fifth_constraint, x3_min, x3_max, x4_min, x4_max)
