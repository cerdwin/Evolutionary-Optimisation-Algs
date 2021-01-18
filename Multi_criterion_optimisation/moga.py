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


def evaluate(population, function, first_constraint, second_constraint):
    result = []
    first = {}
    second = {}
    third = {}
    for solution in population:
        pi = math.pi
        x1 = solution[0]
        x2 = solution[1]
        first[solution] = eval(function)
        second[solution] = eval(first_constraint)
        third[solution] = eval(second_constraint)
    result.append(first)
    result.append(second)
    result.append(third)

    return result#dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))

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
        return True
    if feasible_first and (first_breach>0 or second_breach>0):
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
    return True

def selection(population, first_constraint, second_constraint):

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
                #counter+=1
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
    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))

def validate(item, first_constraint, second_constraint, function):
    x1 = item[0]
    x2 = item[1]
    if eval(first_constraint)>0:
        return 99999999999

    if eval(second_constraint)>0:
        return 99999999999
    return eval(function)


def moga(populations, function):
    # populations is a list of dictionaries
    # 1. first we give each item a count of members they are dominated by
    # we create a dictionary composed of coordinates and values is a list of corresponding values
    alpha = 1  # could be 2
    coordinates = list(populations[0].keys())
    modified = {}
    dominated = {}
    frontiers = {}
    for i in coordinates:
        modified[i] = []
        dominated[i] = 1
        frontiers[i] = 0  # groups of values which are dominated by the same number of other values

    sigma_shares = {}
    for dict in populations:
        for key, value in dict.items():
            tmp = modified[key]
            tmp.append(value)
            modified[key] = tmp

    possible_frontiers = [i + 1 for i in range(len(populations[0].keys()))]

    for item in modified.keys():
        for other_item in modified.keys():
            if item == other_item:
                continue
            dominant = False
            for i in range(len(modified[item])):
                if modified[item][i] > modified[other_item][i]:  # reveals whether first item is dominated
                    dominant = False
                    break
                if modified[item][i] < modified[other_item][i]:
                    if i not in sigma_shares.keys():
                        sigma_shares[i] = [1, abs(modified[item][i] - modified[other_item][i])]
                    else:
                        sigma_shares[i] = [sigma_shares[i][0] + 1,
                                           sigma_shares[i][1] + abs(modified[item][i] - modified[other_item][i])]

                    dominant = True
            if dominant:
                dominated[other_item] = dominated[other_item] + 1 # gives us by how many were other_items dominated
    ret = {}
    for key,value in dominated.items():
        x1 = key[0]
        x2 = key[1]
        ret[key] = 2*value+eval(function)
    return ret




def niche_moga(populations, function):
    # populations is a list of dictionaries
    # 1. first we give each item a count of members they are dominated by
    # we create a dictionary composed of coordinates and values is a list of corresponding values
    alpha = 1  # could be 2
    print("populations[0]:", populations[0])
    coordinates = list(populations[0].keys())
    print("coordinates:", coordinates)
    modified = {}
    dominated = {}
    frontiers = {}
    for i in coordinates:
        modified[i] = []
        dominated[i] = 1
        frontiers[i] = 0 # groups of values which are dominated by the same number of other values



    sigma_shares = {}
    for dict in populations:
        for key, value in dict.items():
            tmp = modified[key]
            tmp.append(value)
            modified[key] = tmp

    possible_frontiers = [i + 1 for i in range(len(populations[0].keys()))]

    for item in modified.keys():
        for other_item in modified.keys():
            if item == other_item:
                continue
            dominant = False
            for i in range(len(modified[item])):
                if modified[item][i] > modified[other_item][i]:  # reveals whether first item is dominated
                    dominant = False
                    break
                if modified[item][i] < modified[other_item][i]:
                    if i not in sigma_shares.keys():
                        sigma_shares[i] = [1, abs(modified[item][i] - modified[other_item][i])]
                    else:
                        sigma_shares[i] = [sigma_shares[i][0] + 1,
                                           sigma_shares[i][1] + abs(modified[item][i] - modified[other_item][i])]

                    dominant = True
            if dominant:
                dominated[other_item] = dominated[other_item] + 1

    frontier_members = {}
    for i in possible_frontiers:
        frontier_members[i] = []
    for key in dominated.keys():
        tmp = frontier_members[dominated[key]]
        tmp.append(key)
        frontier_members[dominated[key]] = tmp
    print("frontier members:", frontier_members)
    # 2. Next, we assign raw fitness to each solution by using a linear mapping function
    # 3. Now we count members of each rank
    rank_sizes = {}
    for key, value in dominated:
        if value not in rank_sizes.keys():
            rank_sizes[value] = 1
        rank_sizes[value] = rank_sizes[value] + 1
    # 4. identify maximum rank
    max_rank = max(rank_sizes.keys())
    N = len(populations[0].keys())  # count of all members of the population
    shared_fitness = {}
    for k, v in frontiers.items():
        shared_fitness[k] = 0  # dictionary, where each member of a fitness frontier shares
    # POZOR, musime kontrolovat v loopu, zda ma kazda frontier aspon jeden prvek
    for key, value in shared_fitness:
        print("key:", key, "value:", value)
        if len(frontier_members[i]) > 0:
            tmp = 0
            for i in range(1, frontiers[(key, value)]):
                tmp += len(frontier_members[i])
            shared_fitness[(key, value)] = N - tmp - 0.5 * (len(frontier_members[(key, value)]) - 1)
    print("shared fitness:", shared_fitness)
    # now we work out the niche count for every value
    niches = {}
    for frontier in frontier_members.keys():
        if not frontier_members[frontier]:
            continue
        for a in frontier_members[frontier]:
            tmp_niche = 0
            for b in frontier_members[frontier]:
                if a == b:
                    continue
                niche_value = 0
                sh_ab = None
                for i in range(len(populations)):
                    a_i = populations[i][a]
                    b_i = populations[i][b]
                    i_max = max(populations[i].values())
                    i_min = min(populations[i].values())
                    niche_value += ((a_i - b_i) / (i_max - i_min)) ** 2
                    print("sigma shares", sigma_shares[i], "other:", (((a_i - b_i) / (i_max - i_min)) ** 2) ** 0.5)
                    #if (((a_i - b_i) / (i_max - i_min)) ** 2) ** 0.5 > sigma_shares[i][0]:
                    #    sh_ab = 0
                    #    break
                if sh_ab is None:
                    d_ab = niche_value ** 0.5
                    sh_ab = 1 - (d_ab / sum(sigma_shares[i]) ) ** alpha
                    tmp_niche += sh_ab
            niches[a] = tmp_niche
    fitness = {}
    for key, value in frontier_members.items():
        Fi = shared_fitness[key]
        for member in value:
            fitness[member] = Fi/niches[member]
    print("fitness:", fitness)
    return fitness



def ga(model, generations, mutation_range, x_min, x_max, y_min, y_max, function, first_constraint, second_constraint):
    initial_population = population_generator(100, x_min, x_max, y_min, y_max)
    evaluated_population = evaluate(initial_population, function, first_constraint, second_constraint)
    print("evaluated pop:", evaluated_population)
    evaluated_population = moga(evaluated_population, function)
    #print(meg)

    best_so_far = None
    best_result = 999999999
    for i in range(generations):
        print("generation", i , "out of ", generations, "generations")
        selected_population = selection(evaluated_population, first_constraint, second_constraint)
        #### keeping a record of the best we've reached so far
        for item in selected_population:
            if best_so_far is None or validate(item, first_constraint, second_constraint, function)<best_result:
                best_so_far = item
                best_result = validate(item, first_constraint, second_constraint, function)
        ###
        new_gen_before = offsprings(selected_population, first_constraint, second_constraint)
        mutated = mutate(new_gen_before, (x_max - x_min) * mutation_range, (y_max - y_min) * mutation_range,
                         first_constraint, second_constraint, function, x_min, x_max, y_min, y_max)
        #print('mutated:', mutated)
        evaluated_population = moga(evaluate(list(mutated.keys()), function, first_constraint, second_constraint), function)
        mutation_range = mutation_range * 0.99


    print("best result is:", best_so_far, "with function value of:", best_result)


if __name__ == '__main__':
    model = 'g06'
    generations = 50
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
        # get population and corresponding constraint values as dicts in a list


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






