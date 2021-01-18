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

def population_generator(pop_size,x_min, x_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
    ret = []
    counter = 0
    pi = math.pi
    while len(ret)<pop_size:
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(y_min, y_max)
        x3 = random.uniform(x3_min, x3_max)
        x4 = random.uniform(x4_min, x4_max)
        ret.append((x1, x2, x3, x4))
    return ret

######################################
def evaluate(population, function, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
    ret = {}
    for solution in population:
        pi = math.pi
        x1 = solution[0]
        x2 = solution[1]
        ret[solution] = eval(function)

    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))

def to_swap(first, second, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
    x1 = first[0]
    x2 = first[1]
    x3 = first[2]
    x4 = first[3]
    first_breach = eval(first_constraint)
    second_breach = eval(second_constraint)
    third_breach = abs(eval(third_constraint))
    fourth_breach = abs(eval(fourth_constraint))
    fifth_breach = abs(eval(fifth_constraint))
    tmp = third_breach+fourth_breach+fifth_breach
    if first_breach>0:
        tmp+=first_breach
    if second_breach>0:
        tmp+=second_breach

    x1 = second[0]
    x2 = second[1]
    x3 = second[2]
    x4 = second[3]
    first_breach = eval(first_constraint)
    second_breach = eval(second_constraint)
    third_breach = abs(eval(third_constraint))
    fourth_breach = abs(eval(fourth_constraint))
    fifth_breach = abs(eval(fifth_constraint))
    tmp2 = third_breach+fourth_breach+fifth_breach
    if first_breach>0:
        tmp2+=first_breach
    if second_breach>0:
        tmp2+=second_breach
    if tmp2 < tmp:
        return True
    return False


def selection(population, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
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
            if die < 0.46: # comparing based off on the objective function alone
                if population[pre_sorted[j]]>population[pre_sorted[j+1]]:
                    tmp = pre_sorted[j]
                    pre_sorted[j] = pre_sorted[j + 1]
                    pre_sorted[j + 1] = tmp
            else:
                if to_swap(pre_sorted[j], pre_sorted[j+1], first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
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


def offsprings(population, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
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
        x1 = heritability*mother_x+(1-heritability)*father_x
        heritability = random.random()
        x2 = heritability*mother_y+(1-heritability)*father_y
        heritability = random.random()
        x3 = heritability*mother_x3+(1-heritability)*father_x3
        heritability = random.random()
        x4 = heritability*mother_x4+(1-heritability)*father_x4
        ret.append((x1, x2, x3, x4))

        heritability = random.random()
        x1 = heritability * father_x + (1 - heritability) * mother_x
        heritability = random.random()
        x2 = heritability * father_x + (1 - heritability) * mother_x
        heritability = random.random()
        x3 = heritability * father_x3 + (1 - heritability) * mother_x3
        heritability = random.random()
        x4 =  heritability * father_x4 + (1 - heritability) * mother_x4

        ret.append((x1, x2, x3, x4))

    return ret


def mutate(population, x1_mutation_range, x2_mutation_range, x3_mutation_range, x4_mutation_range, function, x1_min, x1_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max):
    #print('started process of mutation')
    ret = {}
    counter = 0
    for child in population:
        possible_versions = []
        while len(possible_versions)<5: ## we want to create 5 possible versions and pick the best

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
    return dict(sorted(ret.items(), key=lambda item: item[1], reverse=False))


def ga(generations, mutation_range, x_min, x_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max, function, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint):
    initial_population = population_generator(100, x_min, x_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint)
    evaluated_population = evaluate(initial_population, function, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint)
    print("evaluated pop:", evaluated_population)
    for i in range(generations):
        print("generation", i , "out of ", generations, "generations")
        selected_population = selection(evaluated_population, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint)

        new_gen_before = offsprings(selected_population, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint)
        mutated = mutate(new_gen_before, (x_max - x_min) * mutation_range, (y_max - y_min) * mutation_range, (x3_max-x3_min)*mutation_range, (x4_max-x4_min)*mutation_range,
                          function, x_min, x_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max)
        #print('mutated:', mutated)
        evaluated_population = mutated
        mutation_range = mutation_range * 0.99
    best = min(selected_population, key=selected_population.get)
    x1 = best[0]
    x2 = best[1]
    best_res = eval(function)
    for item in selected_population:
        x1 = item[0]
        x2 = item[1]
        tmp0 = eval(function)
        if best_res>tmp0:
            best = item
            best_res = tmp0
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
          eval(first_constraint), "second:", eval(second_constraint), "third:", eval(third_constraint), "fourth:",
          eval(fourth_constraint))



if __name__ == '__main__':
    model = 'g05'
    generations = 20
    mutation_range = 0.05  # parameter used for uniform mutation employed
    if model == 'g05':
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
        ga(generations, mutation_range, x_min, x_max, y_min, y_max, x3_min, x3_max, x4_min, x4_max, function, first_constraint, second_constraint, third_constraint, fourth_constraint, fifth_constraint)

