#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmo Poblacional Cooperativo para el MDP

@author: Francisco Javier Bolívar Expósito
"""

import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid


# @brief Lee un fichero de la biblioteca MDPLIB
# @param path Ruta del fichero
# @return dist Matriz NxM de distancias entre cada par de elementos
# @return m Numero de elementos a seleccionar
def read_mdplib(path):
    file = open(path, 'r')

    n, m = file.readline().split()

    dist = np.zeros((int(n), int(n)), float)

    for line in file:
        i, j, value = line.split(" ", 3)
        dist[int(i), int(j)] = value
        dist[int(j), int(i)] = value

    file.close()

    return dist, int(m)


# @brief Resuelve con Busqueda Local Primero el Mejor MDP
# @param dist Matriz NxM de distancias entre cada par de elementos
# @param m Numero de elementos a seleccionar
# @param seed Semilla a utilizar para el generador aleatorio
# @return sel Conjunto de elementos seleccionados
def lsbf(dist, sol, fitness, max_evals, stop_th, rng=np.random.default_rng()):
    # Generamos conjunto de no seleccionados
    s = np.delete(np.arange(len(dist)), sol)

    evals = 0
    while True:
        evals_no_change = 0
        mejora = False

        rng.shuffle(s)
        for ind_cambio in fitness.argsort():
            orig = sol[ind_cambio]
            for veci in s:
                # Genera solución vecina intercambiando un elem seleccionado
                sol[ind_cambio] = veci
                # Calcula la contribución del nuevo elemento
                veci_fit = dist[veci][sol]
                cont_veci = veci_fit.sum()
                # Actualizamos el número de evaluaciones
                evals += 1
                evals_no_change += 1

                # Si sol vecina mejor cambiamos a esta
                if cont_veci > fitness[ind_cambio]:
                    fitness -= dist[orig][sol]
                    fitness += veci_fit
                    fitness[ind_cambio] = cont_veci

                    s[s == veci] = orig
                    mejora = True
                    break

                sol[ind_cambio] = orig

                if evals == max_evals or evals_no_change == stop_th:
                    return evals
            if mejora:
                break
        if not mejora:
            break

    return evals


def swap_best_worst(dist, population, fitness, p=0.1):
    evals = 0

    sorted_fit = np.argsort(fitness, axis=None)
    best = list(sorted_fit[:-int(len(sorted_fit) * p):-1])
    worst = list(sorted_fit[:int(len(sorted_fit) * p)])

    row_w = []
    set_w = {}
    for i in range(len(worst)):
        row_w.append(int(worst[i] / population.shape[1]))

        if set_w.get(row_w[i]) is None:
            set_w[row_w[i]] = set(population[row_w[i]])

    for i in range(len(best)):
        for j in range(len(worst)):
            if not population.flat[best[i]] in set_w[row_w[j]]:
                orig = population.flat[worst[j]]
                # Genera solución vecina al intercambiar un punto por otro
                population.flat[worst[j]] = population.flat[best[i]]
                # Calcula la contribución del nuevo punto
                new_fit = dist[population.flat[best[i]]][population[row_w[j]]]
                new_cont = new_fit.sum()

                evals += 1

                if new_cont > fitness.flat[worst[j]]:
                    fitness[row_w[j]] -= dist[orig][population[row_w[j]]]
                    fitness[row_w[j]] += new_fit
                    fitness.flat[worst[j]] = new_cont

                    set_w[row_w[j]].add(population.flat[best[i]])
                    row_w.pop(j)
                    worst.pop(j)
                    break

                population.flat[worst[j]] = orig

    return evals


def rand_population(dist, m, n_people, rng=np.random.default_rng()):
    if n_people == 1:
        population = rng.choice(len(dist), m, replace=False, shuffle=False)
        fitness = np.sum(dist[population[:, None], population], axis=0)
    else:
        population = np.empty((n_people, m), dtype=np.uint16)
        fitness = np.empty((n_people, m))

        for i in range(n_people):
            population[i] = rng.choice(len(dist), (m), replace=False, shuffle=False)

        for i in range(len(population)):
            fitness[i] = np.sum(dist[population[i][:, None], population[i]], axis=0)

    return population, fitness


def mutate_solution(sol, n, p=0.1, rng=np.random.default_rng()):
    s = np.delete(np.arange(n), sol)

    i = rng.choice(len(sol), int(p * len(sol)), replace=False, shuffle=False)
    j = rng.choice(len(s), int(p * len(sol)), replace=False, shuffle=False)

    sol[i] = s[j]


def acr(dist, m, n_people, p_mut_pop, p_swap, p_ls, ls_th, max_evals, rng):
    # Inicializar población
    population, fitness = rand_population(dist, m, n_people, rng)
    evals = n_people

    best_sol = np.empty((m), dtype=np.uint16)
    best_cost = 0

    while evals < max_evals:
        evals += swap_best_worst(dist, population, fitness, p_swap)

        for i in fitness.sum(axis=1).argsort()[:-(int(n_people * p_ls) + 1):-1]:
            lim = max_evals - evals

            evals += lsbf(dist, population[i], fitness[i], lim, ls_th, rng)

        fitness_sum = fitness.sum(axis=1)
        sort_fit = np.argsort(fitness_sum)[::-1]

        if fitness_sum[sort_fit[0]] > best_cost:
            best_sol = population[sort_fit[0]].copy()
            best_cost = fitness[sort_fit[0]].sum()

        n_res = int(n_people * p_mut_pop)
        population[sort_fit[:n_res]], fitness[sort_fit[:n_res]] = rand_population(dist, m, n_res, rng)
        evals += n_res

    return best_sol, best_cost / 2


def acm(dist, m, n_people, p_mut_pop, p_swap, p_ls, p_mut_sol, ls_th, max_evals, rng):
    population, fitness = rand_population(dist, m, n_people, rng)
    evals = n_people

    best_sol = np.empty((m), dtype=np.uint16)
    best_cost = 0

    while evals < max_evals:
        evals += swap_best_worst(dist, population, fitness, p_swap)

        for i in fitness.sum(axis=1).argsort()[:-(int(n_people * p_ls) + 1):-1]:
            lim = max_evals - evals

            evals += lsbf(dist, population[i], fitness[i], lim, ls_th, rng)

        fitness_sum = fitness.sum(axis=1)
        sort_fit = np.argsort(fitness_sum)[::-1]

        if fitness_sum[sort_fit[0]] > best_cost:
            best_sol = population[sort_fit[0]].copy()
            best_cost = fitness[sort_fit[0]].sum()

        for i in sort_fit[:int(n_people * p_mut_pop)]:
            mutate_solution(population[i], len(dist), p_mut_sol, rng)
            fitness[i] = np.sum(dist[population[i][:, None], population[i]], axis=0)
        evals += int(n_people * p_mut_pop)

    return best_sol, best_cost / 2


DIRLIST = os.listdir('../BIN/')
DIRLIST.sort()

# Results for best parameters
data_acm = []
data_acr = []
for i in range(len(DIRLIST)):

    dist, m = read_mdplib('../BIN/' + DIRLIST[i])

    caso = DIRLIST[i].rstrip('.txt')

    rng = np.random.Generator(np.random.SFC64(510))
    start = time.time()
    sel, f = acm(dist, m, 5, 0.8, 0.1, 0.2, 0.3, 1000, 100000, rng)
    end = time.time()
    data_acm.append({'Caso': caso, 'Coste Obtenido': f,
                     'Tiempo': (end - start)})

    rng = np.random.Generator(np.random.SFC64(510))
    start = time.time()
    sel, f = acr(dist, m, 5, 1, 0.3, 0.4, 1000, 100000, rng)
    end = time.time()
    data_acr.append({'Caso': caso, 'Coste Obtenido': f,
                     'Tiempo': (end - start)})

df = pd.concat([pd.DataFrame(data_acm), pd.DataFrame(data_acr)], axis=1)
df.to_csv(r'../results.csv', index=False)


# Parameter study

# BEST_COST = [19587.12891, 19360.23633, 19366.69922, 19458.56641, 19422.15039,
#              19680.20898, 19331.38867, 19461.39453, 19477.32813, 19604.84375,
#              114139, 114092, 114124, 114203, 114180, 114252, 114213, 114378,
#              114201, 114191, 774961.3125, 778030.625, 779963.6875, 776768.4375,
#              775394.625, 775611.0625, 775153.6875, 777232.875, 779168.75,
#              774802.1875]

# ACM_PARAMS = ParameterGrid([{'n_people': [5, 25],
#                              'p_mut_pop': [0.4, 0.8, 1],
#                              'p_swap': [0.1, 0.3],
#                              'p_ls': [0.2, 0.4],
#                              'p_mut_sol': [0.3, 0.6],
#                              'ls_th': [1000, 10000]}])

# ACR_PARAMS = ParameterGrid([{'n_people': [5, 25],
#                              'p_mut_pop': [0.4, 0.8, 1],
#                              'p_swap': [0.1, 0.3],
#                              'p_ls': [0.2, 0.4],
#                              'ls_th': [1000, 10000]}])
# data_desv_acm = []
# data_desv_acr = []
# desv_total_acm = np.zeros(len(ACM_PARAMS))
# desv_total_acr = np.zeros(len(ACR_PARAMS))
# Itera por todos los casos de ejecución
# for i in range(len(DIRLIST)):

#     dist, m = read_mdplib('../BIN/' + DIRLIST[i])

#     caso = DIRLIST[i].rstrip('.txt')

#     for j in range(len(ACM_PARAMS)):
#         rng = np.random.Generator(np.random.SFC64(510))
#         __, f = acm(dist, m, rng=rng, max_evals=100000, **ACM_PARAMS[j])
#         desv_total_acm[j] += (BEST_COST[i] - f) / BEST_COST[i]

#     for j in range(len(ACR_PARAMS)):
#         rng = np.random.Generator(np.random.SFC64(510))
#         __, f = acr(dist, m, rng=rng, max_evals=100000, **ACR_PARAMS[j])
#         desv_total_acr[j] += (BEST_COST[i] - f) / BEST_COST[i]

# for i in range(len(ACM_PARAMS)):
#     dic = ACM_PARAMS[i]
#     dic['desv. media'] = 100 * (desv_total_acm[i] / len(DIRLIST))
#     data_desv_acm.append(dic)

# for i in range(len(ACR_PARAMS)):
#     dic = ACR_PARAMS[i]
#     dic['desv. media'] = 100 * (desv_total_acr[i] / len(DIRLIST))
#     data_desv_acr.append(dic)

# df = pd.DataFrame(data_desv_acm).sort_values(by='desv. media')
# df.to_csv(r'../parameter_study_acm.csv', index=False)

# df = pd.DataFrame(data_desv_acr).sort_values(by='desv. media')
# df.to_csv(r'../parameter_study_acr.csv', index=False)
