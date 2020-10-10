#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas de Búsqueda basadas en Trayectorias
para el Problema de la Máxima Diversidad

@author: Francisco Javier Bolívar Expósito
"""

import numpy as np
import pandas as pd
import os
import time


# @brief Lee la matriz de distancias de un fichero de la biblioteca MDPLIB
# @param path Ruta del fichero
# @return dist Matriz NxM de distancias entre cada par de elementos
# @return m Numero de elementos a seleccionar
def readDistances(path):
    file = open(path, 'r')

    n, m = file.readline().split()

    dist = np.zeros((int(n), int(n)), float)

    for line in file:
        i, j, value = line.split(" ", 3)
        dist[int(i), int(j)] = value
        dist[int(j), int(i)] = value

    file.close()

    return dist, int(m)


# @brief Evalua una solución para MDP siguiendo el modelo MaxSum
# @param dist Matriz NxM de distancias entre cada par de elementos
# @param sel Lista de elementos seleccionados
# @return f Coste obtenido en la función que se quiere maximizar (suma de
#           distancias entre pares de elementos seleccionados)
def calculate_cost(dist, sel):
    indx = sel.reshape(-1, 1) * dist.shape[1] + sel
    dist_sel = dist.take(indx)
    return (dist_sel.sum() - np.einsum('ii', dist_sel))/2


# @brief Resuelve con Busqueda Local Primero el Mejor MDP
# @param dist Matriz NxM de distancias entre cada par de elementos
# @param m Numero de elementos a seleccionar
# @param seed Semilla a utilizar para el generador aleatorio
# @return sel Conjunto de elementos seleccionados
def lsbf(dist, sel, evals):
    # Generamos seleccionables
    s = np.delete(np.arange(len(dist)), sel)

    # Calculo de la contribución de cada elemento
    # (suma de distancias de un elemento al resto de seleccionados)
    z = np.sum(dist[sel[:, None], sel], axis=0)

    # Generamos soluciones vecinas, intercambiando primero los puntos menos
    # prometedores hasta que encontremos una sol mejor cambiando a esta
    # Se repetirá hasta realizar 100.000 evaluaciones de la fObjetivo
    # o hasta que ninguna solución vecina sea mejor que la actual
    evaluaciones = 0
    while evaluaciones < evals:
        mejora = False
        np.random.shuffle(s)

        for ind_cambio in z.argsort():
            orig = sel[ind_cambio]
            for veci in s:
                # Actualizamos el número de evaluaciones
                evaluaciones += 1
                # Genera solución vecina al intercambiar un punto por otro
                sel[ind_cambio] = veci
                # Calcula la contribución del nuevo punto
                cont_veci = dist[veci, sel].sum()

                # Si es mayor actualizamos las contribuciones de seleccionados
                # y colocamos en el conjunto de seleccionables el punto quitado
                if cont_veci > z[ind_cambio]:
                    z = z - dist[sel, orig]
                    z = z + dist[sel, veci]

                    z[ind_cambio] = cont_veci
                    s[s == veci] = orig

                    mejora = True
                    break
                if evaluaciones == evals:
                    break
            if mejora or evaluaciones == evals:
                break
            else:
                sel[ind_cambio] = orig
        if not mejora:
            break

    return sel, z.sum()/2


def swap(sol, s, i, j):
    sol_veci = sol.copy()

    sol_veci[i] = s[j]

    return sol_veci


def sa(dist, sel, max_evals=100000, u=0.3, t_final=10**-3, max_veci_prop=10, max_exitos_prop=0.1, k=1):
    best_sol = sel.copy()

    # Calculo de la temperatura inicial
    t_inicial = (u * calculate_cost(dist, sel)) / -np.log(u)
    t = t_inicial

    # Guardamos seleccionables (no seleccionados)
    s = np.delete(np.arange(len(dist)), sel)

    # Calculo de la contribución de cada elemento a la solución
    # (suma de distancias de cada elemento al resto de seleccionados)
    z = np.sum(dist[sel[:, None], sel], axis=0)
    best_cost = z.sum()

    # Parámetros del algoritmo
    max_veci = int((max_veci_prop * len(dist)) / (max_evals/100000))
    max_exitos = max_exitos_prop * max_veci
    beta = (t_inicial - t_final) / ((max_evals / max_veci) * t_inicial * t_final)
    n_evals = 0

    while t > t_final:
        n_veci = 0
        n_exitos = 0

        random_i = np.random.randint(0, len(sel), max_veci)
        random_j = np.random.randint(0, len(s), max_veci)

        while n_veci < max_veci and n_exitos < max_exitos and n_evals < max_evals:
            i = random_i[n_veci]
            j = random_j[n_veci]

            sol_veci = swap(sel, s, i, j)
            cont_veci = dist[s[j], sol_veci].sum()
            n_veci += 1
            n_evals += 1

            diff_cont = z[i] - cont_veci

            if diff_cont < 0 or np.random.random() <= np.exp(-diff_cont / k * t):
                z = z - dist[sol_veci, sel[i]]
                z = z + dist[sol_veci, s[j]]
                z[i] = cont_veci

                s[j] = sel[i]
                sel = sol_veci

                cost = z.sum()

                n_exitos += 1

                if cost > best_cost:
                    best_sol = sel.copy()
                    best_cost = cost

        t = t / (1 + beta * t)

        if n_exitos == 0:
            break

    return best_sol, best_cost / 2


def bmb(dist, m, n_sol, max_evals_bl):
    best_cost = 0
    best_sol = np.empty(m)

    for i in range(n_sol):
        sol_inicial = np.random.choice(len(dist), m, replace=False)

        sol, cost = lsbf(dist, sol_inicial, max_evals_bl)

        if cost > best_cost:
            best_sol = sol
            best_cost = cost

    return best_sol, best_cost


def ils(dist, m, n_sol, max_evals_ref, refine=lsbf):
    sol_inicial = np.random.choice(len(dist), m, replace=False)
    sol, cost = refine(dist, sol_inicial, max_evals_ref)

    best_sol = sol.copy()
    best_cost = cost

    for i in range(n_sol - 1):
        # Guardamos seleccionables (no seleccionados)
        s = np.delete(np.arange(len(dist)), sol)

        i = np.random.choice(m, int(0.1 * m), replace=False)
        j = np.random.choice(len(s), int(0.1 * m), replace=False)

        sol = swap(sol, s, i, j)
        sol, cost = refine(dist, sol, max_evals_ref)

        if cost > best_cost:
            best_sol = sol.copy()
            best_cost = cost
        else:
            sol = best_sol.copy()

    return best_sol, best_cost


list_directory = os.listdir('../BIN/')
list_directory.sort()

data_sa = []
data_bmb = []
data_ils_ls = []
data_ils_sa = []

# Itera por todos los casos de ejecución
for filename in list_directory:
    dist, m = readDistances('../BIN/'+filename)

    caso = filename.rstrip('.txt')

    np.random.seed(510)
    start = time.time()
    sel = np.random.choice(len(dist), m, replace=False)
    sel, f = sa(dist, sel)
    end = time.time()
    data_sa.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

    np.random.seed(510)
    start = time.time()
    sel, f = bmb(dist, m, 10, 10000)
    end = time.time()
    data_bmb.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

    np.random.seed(510)
    start = time.time()
    sel, f = ils(dist, m, 10, 10000)
    end = time.time()
    data_ils_ls.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

    np.random.seed(510)
    start = time.time()
    sel, f = ils(dist, m, 10, 10000, sa)
    end = time.time()
    data_ils_sa.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

df = pd.concat([pd.DataFrame(data_sa), pd.DataFrame(data_bmb), 
                pd.DataFrame(data_ils_ls), pd.DataFrame(data_ils_sa)], axis=1)
df.to_csv(r'../resultados2.csv', index=False)