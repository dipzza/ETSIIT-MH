#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:06:22 2020

@author: dipzza
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
        i, j, value = line.split()
        dist[int(i), int(j)] = value
        dist[int(j), int(i)] = value

    file.close()

    return dist, int(m)


# @brief Evalua una solución para MDP siguiendo el modelo MaxSum
# @param dist Matriz NxM de distancias entre cada par de elementos
# @param sel Lista de elementos seleccionados
# @return f Coste obtenido en la función que se quiere maximizar (suma de
#           distancias entre pares de elementos seleccionados)
def fObjetivo(dist, sel):
    return np.triu(dist[sel][:, sel], 1).sum()


# @brief Resuelve de forma greedy el MDP variante MaxSum
# @param dist Matriz NxM de distancias entre cada par de elementos
# @param m Numero de elementos a seleccionar
# @return sel Conjunto de elementos seleccionados
def greedyMDP(dist, m):
    sel = set()
    distAc = np.sum(dist, axis=1)
    ult_sel = distAc.argmax()
    sel.add(ult_sel)

    distAc[:] = 0

    while (len(sel) < m):
        distAc = np.sum([distAc, dist[:, ult_sel]], axis=0)

        for i in sel:
            distAc[i] = 0

        ult_sel = distAc.argmax()
        distAc[ult_sel] = 0
        sel.add(ult_sel)

    sel = np.fromiter(sel, int, len(sel))

    return sel


# @brief Resuelve con Busqueda Local Primero el Mejor el MDP variante MaxSum
# @param dist Matriz NxM de distancias entre cada par de elementos
# @param m Numero de elementos a seleccionar
# @param seed Semilla a utilizar para el generador aleatorio
# @return sel Conjunto de elementos seleccionados
def lsbf(sel, dist, m, evals):
    # Generamos seleccionables
    s = np.delete(np.arange(len(dist)), sel)

    # Calculo de la contribución de cada elemento
    # (suma de distancias de un elemento al resto de seleccionados)
    z = np.sum(dist[sel[:, None], sel], axis=0)

    # Generamos soluciones vecinas, intercambiando primero los puntos menos
    # prometedores hasta que encontremos una sol mejor cambiando a esta
    # Se repetirá hasta realizar 100.000 evaluaciones de la función objetivo
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
                sel[ind_cambio] = orig
                if evaluaciones == evals:
                    break
            if mejora or evaluaciones == evals:
                break
        if not mejora:
            break

    return sel, z.sum() / 2


list_directory = os.listdir('../BIN/')
list_directory.sort()

data_greedy = []
data_lsbf = []
data_gryls = []

# Itera por todos los casos de ejecución, usando greedy y búsqueda local
for filename in list_directory:
    dist, m = readDistances('../BIN/'+filename)
    
    caso = filename.rstrip('.txt')

    np.random.seed(510)
    start = time.time()
    sel_greedy = greedyMDP(dist, m)
    end = time.time()
    f = fObjetivo(dist, sel_greedy)
    data_greedy.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

    np.random.seed(510)
    start = time.time()
    sel = np.random.choice(len(dist), m, replace=False)
    sel, f = lsbf(sel, dist, m, 100000)
    end = time.time()
    data_lsbf.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

    np.random.seed(510)
    start = time.time()
    sel, f = lsbf(sel_greedy, dist, m, 100000)
    end = time.time()
    data_gryls.append({'Caso': caso, 'Coste Obtenido': f, 'Tiempo': (end-start)})

df = pd.concat([pd.DataFrame(data_greedy), pd.DataFrame(data_lsbf),
                pd.DataFrame(data_gryls)], axis=1)
df.to_csv(r'../resultados.csv', index=False)
