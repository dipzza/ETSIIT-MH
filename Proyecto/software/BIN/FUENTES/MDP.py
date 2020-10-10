#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas de Búsqueda basadas en Poblaciones
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


class AG_MDP(object):
    def __init__(self, n_chromosome=50, dist=None, m=None):
        self._n_evals = 0
        self.n_chromosome = n_chromosome
        self._fitness = np.empty((self.n_chromosome))

        self._population = None
        self._sons_population = None

        self.loadCase(dist, m)

    def loadCase(self, dist, m):
        self.dist = dist
        self.m = m

    # @brief Evalua una solución para MDP siguiendo el modelo MaxSum
    # @param dist Matriz NxM de distancias entre cada par de elementos
    # @param sel Lista de elementos seleccionados
    # @return f Coste obtenido en la función que se quiere maximizar (suma de
    #           distancias entre pares de elementos seleccionados)
    def fObjetivo(self, dist, sel):
        indx = sel.reshape(-1, 1) * dist.shape[1] + sel
        dist_sel = dist.take(indx)
        return (dist_sel.sum() - np.einsum('ii', dist_sel))/2

    # @brief Resuelve con Busqueda Local Primero el Mejor MDP
    # @param dist Matriz NxM de distancias entre cada par de elementos
    # @param m Numero de elementos a seleccionar
    # @param seed Semilla a utilizar para el generador aleatorio
    # @return sel Conjunto de elementos seleccionados
    def lsbf(self, sel, evals):
        # Generamos seleccionables
        s = np.delete(np.arange(len(self.dist)), sel)

        # Calculo de la contribución de cada elemento
        # (suma de distancias de un elemento al resto de seleccionados)
        z = np.sum(self.dist[sel[:, None], sel], axis=0)

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
                    cont_veci = self.dist[veci, sel].sum()

                    # Si es mayor actualizamos las contribuciones de los puntos seleccionados
                    # y colocamos en el conjunto de seleccionables el punto quitado
                    if cont_veci > z[ind_cambio]:
                        z = z - self.dist[sel, orig]
                        z = z + self.dist[sel, veci]
    
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
            
        self._n_evals += evaluaciones
        
        return sel

    # Algoritmo genético generacional con operador de cruce uniforme
    def agg_uniform(self, evals_max):
        self._initializePopulation()

        self._n_evals = len(self._population)
        for i in range(len(self._population)):
            self._fitness[i] = self.fObjetivo(self.dist, np.where(self._population[i] == 1)[0])

        while(self._n_evals < evals_max):
            self._selectBinaryTourney(n_to_select=self.n_chromosome)
            self.recombinatePopulation(p_cruce=0.7, o_cruce='u')
            self.mutate()
            self.replaceGeneration()

        return self._population[np.argmax(self._fitness)], np.max(self._fitness)
   
    # Algoritmo genético generacional con operador de cruce posición
    def agg_position(self, evals_max):
        self._initializePopulation()

        self._n_evals = len(self._population)
        for i in range(len(self._population)):
            self._fitness[i] = self.fObjetivo(self.dist, np.where(self._population[i] == 1)[0])

        while(self._n_evals < evals_max):
            self._selectBinaryTourney(n_to_select=self.n_chromosome)
            self.recombinatePopulation(p_cruce=0.7, o_cruce='p')
            self.mutate()
            self.replaceGeneration()

        return self._population[np.argmax(self._fitness)], np.max(self._fitness)

    # Algoritmo estacionario con operador de cruce uniforme
    def age_uniform(self, evals_max):
        self._initializePopulation(2)
        
        self._n_evals = len(self._population)
        for i in range(len(self._population)):
            self._fitness[i] = self.fObjetivo(self.dist, np.where(self._population[i] == 1)[0])
        
        while(self._n_evals < evals_max):
            self._selectBinaryTourney(n_to_select=2)
            self.recombinatePopulation(p_cruce=1, o_cruce='u')
            self.mutate()
            self.replaceStationary()
        
        return self._population[np.argmax(self._fitness)], np.max(self._fitness)
    
    # Algoritmo estacionario con operador de cruce posicional
    def age_position(self, evals_max):
        self._initializePopulation(2)
        
        self._n_evals = len(self._population)
        for i in range(len(self._population)):
            self._fitness[i] = self.fObjetivo(self.dist, np.where(self._population[i] == 1)[0])
        
        while(self._n_evals < evals_max):
            self._selectBinaryTourney(n_to_select=2)
            self.recombinatePopulation(p_cruce=1, o_cruce='p')
            self.mutate()
            self.replaceStationary()
        
        return self._population[np.argmax(self._fitness)], np.max(self._fitness)
    
    def am(self, evals_max, evals_max_bl, p_ls=0.1, period=10, best=False):
        self._initializePopulation()
        
        self._n_evals = len(self._population)
        for i in range(len(self._population)):
            self._fitness[i] = self.fObjetivo(self.dist, np.where(self._population[i] == 1)[0])
        
        generations = 1
        while(self._n_evals < evals_max):
            self._selectBinaryTourney(n_to_select=self.n_chromosome)
            self.recombinatePopulation(p_cruce=0.7, o_cruce='u')
            self.mutate()
            self.replaceGeneration()
            
            generations = (generations + 1) % period
            
            if (generations == 0):
                n_ls = int(len(self._population) * p_ls)
                self._n_evals += n_ls
                
                if n_ls == len(self._population):
                    chromosomes = range(len(self._population))
                elif best:
                    chromosomes = np.argsort(self._fitness)[-n_ls:]
                else:
                    chromosomes = np.random.choice(len(self._population), len(self._population), replace=False)
                
                for i in chromosomes:
                    sel = self.lsbf(np.where(self._population[i] == 1)[0], evals_max_bl)
                    self._population[i] = 0
                    self._population[i, sel] = 1
                    self._fitness[i] = self.fObjetivo(self.dist, sel)
        
        return self._population[np.argmax(self._fitness)], np.max(self._fitness)

    def _initializePopulation(self, n_sons=-1):
        self._population = np.zeros((self.n_chromosome, len(self.dist)), np.ubyte)
        
        if (n_sons == -1):
            self._sons_population = np.zeros_like(self._population)
        else:
            self._sons_population = np.zeros((n_sons, len(self.dist)), np.ubyte)
        
        for chromosome in self._population:
            index = np.random.choice(chromosome.shape[0], self.m, replace=False)
            chromosome[index] = 1

    def _selectBinaryTourney(self, n_to_select):
        contestants = np.random.choice(self.n_chromosome, (n_to_select, 2), replace=True)

        for i in range(len(contestants)):
            if (self._fitness[contestants[i, 0]] > self._fitness[contestants[i, 1]]):
                self._sons_population[i] = self._population[contestants[i, 0]]
            else:
                self._sons_population[i] = self._population[contestants[i, 1]]
    
    def recombinatePopulation(self, p_cruce=0.7, o_cruce='position'):
        n_recombinations = int(p_cruce * len(self._sons_population))
        
        if o_cruce == 'position' or o_cruce == 'p':
            f_cruce = self.recombinatePosition
        elif o_cruce == 'uniform' or o_cruce == 'u':
            f_cruce = self.recombinateUniform
        
        for i in range(0, n_recombinations, 2):
            self._sons_population[i], self._sons_population[i + 1] = f_cruce(self._sons_population[i], self._sons_population[i + 1])

    def recombinatePosition(self, father_a, father_b):
        son_a = father_a.copy()
        son_b = father_a.copy()

        index_diff_value = np.where(father_a != father_b)[0]

        rng_index_a = np.random.choice(index_diff_value, len(index_diff_value), replace=False)
        rng_index_b = np.random.choice(index_diff_value, len(index_diff_value), replace=False)
        
        son_a[index_diff_value] = father_a[rng_index_a]
        son_b[index_diff_value] = father_a[rng_index_b]
        
        return son_a, son_b
    
    def recombinateUniform(self, father_a, father_b):
        son_a = father_a.copy()
        son_b = father_b.copy()
        
        fathers = np.array([father_a, father_b])

        index_diff_value = np.where(father_a != father_b)[0]

        rng_index_a = np.random.randint(0, 2, len(index_diff_value))
        rng_index_b = np.random.randint(0, 2, len(index_diff_value))
        
        son_a[index_diff_value] = fathers[rng_index_a, index_diff_value]
        son_b[index_diff_value] = fathers[rng_index_b, index_diff_value]
        
        self.fixChromosome(son_a)
        self.fixChromosome(son_b)

        return son_a, son_b
    
    def fixChromosome(self, chromosome, keep_best=True):
        sel = np.where(chromosome == 1)[0]
        n_sel = len(sel)
        
        
        if n_sel > self.m:
            z = np.sum(dist[sel[:, None], sel], axis=0)
        
            if (keep_best):
                # Eliminar elementos de menor contribución
                while n_sel > self.m:
                    i = z.argmin()
                    chromosome[sel[i]] = 0
                    
                    z[i] = np.finfo(np.float64).max
                    z = z - dist[sel, sel[i]]
                    
                    n_sel -= 1
            else:
                # Eliminar elementos de mayor contribución
                while n_sel > self.m:
                    i = z.argmax()
                    chromosome[sel[i]] = 0
                    
                    z[i] = 0
                    z = z - dist[sel, sel[i]]
                    
                    n_sel -= 1
                
        elif n_sel < self.m:
            s = np.delete(np.arange(len(self.dist)), sel)
            
            z = np.sum(dist[sel[:, None], s], axis=0)
            # Añadir elementos de mayor contribución
            while n_sel < self.m:
                i = z.argmax()
                chromosome[s[i]] = 1
                
                z[i] = 0
                z = z + dist[s, s[i]]
                
                n_sel += 1
    
    def mutate(self, p_mutation=0.001):
        n_mutations = int(p_mutation * len(self._sons_population) * len(self.dist))
        
        idxs_chr = np.random.randint(0, len(self._sons_population), n_mutations)
        idxs_gen = np.random.randint(0, len(self.dist), n_mutations)
        
        for idx_chr, idx_gen_i in zip(idxs_chr, idxs_gen):
            tmp = self._sons_population[idx_chr, idx_gen_i]
            idx_gen_j = np.random.choice(np.where(self._sons_population[idx_chr] != tmp)[0])
        
            self._sons_population[idx_chr, idx_gen_i] = self._sons_population[idx_chr, idx_gen_j]
            self._sons_population[idx_chr, idx_gen_j] = tmp
    
    def replaceGeneration(self):
        best_old_chromosome = self._population[np.argmax(self._fitness)]
        best_old_fitness = np.max(self._fitness)

        self._n_evals += len(self._sons_population)

        for i in range(len(self._sons_population)):
            self._fitness[i] = self.fObjetivo(self.dist, np.where(self._sons_population[i] == 1)[0])

        if not best_old_chromosome in self._sons_population:
            worst_chr_idx = np.argmin(self._fitness)
            self._sons_population[worst_chr_idx] = best_old_chromosome
            self._fitness[worst_chr_idx] = best_old_fitness

        self._population = self._sons_population.copy()

    # Sustituir a los dos peores de la población actual
    def replaceStationary(self):
        self._n_evals += 2
        fitness_son = np.array([self.fObjetivo(self.dist, np.where(self._sons_population[0] == 1)[0]),
                                self.fObjetivo(self.dist, np.where(self._sons_population[1] == 1)[0])])
        
        for i in range(2):
            worst_parent = np.argmin(self._fitness)
            
            if fitness_son[i] > self._fitness[worst_parent]:
                self._population[worst_parent] = self._sons_population[i]
                self._fitness[worst_parent] = fitness_son[i]


list_directory = os.listdir('../BIN/')
list_directory.sort()

data_agg_u = []
data_agg_p = []
data_age_u = []
data_age_p = []
data_am_all = []
data_am_some = []
data_am_best = []

data_agg_u_ls = []
data_agg_p_ls = []
data_age_u_ls = []
data_age_p_ls = []
data_am_all_ls = []
data_am_some_ls = []
data_am_best_ls = []

evals = 100000
ag = AG_MDP(50)
# Itera por todos los casos de ejecución, usando greedy y búsqueda local
for filename in list_directory:
    dist, m = readDistances('../BIN/'+filename)
    ag.loadCase(dist, m)

    caso = filename.rstrip('.txt')

    np.random.seed(510)
    start = time.time()
    sel, f = ag.agg_uniform(evals)
    end = time.time()
    data_agg_u.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_agg_u_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

    np.random.seed(510)
    start = time.time()
    sel, f = ag.agg_position(evals)
    end = time.time()
    data_agg_p.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_agg_p_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

    np.random.seed(510)
    start = time.time()
    sel, f = ag.age_uniform(evals)
    end = time.time()
    data_age_u.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_age_u_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

    np.random.seed(510)
    start = time.time()
    sel, f = ag.age_position(evals)
    end = time.time()
    data_age_p.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_age_p_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

    np.random.seed(510)
    start = time.time()
    sel, f = ag.am(evals, 400, 1, 10)
    end = time.time()
    data_am_all.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_am_all_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

    np.random.seed(510)
    start = time.time()
    sel, f = ag.am(evals, 400, 0.1, 10)
    end = time.time()
    data_am_some.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_am_some_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

    np.random.seed(510)
    start = time.time()
    sel, f = ag.am(evals, 400, 0.1, 10, best=True)
    end = time.time()
    data_am_best.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : (end-start)})

    total = (end-start)
    start = time.time()
    sel = ag.lsbf(np.where(sel == 1)[0], evals)
    end = time.time()
    f = ag.fObjetivo(dist, sel)
    total += (end-start)
    data_am_best_ls.append({'Caso' : caso, 'Coste Obtenido' : f, 'Tiempo' : total})

df = pd.concat([pd.DataFrame(data_agg_u), pd.DataFrame(data_agg_p), 
                pd.DataFrame(data_age_u), pd.DataFrame(data_age_p),
                pd.DataFrame(data_am_all), pd.DataFrame(data_am_some),
                pd.DataFrame(data_am_best), pd.DataFrame(data_agg_u_ls), 
                pd.DataFrame(data_agg_p_ls), pd.DataFrame(data_age_u_ls), 
                pd.DataFrame(data_age_p_ls), pd.DataFrame(data_am_all_ls), 
                pd.DataFrame(data_am_some_ls), pd.DataFrame(data_am_best_ls)], 
               axis=1)
df.to_csv(r'../resultados.csv', index=False)