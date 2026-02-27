import pulp
import math
import numpy as np
import pandas as pd
import train_co2_emissions
import fuel_consumption_calc
import function_data
import itertools
import random

# Données

df_train = pd.read_csv('train_lines.csv')
train_cities = pd.read_csv('train_cities.csv')
df_plane = pd.read_csv('data/Flights_20191201_20191231.csv')

# on crée le dictionnaire des arcs sous la forme : (ville_dep, ville_arr, mode) : (emissions_kgCO2, duree_min)
df_trains_arcs = train_co2_emissions.treatment(df_train)
df_planes_arcs = function_data.treatment(df_plane)

def create_arcs(df, mode):
    arcs = {}
    for row in df.itertuples(index=False):
        if mode == "train" :
            key_1 = (row.ville_dep, row.ville_arr, mode)
            key_2 = (row.ville_arr, row.ville_dep, mode)
            value = (row.Emissions_Train_kgCO2, row.duree_min)
            # Si l'arc existe déjà, on peut choisir de garder le moins émetteur ou le plus rapide (dans le cas ou une ville a deux aéroports ou deux gares)
            for key in [key_1, key_2]:
                if key in arcs:
                    existing_emissions, existing_time = arcs[key]
                    if row.Emissions_Train_kgCO2 < existing_emissions:
                        arcs[key] = (row.Emissions_Train_kgCO2, row.duree_min)
                    elif row.Emissions_Train_kgCO2 == existing_emissions and row.duree_min < existing_time:
                        arcs[key] = (row.Emissions_Train_kgCO2, row.duree_min)
                else:
                    arcs[key] = value
        elif mode == "avion":
            key = (row.Ville_DEP, row.Ville_DES, mode)
            value = (row.mean_emissions, row.mean_time)
            if key in arcs:
                existing_emissions, existing_time = arcs[key]
                if row.mean_emissions < existing_emissions:
                    arcs[key] = value
                elif row.mean_emissions == existing_emissions and row.mean_time < existing_time:
                    arcs[key] = value
            else:
                arcs[key] = value
    return arcs

train_arcs = create_arcs(df_trains_arcs, "train")
plane_arcs = create_arcs(df_planes_arcs, "avion")

Arcs = train_arcs | plane_arcs
Modes = ["train", "avion"]
Villes = train_cities["Villes"].tolist()

#for arcs in itertools.permutations(Villes, 2):
ind1 = random.randint(0, len(Villes)-1)
ind2 = random.randint(0, len(Villes)-1)
# --- 1. DONNÉES FICTIVES (Pour l'exemple) ---
while ind1 == ind2:
    ind2 = random.randint(0, len(Villes)-1)

O = Villes[ind1]
D = Villes[ind2]
print(f"Optimisation du trajet de {O} à {D}")

# Paramètres
T_transf = 15 # Temps de transfert en minutes
T_access = {"train": 30, "avion": 120} # Temps d'accès en minutes
k = 0.5
lb = 0.01
key = (O, D, "avion")
T_max = Arcs[key][1] * (1 + k*np.exp(lb*Arcs[key][1])) if key in Arcs else 300 # Si pas d'arc direct, on fixe un temps max arbitraire de 5h (300 min)

# --- 2. INITIALISATION DU MODÈLE ---
prob = pulp.LpProblem("Minimisation_Emissions_Multimodal", pulp.LpMinimize)

# --- 3. VARIABLES DE DÉCISION ---
# Variable x_ij^m
x = pulp.LpVariable.dicts("x", Arcs.keys(), cat=pulp.LpBinary)

# Variable y_i^{m,k} (Nœud, Mode_Arrivée, Mode_Départ)
# On ne crée les variables de transfert que pour les croisements de modes différents
Transferts = [(i, m, k) for i in Villes for m in Modes for k in Modes if m != k]
y = pulp.LpVariable.dicts("y", Transferts, cat=pulp.LpBinary)

# --- 4. FONCTION OBJECTIF ---
# Minimiser les émissions : sum(x_ij^m * E_ij^m)
prob += pulp.lpSum([x[(i, j, m)] * Arcs[(i, j, m)][0] for (i, j, m) in Arcs.keys()]), "Emissions_Totales"

# --- 5. CONTRAINTES ---

# A. Conservation du flux
for i in Villes:
    # Ce qui rentre dans la ville 'i'
    in_flow = pulp.lpSum([x[(j, ville_i, m)] for (j, ville_i, m) in Arcs.keys() if ville_i == i])
    # Ce qui sort de la ville 'i'
    out_flow = pulp.lpSum([x[(ville_i, j, m)] for (ville_i, j, m) in Arcs.keys() if ville_i == i])
    
    if i == O:
        prob += out_flow - in_flow == 1, f"Flux_Origine_{i}"
    elif i == D:
        prob += out_flow - in_flow == -1, f"Flux_Destination_{i}"
    else:
        prob += out_flow - in_flow == 0, f"Flux_Intermediaire_{i}"

# B. Contrainte de liaison logique entre x et y (Crucial)
for i in Villes:
    if i not in [O, D]: # Pas de transfert à l'origine absolue ou destination finale
        for m in Modes:
            for k in Modes:
                if m != k:
                    # Somme des arrivées en mode m à la ville i
                    arrivee_m = pulp.lpSum([x[(j, ville_i, mode)] for (j, ville_i, mode) in Arcs.keys() if ville_i == i and mode == m])
                    # Somme des départs en mode k de la ville i
                    depart_k = pulp.lpSum([x[(ville_i, l, mode)] for (ville_i, l, mode) in Arcs.keys() if ville_i == i and mode == k])
                    
                    # Si on arrive en 'm' ET qu'on repart en 'k', alors (1 + 1 - 1 = 1) force y à être >= 1
                    prob += y[(i, m, k)] >= arrivee_m + depart_k - 1, f"Liaison_Transfert_{i}_{m}_{k}"

# C. Contrainte de Qualité de Service (Temps Max)
temps_trajet = pulp.lpSum([x[(i, j, m)] * (Arcs[(i, j, m)][1] + T_access[m]) for (i, j, m) in Arcs.keys()])
temps_transfert = pulp.lpSum([y[(i, m, k)] * T_transf for (i, m, k) in Transferts])

prob += (temps_trajet + temps_transfert <= T_max), "Contrainte_QoS_Temps_Max"

# --- 6. RÉSOLUTION ---
prob.solve()

# --- 7. AFFICHAGE DES RÉSULTATS ---
print(f"Statut : {pulp.LpStatus[prob.status]}")
print(f"Émissions totales (Objectif) : {pulp.value(prob.objective)}")

print("\nChemin emprunté :")
for (i, j, m) in Arcs.keys():
    if x[(i, j, m)].varValue == 1:
        print(f"-> Trajet de {i} à {j} en {m}")

for (i, m, k) in Transferts:
    if y[(i, m, k)].varValue == 1:
        print(f"-> ** Transfert à {i} du mode {m} au mode {k} **")