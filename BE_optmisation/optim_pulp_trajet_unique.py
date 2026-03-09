import pulp
import math
import numpy as np
import pandas as pd
import train_co2_emissions
import fuel_consumption_calc
import function_data
import itertools
import random

#  Données 
df_train = pd.read_csv("train_lines.csv")
train_cities = pd.read_csv("train_cities.csv")
df_plane = pd.read_csv("data/Flights_20191201_20191231.csv")

# Chargement direct des données déjà traitées
df_trains_arcs = pd.read_csv("trains_arcs_processed.csv")
df_planes_arcs = pd.read_csv("planes_arcs_processed.csv")


def create_arcs(df, mode):
    arcs = {}
    for row in df.itertuples(index=False):
        if mode == "train":
            key_1 = (row.ville_dep, row.ville_arr, mode)
            key_2 = (row.ville_arr, row.ville_dep, mode)
            value = (row.Emissions_Train_kgCO2, row.duree_min)
            for key in [key_1, key_2]:
                if key in arcs:
                    existing_emissions, existing_time = arcs[key]
                    if row.Emissions_Train_kgCO2 < existing_emissions:
                        arcs[key] = (row.Emissions_Train_kgCO2, row.duree_min)
                    elif (
                        row.Emissions_Train_kgCO2 == existing_emissions
                        and row.duree_min < existing_time
                    ):
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
                elif (
                    row.mean_emissions == existing_emissions
                    and row.mean_time < existing_time
                ):
                    arcs[key] = value
            else:
                arcs[key] = value
    return arcs


train_arcs = create_arcs(df_trains_arcs, "train")
plane_arcs = create_arcs(df_planes_arcs, "avion")

Arcs = train_arcs | plane_arcs
Modes = ["train", "avion"]

Villes_set = set()
for i, j, m in Arcs.keys():
    Villes_set.add(i)
    Villes_set.add(j)
Villes = list(Villes_set)

# SÉLECTION DES VILLES
print("--- OPTIMISATEUR DE TRAJET ---")

O = input("Entrez la ville de départ : ").strip()
while O not in Villes:
    O = input(f"Erreur. '{O}' n'est pas dans la liste. Réessayez : ").strip()

D = input("Entrez la ville d'arrivée : ").strip()
while D not in Villes or D == O:
    D = input("Erreur (ville inconnue ou identique au départ). Réessayez : ").strip()

print(f"\nRecherche d'une alternative verte de {O} à {D}...")

# Paramètres
T_transf = 15  # Temps de transfert en minutes
T_access = {"train": 30, "avion": 120}  # Temps d'accès en minutes
key_avion = (O, D, "avion")

# Définition de T_max
if key_avion in Arcs:
    temps_avion_total = Arcs[key_avion][1] + T_access["avion"]
    emissions_avion = Arcs[key_avion][0]
    temps_avion_vol = Arcs[key_avion][1]
    A = 0.53
    B = 0.0085
    C = 0.45
    T_max = temps_avion_total * (1 + (A * np.exp(-B * temps_avion_total) + C))
else:
    T_max = 420  # Sécurité si pas de vol direct
    temps_avion_total = None
    temps_avion_vol = None
    emissions_avion = None

# INITIALISATION DU MODÈLE
prob = pulp.LpProblem("Minimisation_Emissions", pulp.LpMinimize)

# Variables
x = pulp.LpVariable.dicts("x", Arcs.keys(), cat=pulp.LpBinary)
Transferts = [(i, m, k) for i in Villes for m in Modes for k in Modes if m != k]
y = pulp.LpVariable.dicts("y", Transferts, cat=pulp.LpBinary)

# Fonction Objectif
# rajout d'un poids sur le temps de trajet pour éviter des abérations
temps_trajet = pulp.lpSum(
    [x[(i, j, m)] * (Arcs[(i, j, m)][1] + T_access[m]) for (i, j, m) in Arcs.keys()]
)
temps_transfert = pulp.lpSum([y[(i, m, k)] * T_transf for (i, m, k) in Transferts])

prob += (
    pulp.lpSum([x[(i, j, m)] * Arcs[(i, j, m)][0] for (i, j, m) in Arcs.keys()])
    + 0.000001 * (temps_trajet + temps_transfert),
    "Emissions",
)

# Contraintes de flux
for i in Villes:
    in_flow = pulp.lpSum(
        [x[(j, ville_i, m)] for (j, ville_i, m) in Arcs.keys() if ville_i == i]
    )
    out_flow = pulp.lpSum(
        [x[(ville_i, j, m)] for (ville_i, j, m) in Arcs.keys() if ville_i == i]
    )

    if i == O:
        prob += out_flow - in_flow == 1
    elif i == D:
        prob += out_flow - in_flow == -1
    else:
        prob += out_flow - in_flow == 0

# Contraintes de transferts
for i in Villes:
    if i not in [O, D]:
        for m in Modes:
            for k in Modes:
                if m != k:
                    arrivee_m = pulp.lpSum(
                        [
                            x[(j, ville_i, mode)]
                            for (j, ville_i, mode) in Arcs.keys()
                            if ville_i == i and mode == m
                        ]
                    )
                    depart_k = pulp.lpSum(
                        [
                            x[(ville_i, l, mode)]
                            for (ville_i, l, mode) in Arcs.keys()
                            if ville_i == i and mode == k
                        ]
                    )
                    prob += y[(i, m, k)] >= arrivee_m + depart_k - 1

# Contrainte de Temps (T_max)
temps_trajet = pulp.lpSum(
    [x[(i, j, m)] * (Arcs[(i, j, m)][1] + T_access[m]) for (i, j, m) in Arcs.keys()]
)
temps_transfert = pulp.lpSum([y[(i, m, k)] * T_transf for (i, m, k) in Transferts])
prob += temps_trajet + temps_transfert <= T_max

# Résolution
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# AFFICHAGE DES RÉSULTATS 
print("\n" + "=" * 50)
print(" TRAJET INITIAL EN AVION (Référence) :")
if key_avion in Arcs:
    print(
        f"  Temps total avec accès : {temps_avion_total} minutes ({temps_avion_total/60:.1f}h)"
    )
    print(f"  Temps de vol : {temps_avion_vol} minutes ({temps_avion_vol/60:.1f}h)")

    print(f"  Émissions totales      : {emissions_avion:.2f} kgCO2")
else:
    print("  Aucun vol direct référencé dans les données.")

print("-" * 50)

# Afficher le résultat de l'optimisation
if prob.status == pulp.LpStatusOptimal:
    print(" ALTERNATIVE MULTIMODALE TROUVÉE :")
    temps_total_final = pulp.value(temps_trajet + temps_transfert)
    emissions_finales = pulp.value(prob.objective)

    print(f"  Temps max autorisé : {T_max/60:.1f}h")
    print(
        f"  Temps total final  : {temps_total_final:.0f} minutes ({temps_total_final/60:.1f}h)"
    )
    print(f"  Émissions finales  : {emissions_finales:.2f} kgCO2")

    # Calcul du gain de CO2
    if key_avion in Arcs:
        gain_co2 = ((emissions_avion - emissions_finales) / emissions_avion) * 100
        print(f"   Bilan Carbone    : Émissions réduites de {gain_co2:.1f}%")

    print("\n Chemin emprunté :")

    #  EXTRACTION ET CALCULS DE DÉCOMPOSITION DU TEMPS 
    temps_effectif_total = 0
    temps_access_total = 0
    temps_transfert_total = 0

    for i, j, m in Arcs.keys():
        if x[(i, j, m)].varValue is not None and x[(i, j, m)].varValue > 0.5:
            emissions_etape = Arcs[(i, j, m)][0]
            duree_etape = Arcs[(i, j, m)][1]

            temps_effectif_total += duree_etape
            temps_access_total += T_access[m]

            print(
                f"  -> {i} à {j} en {m.upper()} ({emissions_etape:.2f} kgCO2 | {duree_etape:.0f} min)"
            )

    for i, m, k in Transferts:
        if y[(i, m, k)].varValue is not None and y[(i, m, k)].varValue > 0.5:
            temps_transfert_total += T_transf
            print(f"  ->  Transfert à {i} : {m} vers {k} ({T_transf} min)")

    print("\n⏱ Décomposition du temps :")
    print(
        f"  - Temps de trajet effectif : {temps_effectif_total:.0f} minutes ({temps_effectif_total/60:.1f}h)"
    )
    print(f"  - Temps d'accès aux gares/aéroports : {temps_access_total:.0f} minutes")
    print(
        f"  - Temps de transfert (correspondances) : {temps_transfert_total:.0f} minutes"
    )

else:
    print(" AUCUNE ALTERNATIVE RESPECTANT LE TEMPS MAX.")
    print("  Maintien du trajet initial en avion par défaut.")

print("=" * 50)
