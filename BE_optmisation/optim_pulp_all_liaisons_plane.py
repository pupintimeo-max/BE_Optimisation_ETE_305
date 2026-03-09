import pulp
import pandas as pd
import time
import numpy as np

# Chargement des données 
df_trains_arcs = pd.read_csv("BE_optmisation/trains_arcs_processed.csv")
df_planes_arcs = pd.read_csv("BE_optmisation/planes_arcs_processed.csv")


def create_arcs(df, mode):
    arcs = {}
    for row in df.itertuples(index=False):
        if mode == "train":
            key_1, key_2 = (row.ville_dep, row.ville_arr, mode), (
                row.ville_arr,
                row.ville_dep,
                mode,
            )
            val = (row.Emissions_Train_kgCO2, row.duree_min)
            for k in [key_1, key_2]:
                if k not in arcs or val[0] < arcs[k][0]:
                    arcs[k] = val
        elif mode == "avion":
            key = (row.Ville_DEP, row.Ville_DES, mode)
            val = (row.mean_emissions, row.mean_time)
            if key not in arcs or val[0] < arcs[key][0]:
                arcs[key] = val
    return arcs


train_arcs = create_arcs(df_trains_arcs, "train")
plane_arcs = create_arcs(df_planes_arcs, "avion")
Arcs = {**train_arcs, **plane_arcs}

# Paramètres
Modes = ["train", "avion"]
T_transf = 15
T_access = {"train": 30, "avion": 120}

relevant_arcs_keys = list(Arcs.keys())
villes_presentes = list(
    set([k[0] for k in relevant_arcs_keys] + [k[1] for k in relevant_arcs_keys])
)
Transferts_keys = [
    (i, m, k) for i in villes_presentes for m in Modes for k in Modes if m != k
]

# Préparation des trajets à tester 
liaisons_a_tester = (
    df_planes_arcs[["Ville_DEP", "Ville_DES"]].drop_duplicates(keep="first")#.head()
)

print(f"Analyse de {len(liaisons_a_tester)} liaisons aériennes réelles...")


# Fonction d'Optimisation
def optimiser_trajet(O, D):
    key_avion = (O, D, "avion")
    if key_avion not in Arcs:
        return None

    emissions_avion, temps_avion_vol = Arcs[key_avion]
    temps_avion_total = temps_avion_vol + T_access["avion"]

    # Formule exponentielle ajustée
    A = 0.53
    B = 0.0085
    C = 0.48
    T_max = temps_avion_total * (1 + (A * np.exp(-B * temps_avion_total) + C))

    prob = pulp.LpProblem(f"Optim_{O}_{D}", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", relevant_arcs_keys, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", Transferts_keys, cat=pulp.LpBinary)

    # Expressions de temps
    temps_trajet = pulp.lpSum(
        [x[a] * (Arcs[a][1] + T_access[a[2]]) for a in relevant_arcs_keys]
    )
    temps_transfert = pulp.lpSum([y[t] * T_transf for t in Transferts_keys])
    temps_total_expr = temps_trajet + temps_transfert

    # Contraintes de flux
    for i in villes_presentes:
        in_f = pulp.lpSum([x[a] for a in relevant_arcs_keys if a[1] == i])
        out_f = pulp.lpSum([x[a] for a in relevant_arcs_keys if a[0] == i])
        if i == O:
            prob += out_f - in_f == 1
        elif i == D:
            prob += out_f - in_f == -1
        else:
            prob += out_f - in_f == 0

    # Contrainte de temps T_max
    prob += temps_total_expr <= T_max

    # Contraintes de transferts (Corrigées pour éviter les transferts fantômes)
    for i, m, k in Transferts_keys:
        if i not in [O, D]:
            arr = pulp.lpSum(
                [x[a] for a in relevant_arcs_keys if a[1] == i and a[2] == m]
            )
            dep = pulp.lpSum(
                [x[a] for a in relevant_arcs_keys if a[0] == i and a[2] == k]
            )

            prob += y[(i, m, k)] >= arr + dep - 1
            prob += y[(i, m, k)] <= arr
            prob += y[(i, m, k)] <= dep

    # Fonction Objectif avec Tie-Breaker
    emissions_totales = pulp.lpSum([x[a] * Arcs[a][0] for a in relevant_arcs_keys])
    prob += emissions_totales + 0.000001 * temps_total_expr, "Objectif"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == "Optimal":
        # Reconstruction du chemin
        arcs_sel = [a for a in relevant_arcs_keys if pulp.value(x[a]) > 0.5]
        path, curr = [], O

        # Initialisation des compteurs de temps pour la décomposition
        t_effectif = 0
        t_access = 0
        t_transfert = 0
        emissions_finales = 0

        # Calcul précis sans le poids du tie-breaker
        for a in arcs_sel:
            emissions_finales += Arcs[a][0]
            t_effectif += Arcs[a][1]
            t_access += T_access[a[2]]

        for t in Transferts_keys:
            if pulp.value(y[t]) > 0.5:
                t_transfert += T_transf

        # Ordonnancement de l'itinéraire
        while curr != D:
            found = False
            for a in arcs_sel:
                if a[0] == curr:
                    path.append(f"{a[0]}->{a[1]} ({a[2]})")
                    curr = a[1]
                    found = True
                    break
            if not found:
                break

        temps_final_opti = pulp.value(temps_total_expr)

        return {
            "Départ": O,
            "Arrivée": D,
            "CO2_Avion_Direct": round(emissions_avion, 2),
            "CO2_Optimisé": round(emissions_finales, 2),
            "Gain_CO2_kg": round(emissions_avion - emissions_finales, 2),
            "Gain_CO2_%": round(
                ((emissions_avion - emissions_finales) / emissions_avion) * 100, 2
            ),
            "Temps_Avion_Direct_min": round(temps_avion_total, 1),
            "Temps_Max_Autorisé_min": round(T_max, 1),
            "Temps_Optimisé_Total_min": round(temps_final_opti, 1),
            "Temps_Effectif_min": round(t_effectif, 1),
            "Temps_Access_min": round(t_access, 1),
            "Temps_Transfert_min": round(t_transfert, 1),
            "Allongement_min": round(temps_final_opti - temps_avion_total, 1),
            "Trajet_Retenu": " | ".join(path),
        }
    return None


# Exécution 
start_time = time.time()
resultats = []

for idx, row in enumerate(liaisons_a_tester.itertuples()):
    res = optimiser_trajet(row.Ville_DEP, row.Ville_DES)
    if res:
        resultats.append(res)

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(
            f"Progression : {idx + 1}/{len(liaisons_a_tester)} liaisons traitées... ({elapsed:.1f}s)"
        )

# Bilan Final 
df_bilan = pd.DataFrame(resultats)

if not df_bilan.empty:
    total_avion = df_bilan["CO2_Avion_Direct"].sum()
    total_opti = df_bilan["CO2_Optimisé"].sum()
    gain_reseau_global = ((total_avion - total_opti) / total_avion) * 100

    nb_alternatives = len(df_bilan[df_bilan["Gain_CO2_%"] > 0])

    print("\n" + "=" * 60)
    print("BILAN ÉCOLOGIQUE DU RÉSEAU AÉRIEN")
    print("-" * 60)
    print(f"Liaisons traitées           : {len(df_bilan)}")
    print(f"Alternatives réelles        : {nb_alternatives} trajets")
    print("-" * 60)
    print(f"Total CO2 (Référence Avion) : {total_avion:,.0f} kg")
    print(f"Total CO2 (Optimisé)        : {total_opti:,.0f} kg")
    print(f"GAIN RÉEL GLOBAL DU RÉSEAU  : {gain_reseau_global:.2f} %")
    print("-" * 60)

    if nb_alternatives > 0:
        moyenne_allongement = df_bilan[df_bilan["Gain_CO2_%"] > 0][
            "Allongement_min"
        ].mean()
        print(f"Allongement moyen (sur alternatives) : {moyenne_allongement:.1f} min")
    else:
        print("Allongement moyen (sur alternatives) : N/A (Aucun gain)")

    print("=" * 60)

    df_bilan.to_csv(
        "optimisation_vols_existants.csv", index=False, sep=";", encoding="utf-8-sig"
    )
    print(
        f"Résultats sauvegardés dans 'optimisation_vols_existants.csv' ({len(df_bilan.columns)} colonnes)"
    )
else:
    print("Aucun résultat généré.")
