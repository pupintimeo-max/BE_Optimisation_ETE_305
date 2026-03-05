import pulp
import pandas as pd
import itertools
import time
import numpy as np

# --- 1. Chargement des données ---
df_trains_arcs = pd.read_csv("trains_arcs_processed.csv")
df_planes_arcs = pd.read_csv("planes_arcs_processed.csv")
train_cities = pd.read_csv("train_cities.csv")


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
Modes = ["train", "avion"]
T_transf = 15
T_access = {"train": 30, "avion": 120}

# --- 2. Génération des combinaisons ---
villes_avec_gares = set(train_cities["Villes"].unique())
mask_gare = (df_planes_arcs["Ville_DEP"].isin(villes_avec_gares)) & (
    df_planes_arcs["Ville_DES"].isin(villes_avec_gares)
)
df_vols_eligibles = df_planes_arcs[mask_gare].drop_duplicates(
    subset=["Ville_DEP", "Ville_DES"]
)

combinaisons = list(
    df_vols_eligibles[["Ville_DEP", "Ville_DES"]].itertuples(index=False, name=None)
)[:20]

print(f"Début de l'optimisation sur {len(combinaisons)} trajets...")


# --- 3. Fonction d'Optimisation ---
def optimiser_trajet(O, D):
    key_avion = (O, D, "avion")
    if key_avion not in Arcs:
        return None

    emissions_avion, temps_avion_vol = Arcs[key_avion]
    temps_avion_total = temps_avion_vol + T_access["avion"]

    A = 0.8
    B = 0.0085
    C = 0.45
    T_max = temps_avion_total * (1 + (A * np.exp(-B * temps_avion_total) + C))

    prob = pulp.LpProblem(f"Optim_{O}_{D}", pulp.LpMinimize)
    relevant_arcs = list(Arcs.keys())
    villes_presentes = list(
        set([k[0] for k in relevant_arcs] + [k[1] for k in relevant_arcs])
    )

    x = pulp.LpVariable.dicts("x", relevant_arcs, cat=pulp.LpBinary)
    Transferts = [
        (i, m, k) for i in villes_presentes for m in Modes for k in Modes if m != k
    ]
    y = pulp.LpVariable.dicts("y", Transferts, cat=pulp.LpBinary)

    # Contraintes de flux
    for i in villes_presentes:
        in_f = pulp.lpSum([x[a] for a in relevant_arcs if a[1] == i])
        out_f = pulp.lpSum([x[a] for a in relevant_arcs if a[0] == i])
        if i == O:
            prob += out_f - in_f == 1
        elif i == D:
            prob += out_f - in_f == -1
        else:
            prob += out_f - in_f == 0

    # Expressions du temps et contrainte de temps (T_max)
    temps_trajet = pulp.lpSum(
        [x[a] * (Arcs[a][1] + T_access[a[2]]) for a in relevant_arcs]
    )
    temps_transfert = pulp.lpSum([y[t] * T_transf for t in Transferts])
    temps_total_expr = temps_trajet + temps_transfert

    prob += temps_total_expr <= T_max

    # Contraintes de transferts (Corrigées)
    for i, m, k in Transferts:
        if i not in [O, D]:
            arr = pulp.lpSum([x[a] for a in relevant_arcs if a[1] == i and a[2] == m])
            dep = pulp.lpSum([x[a] for a in relevant_arcs if a[0] == i and a[2] == k])

            # Force le transfert à 1 SI on arrive et repart par des modes différents
            prob += y[(i, m, k)] >= arr + dep - 1
            # INTERDIT le transfert SI on n'arrive pas par le mode m
            prob += y[(i, m, k)] <= arr
            # INTERDIT le transfert SI on ne repart pas par le mode k
            prob += y[(i, m, k)] <= dep

    # Fonction Objectif : Minimisation CO2 avec "Tie-Breaker" sur le temps
    emissions_totales = pulp.lpSum([x[a] * Arcs[a][0] for a in relevant_arcs])
    prob += (
        emissions_totales + 0.000001 * temps_total_expr,
        "Objectif_CO2_avec_TieBreaker",
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == "Optimal":
        arcs_sel = [a for a in relevant_arcs if pulp.value(x[a]) > 0.5]
        path, curr = [], O

        # Calcul précis des émissions réelles sans le poids du tie-breaker
        emissions_finales = sum([Arcs[a][0] for a in arcs_sel])

        while curr != D:
            for a in arcs_sel:
                if a[0] == curr:
                    path.append(f"{a[0]}->{a[1]} ({a[2]})")
                    curr = a[1]
                    break
            else:
                break

        temps_final_opti = pulp.value(temps_total_expr)

        return {
            "Départ": O,
            "Arrivée": D,
            "CO2_Avion_Direct": round(emissions_avion, 2),
            "CO2_Optimisé": round(emissions_finales, 2),
            "Gain_CO2_%": round(
                ((emissions_avion - emissions_finales) / emissions_avion) * 100, 2
            ),
            "Temps_Avion_Direct_min": round(temps_avion_total, 1),
            "Temps_Max_Autorisé_min": round(T_max, 1),
            "Temps_Optimisé_min": round(temps_final_opti, 1),
            "Allongement_min": round(temps_final_opti - temps_avion_total, 1),
            "Trajet_Retenu": " | ".join(path),
        }
    return None


# --- 4. Boucle de traitement ---
start_time = time.time()
resultats = []
for idx, (O, D) in enumerate(combinaisons):
    res = optimiser_trajet(O, D)
    if res:
        resultats.append(res)
    if (idx + 1) % 50 == 0:
        print(
            f"Progression : {idx + 1}/{len(combinaisons)}... ({time.time()-start_time:.1f}s)"
        )

# --- 5. Bilan Final ---
df_bilan = pd.DataFrame(resultats)

if not df_bilan.empty:
    total_co2_avion = df_bilan["CO2_Avion_Direct"].sum()
    total_co2_opti = df_bilan["CO2_Optimisé"].sum()
    gain_reseau_global = ((total_co2_avion - total_co2_opti) / total_co2_avion) * 100
    nb_alternatives = len(df_bilan[df_bilan["Gain_CO2_%"] > 0])

    print("\n" + "=" * 60)
    print("BILAN RÉSEAU GLOBAL")
    print("-" * 60)
    print(f"Trajets analysés            : {len(combinaisons)}")
    print(f"Alternatives réelles trouvées : {nb_alternatives}")
    print("-" * 60)
    print(f"Total CO2 (Référence Avion) : {total_co2_avion:,.0f} kg")
    print(f"Total CO2 (Optimisé)        : {total_co2_opti:,.0f} kg")
    print(f"GAIN RÉEL DU RÉSEAU         : {gain_reseau_global:.2f} %")
    print("-" * 60)

    if nb_alternatives > 0:
        moyenne_allongement = df_bilan[df_bilan["Gain_CO2_%"] > 0][
            "Allongement_min"
        ].mean()
        print(f"Allongement moyen (alternatives) : {moyenne_allongement:.1f} min")
    else:
        print("Allongement moyen (alternatives) : N/A (Aucun gain)")

    print("=" * 60)

    df_bilan.sort_values(by="Gain_CO2_%", ascending=False).to_csv(
        "bilan_airport_train.csv", index=False, sep=";", encoding="utf-8-sig"
    )
    print(
        f"Fichier exporté avec succès : 'bilan_airport_train.csv' ({len(df_bilan.columns)} colonnes)"
    )
