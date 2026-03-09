import pandas as pd
import numpy as np
from dico_villes import dico_villes
from fuel_consumption_calc import calculer_fuel_feat

feat_model_df = pd.read_csv("./data/ac_model_coefficients.csv")

file_path_ac = "./data/ac_model_coefficients_airliners.csv"

blacklist_operators = [
    "ZZZ",
    "FDX",
    "UPS",
    "BCS",
    "TAY",
    "SRR",
    "NPT",
    "SWN",
    "MNB",
    "AIZ",
    "SRN",
    "ABR",
    "SWT",
    "NJE",
    "VJT",
    "AHO",
    "GAC",
    "SVW",
    "MMD",
    "DOP",
    "BHL",
    "ELY",
    "ISR",
    "HKS",
    "AWC",
    "MSA",
]


def select_top_villles(df_v, n=10):
    list_top = np.arange(1, n + 1, 1)
    mask_villes = df_v["Rang"].isin(list_top)

    top_v = df_v[mask_villes]
    return top_v


def select_flights_train(df_v, df_F):

    airports_france = df_v["ICAO"].unique()

    vols_France_mask = df_F["ADEP"].isin(airports_france) & df_F["ADES"].isin(
        airports_france
    )

    vols_France = df_F[vols_France_mask]

    vols_France = vols_France[vols_France["ICAO"] != "ZZZ"]


    return vols_France


def select_flights_europe(df):
    codes_europe = ["L", "E", "B"]

    filtre_europe = df["ADEP"].str[0].isin(codes_europe) & df["ADES"].str[0].isin(
        codes_europe
    )

    vols_europe = df[filtre_europe]

    print(f"Nombre de vols europe : {len(vols_europe)}")

    return vols_europe


def select_top_companies(df, n):
    df_filtered = df[~df["AC Operator"].isin(blacklist_operators)]
    classement = df_filtered["AC Operator"].value_counts().reset_index()
    top_n = classement.head(n)

    return top_n


def filter_planes_and_companies(df, df_ac, df_comp):
    all_ac = df_ac["ac_code_icao"].values
    all_companies = df_comp["AC Operator"].values

    df_filtered_ac = df[df["AC Type"].isin(all_ac)]
    df_filtered = df_filtered_ac[df_filtered_ac["AC Operator"].isin(all_companies)]

    print(f"Nombre de vols europe filtrés : {len(df_filtered)}")

    return df_filtered


def all_liaisons(df, seuil_vols=60):
    df_temp = df.copy()

    routes_triees = np.sort(df_temp[["ADEP", "ADES"]].values, axis=1)

    df_temp["Aeroport_1"] = routes_triees[:, 0]
    df_temp["Aeroport_2"] = routes_triees[:, 1]

    classement_liaisons = df_temp.groupby(["Aeroport_1", "Aeroport_2"]).size()

    classement_liaisons = classement_liaisons.sort_values(ascending=False)
    classement_liaisons = classement_liaisons.reset_index(name="Nombre_de_vols")

    classement_liaisons = classement_liaisons[
        classement_liaisons["Nombre_de_vols"] >= seuil_vols
    ]


    return classement_liaisons


def name_liaison(df_liaisons, dico):

    df_liaisons_copy = df_liaisons.copy()

    df_liaisons_copy["Ville_1"] = df_liaisons_copy["Aeroport_1"].map(dico)
    df_liaisons_copy["Ville_2"] = df_liaisons_copy["Aeroport_2"].map(dico)

    df_liaisons_copy = df_liaisons_copy.dropna(subset=["Ville_1", "Ville_2"])



    return df_liaisons_copy


def calculate_emmissions(df_flights, df_feat_model):

    df_flights_copy = df_flights.copy()

    merged_df = df_flights_copy.merge(
        df_feat_model[
            [
                "ac_code_icao",
                "reduced_fuel_a1",
                "reduced_fuel_a2",
                "reduced_fuel_intercept",
            ]
        ],
        left_on="AC Type",
        right_on="ac_code_icao",
        how="left",
    )

    merged_df["Actual Distance Flown (km)"] = (
        merged_df["Actual Distance Flown (nm)"] * 1.852
    )
    df_flights_copy["Fuel_Emissions"] = (
        merged_df["reduced_fuel_a1"] * (merged_df["Actual Distance Flown (km)"] ** 2)
        + merged_df["reduced_fuel_a2"] * merged_df["Actual Distance Flown (km)"]
        + merged_df["reduced_fuel_intercept"]
    ).values

    return df_flights_copy


def mean_flight(df_flights):
    """
    Agrège les vols pour obtenir une moyenne des émissions et du temps
    par liaison (Aéroport A -> Aéroport B).
    """
    df = df_flights.copy()

    df["OFF_BLOCK"] = pd.to_datetime(
        df["ACTUAL OFF BLOCK TIME"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    df["ARRIVAL"] = pd.to_datetime(
        df["ACTUAL ARRIVAL TIME"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )

    df["Flight_Time_min"] = (df["ARRIVAL"] - df["OFF_BLOCK"]).dt.total_seconds() / 60.0


    edges_df = (
        df.groupby(["ADEP", "ADES"])
        .agg(
            mean_emissions=("Fuel_Emissions", "mean"),
            mean_time=("Flight_Time_min", "mean"),
            num_flights=("Fuel_Emissions", "count"),
        )
        .reset_index()
    )

    return edges_df


def filter_mean(edges_df, dico, all_liaisons):
    """
    Filtre les liaisons orientées (A -> B) issues de mean_flight
    selon un nombre minimum de vols, sans mélanger l'aller et le retour.
    """
    df_temp = edges_df.copy()
    copy_all_liasons = all_liaisons.copy()

    copy_all_liasons_aller = copy_all_liasons.rename(
        columns={"Aeroport_1": "ADEP", "Aeroport_2": "ADES"}
    )
    copy_all_liasons_retour = copy_all_liasons.rename(
        columns={"Aeroport_1": "ADES", "Aeroport_2": "ADEP"}
    )

    all_aller = pd.merge(
        df_temp,
        copy_all_liasons_aller[["ADEP", "ADES"]],
        on=["ADES", "ADEP"],
        how="inner",
    )
    all_retour = pd.merge(
        df_temp,
        copy_all_liasons_retour[["ADEP", "ADES"]],
        on=["ADES", "ADEP"],
        how="inner",
    )

    liaisons_filtrees = pd.concat([all_aller, all_retour])

    liaisons_finales = liaisons_filtrees.sort_values(by="num_flights", ascending=False)

    liaisons_finales["Ville_DEP"] = liaisons_finales["ADEP"].map(dico)
    liaisons_finales["Ville_DES"] = liaisons_finales["ADES"].map(dico)

    return liaisons_finales


def get_vols_liaison(df, code_depart, code_arrivee, sens_confondus=False):
    """
    Sélectionne les vols entre deux aéroports.

    Paramètres:
    - df : Le DataFrame contenant les vols
    - code_depart : Code ICAO de l'aéroport 1 (ex: 'LFPG')
    - code_arrivee : Code ICAO de l'aéroport 2 (ex: 'LFMN')
    - sens_confondus : Si True, récupère A->B ET B->A (Aller-Retour)
    """
    condition_aller = (df["ADEP"] == code_depart) & (df["ADES"] == code_arrivee)

    if not sens_confondus:
        resultat = df[condition_aller]

    else:
        condition_retour = (df["ADEP"] == code_arrivee) & (df["ADES"] == code_depart)
        resultat = df[condition_aller | condition_retour]

    return resultat

def treatment(df) :
    df_europe = select_flights_europe(df)
    top_companies = select_top_companies(df_europe, 50)

    df_final_flights = filter_planes_and_companies(df_europe, feat_model_df, top_companies)
    df_liaisons = all_liaisons(df_final_flights)

    df_final_flights_emissions = calculate_emmissions(df_final_flights, feat_model_df)
    mean = mean_flight(df_final_flights_emissions)
    filtred_mean = filter_mean(mean, dico_villes, df_liaisons)
    return filtred_mean


