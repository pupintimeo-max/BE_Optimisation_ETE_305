import pandas as pd
 
# Facteurs d'émission par pays (kgCO2 / passager.km)
dict_ef_train = {
    "87": 0.0025,  # France
    "80": 0.0300,  # Allemagne
    "71": 0.0060,  # Espagne
    "83": 0.0150,  # Italie
    "84": 0.0250,  # Pays-Bas
    "88": 0.0200,  # Belgique
    "85": 0.0050,  # Suisse
    "70": 0.0350,  # Royaume-Uni
    "76": 0.0020,  # Norvège
    "81": 0.0150,  # Autriche
    "86": 0.0150,  # Danemark
    "54": 0.0400,  # République Tchèque
    "51": 0.0800,  # Pologne
    "55": 0.0500,  # Hongrie
    "24": 0.0300,  # Lituanie
    "74": 0.0100,  # Suède
    "82": 0.0280,  # Luxembourg
    "default": 0.0280 # Moyenne européenne (EEA)
}
 
 
dict_vitesse = {
    "87": 210,  # France
    "80": 150,  # Allemagne
    "71": 190,  # Espagne
    "83": 170,  # Italie
    "84": 130,  # Pays-Bas
    "88": 140,  # Belgique
    "85": 120,  # Suisse
    "70": 140,  # Royaume-Uni
    "76": 100,  # Norvège
    "81": 160,  # Autriche
    "86": 130,  # Danemark
    "54": 110,  # République Tchèque
    "51": 100,  # Pologne
    "55": 100,  # Hongrie
    "24": 90,   # Lituanie
    "74": 140,  # Suède
    "82": 110,  # Luxembourg
    "default": 130 
}
 
 
def get_estimated_distance(duree_min, uic_dep, uic_arr):
    """Calcule la distance en fonction de la vitesse moyenne des pays."""
    uic_dep_str, uic_arr_str = str(uic_dep), str(uic_arr)
    
    v_dep = dict_vitesse.get(uic_dep_str[:2], dict_vitesse["default"])
    v_arr = dict_vitesse.get(uic_arr_str[:2], dict_vitesse["default"])
    
    vitesse_moyenne = (v_dep + v_arr) / 2
    distance_km = (duree_min / 60) * vitesse_moyenne
    return distance_km
 
 
def get_train_emissions(distance_km, uic_dep, uic_arr):
    """
    Calcule les émissions CO2 pour un trajet train.
    Si trajet entre deux pays, fait la moyenne des facteurs d'émission.
    """
    uic_dep_str = str(uic_dep)
    uic_arr_str = str(uic_arr)
    
    # CAS PARTICULIER : EUROSTAR (Londres <-> France/Belgique/Pays-Bas)
    # On vérifie si l'un est Londres ET l'autre est sur le continent
    is_london = "7032732" in [uic_dep_str, uic_arr_str]
    is_continent = any(uic_str.startswith(('87', '88', '84')) for uic_str in [uic_dep_str, uic_arr_str])
    
    if is_london and is_continent:
        return distance_km * 0.006  # Facteur spécifique Eurostar
    
    # CAS GÉNÉRAL (Moyenne par pays)
    country_dep = uic_dep_str[:2]
    country_arr = uic_arr_str[:2]
    
    ef_dep = dict_ef_train.get(country_dep, dict_ef_train["default"])
    ef_arr = dict_ef_train.get(country_arr, dict_ef_train["default"])
    
    moyenne_ef = (ef_dep + ef_arr) / 2
    
    total_emissions = distance_km * moyenne_ef
    return total_emissions
 
 
def treatment(df):
    # df = pd.read_csv(file_path)
    df_emissions = df.copy()
    
    def apply_logic(row):
        dist = get_estimated_distance(row['duree_min'], row['iuc_dep'], row['iuc_arr'])
        return get_train_emissions(dist, row['iuc_dep'], row['iuc_arr'])
    
    df_emissions['Emissions_Train_kgCO2'] = df_emissions.apply(apply_logic, axis=1)
    
    # df.to_csv(file_path, index=False)
    print(f"Le dataframe a été mis à jour avec la colonne Emissions.")
    return df_emissions
 
# treatment()