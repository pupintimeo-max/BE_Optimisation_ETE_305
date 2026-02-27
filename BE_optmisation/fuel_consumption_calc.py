import pandas as pd
import numpy as np

feat_model_df = pd.read_csv('data/ac_model_coefficients.csv')

def calculer_fuel_feat(flight_row):
    aircraft = flight_row['AC Type']
    print("Type of aircraft : ",aircraft)
    dist_nm = flight_row['Actual Distance Flown (nm)']
    dist_km = dist_nm * 1.852
    print("Distance flown (km) : " , dist_km)

    if aircraft in feat_model_df['ac_code_icao'].values :
        coefficients = feat_model_df[feat_model_df['ac_code_icao'] == aircraft]

        alpha = coefficients['reduced_fuel_a1'].values[0]
        beta = coefficients['reduced_fuel_a2'].values[0]
        gamma = coefficients['reduced_fuel_intercept'].values[0]
        
        # Formule quadratique FEAT
        return alpha * (dist_km**2) + beta * dist_km + gamma
    return None
