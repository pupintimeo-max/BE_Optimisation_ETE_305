import numpy as np
import matplotlib.pyplot as plt

#  Paramètres de la fonction 
A = 0.8
B = 0.0085
C = 0.45

# Génération des temps de trajet (en minutes pour le calcul, de 30 min à 11h)
t_avion_min = np.linspace(30, 660, 500)
# Conversion en heures pour l'affichage X
t_avion_h = t_avion_min / 60

# Calcul du coefficient et du T_max
coeff_allongement = A * np.exp(-B * t_avion_min) + C
t_max_h = (t_avion_min * (1 + coeff_allongement)) / 60

#  Création du graphique 
fig, ax1 = plt.subplots(figsize=(11, 6))

# Axe X : Temps en HEURES
ax1.set_xlabel("Temps de trajet initial (heures)")

# Courbe 1 : Le Coefficient (Axe de gauche)
color = "tab:blue"
ax1.set_ylabel("Coefficient d'allongement max (%)", color=color)
ax1.plot(t_avion_h, coeff_allongement * 100, color=color, linewidth=2, label="Coeff %")
ax1.tick_params(axis="y", labelcolor=color)
ax1.grid(True, linestyle="--", alpha=0.6)

# Ajout dynamique des points de contrôle (basé sur tes paliers : 1h, 4h, 10h)
points_h = [1, 4, 10]

for h in points_h:
    t_min = h * 60
    # Calcul dynamique de la valeur réelle
    val_percent = (A * np.exp(-B * t_min) + C) * 100

    ax1.scatter(h, val_percent, color="red", zorder=5)
    # Affichage dynamique arrondi au pourcentage près
    ax1.annotate(
        f" {val_percent:.0f}% à {h}h",
        (h, val_percent),
        textcoords="offset points",
        xytext=(0, 10),
        fontweight="bold",
        color="red",
    )

# Courbe 2 : Le Temps Total T_max (Axe de droite en HEURES)
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("T_max : Temps total autorisé (heures)", color=color)
ax2.plot(
    t_avion_h, t_max_h, color=color, linestyle="--", linewidth=2, label="T_max total"
)
ax2.tick_params(axis="y", labelcolor=color)

plt.title(
    "Évolution de la contrainte de temps",
    fontsize=13,
)
fig.tight_layout()

# Affichage des valeurs clés en console
print(
    f"{'Temps initial (h)':<18} | {'Coeff calculé (%)':<18} | {'T_max final (h)':<15}"
)
print("-" * 55)
for h in points_h:
    t_min = h * 60
    c = A * np.exp(-B * t_min) + C
    tm_h = h * (1 + c)
    print(f"{h:>2}h ({t_min:>3} min)      | {c*100:>15.1f}% | {tm_h:>10.2f}h")

plt.show()
