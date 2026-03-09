# BE_Optimisation_ETE_305

Projet d'optimisation de trajets inter-villes en Europe, orienté **réduction des émissions de CO₂** sous contrainte de temps, via un modèle **multimodal train + avion** (programmation linéaire en nombres entiers avec PuLP).

## 1) Objectif du dépôt

Le dépôt implémente une chaîne de traitement en 3 briques :

1. **Estimation des émissions**
	 - train : à partir de temps de trajet ferroviaire et facteurs pays ;
	 - avion : à partir de coefficients FEAT par type d'appareil et distances observées.
2. **Construction d'un graphe multimodal** de villes (arcs train/avion avec coût CO₂ et durée).
3. **Optimisation de chemin** entre origine/destination :
	 - objectif = minimiser les émissions ;
	 - contrainte = ne pas dépasser un temps maximal dérivé du trajet avion de référence.

## 2) Arborescence

```text
.
├── README.md
└── BE_optmisation/
		├── dico_villes.py
		├── fuel_consumption_calc.py
		├── function_data.py
		├── optim_pulp_trajet_unique.py
		├── optim_pulp_all_liaisons_plane.py
		├── optim_pulp_all_liaisons_plane_gare.py
		├── t_max.py
		├── train_co2_emissions.py
		├── .gitignore
		└── data/
				├── ac_model_coefficients.csv
				└── ac_model_coefficients_airliners.csv
```

## 3) Description des fichiers

### Scripts de préparation des données

- **`BE_optmisation/train_co2_emissions.py`**
	- Définit des facteurs d'émission par pays (`dict_ef_train`) et vitesses moyennes par pays (`dict_vitesse`).
	- Estime la distance train à partir de `duree_min`.
	- Calcule `Emissions_Train_kgCO2` pour chaque liaison de `train_lines.csv`.
	- Contient un cas spécifique Eurostar (Londres ↔ continent).

- **`BE_optmisation/fuel_consumption_calc.py`**
	- Calcule la consommation carburant estimée par vol via une formule quadratique FEAT 

- **`BE_optmisation/function_data.py`**
	- Pipeline principal côté vols :
		- filtrage Europe,
		- exclusion d'opérateurs (cargo/privé/etc.),
		- sélection des types avion connus,
		- calcul d'émissions vol,
		- agrégation par liaison orientée (émissions moyennes, temps moyen, nombre de vols),
		- mapping codes ICAO → villes via `dico_villes.py`.
	- Fonction d'entrée importante : `treatment(df)` qui renvoie un DataFrame d'arcs avion prêts pour optimisation.

- **`BE_optmisation/dico_villes.py`**
	- Grand dictionnaire de correspondance `ICAO -> Ville`.
	- Gère explicitement les villes multi-aéroports (ex: Londres, Paris, Milan, Rome...).

### Scripts d'optimisation

- **`BE_optmisation/optim_pulp.py`**
	- Version « prototype » : construit arcs train/avion et optimise un trajet origine-destination tiré aléatoirement.
	- Dépend d'un fichier vols brut (`data/Flights_20191201_20191231.csv`) non versionné ici.

- **`BE_optmisation/optim_pulp_trajet_unique.py`**
	- Version interactive (saisie utilisateur de la ville de départ/arrivée).
	- Travaille à partir de fichiers d'arcs pré-traités :
		- `trains_arcs_processed.csv`
		- `planes_arcs_processed.csv`
	- Affiche : trajet avion de référence, alternative optimisée, gain CO₂, et décomposition du temps.

- **`BE_optmisation/optim_pulp_all_liaisons_plane.py`**
	- Exécute l'optimisation sur un ensemble de liaisons aériennes existantes.
	- Produit un bilan global réseau et exporte `optimisation_vols_existants.csv`.

- **`BE_optmisation/optim_pulp_all_liaisons_plane_gare.py`**
	- Variante batch limitée aux liaisons avion dont les deux villes ont une gare connue (`train_cities.csv`).
	- Exporte `bilan_airport_train.csv`.

- **`BE_optmisation/t_max.py`**
	- Script d'analyse/visualisation de la fonction de contrainte temporelle \(T_{max}\).
	- Permet de visualiser l'allongement de temps autorisé selon la durée de référence.

## 4) Données d'entrée

### Données versionnées dans le dépôt

- **`BE_optmisation/train_lines.csv`**
	- Liaisons ferroviaires avec colonnes :
		- `ville_dep`, `ville_arr`, `iuc_dep`, `iuc_arr`, `duree_min`.

- **`BE_optmisation/train_cities.csv`**
	- Référentiel de villes ferroviaires (`Villes`) + code UIC (`code_iuc`).

- **`BE_optmisation/data/ac_model_coefficients.csv`**
- **`BE_optmisation/data/ac_model_coefficients_airliners.csv`**
	- Coefficients FEAT par type avion (`ac_code_icao`, `reduced_fuel_a1`, `reduced_fuel_a2`, `reduced_fuel_intercept`, etc.).

### Données attendues mais non versionnées

Le fichier de vols brut utilisé dans plusieurs scripts est attendu sous :

- `BE_optmisation/data/Flights_20191201_20191231.csv`

Le dossier `data/` et les fichiers `.csv` sont ignorés par `BE_optmisation/.gitignore`, donc plusieurs jeux de données générés/intermédiaires ne sont pas committés.

## 5) Variables et logique d'optimisation

Le modèle PuLP repose sur :

- variable binaire `x(i,j,m)` : l'arc de `i` vers `j` est utilisé en mode `m` (`train` ou `avion`) ;
- variable binaire `y(i,m,k)` : transfert effectué à `i` entre modes `m` et `k` ;
- objectif : minimiser la somme des émissions sur les arcs choisis (avec parfois un très faible tie-breaker sur le temps) ;
- contraintes :
	- conservation de flux (origine, destination, villes intermédiaires) ;
	- cohérence transfert (`x` ↔ `y`) ;
	- contrainte de temps globale `temps_total <= T_max`.

Le temps total combine :

- durée des arcs,
- temps d'accès par mode (`train` / `avion`),
- pénalité fixe par transfert.

## 6) Pré-requis

- Python 3.10+ recommandé
- Packages :
	- `pandas`
	- `numpy`
	- `pulp`
	- `matplotlib` (pour `t_max.py`)

Installation rapide :

```bash
pip install pandas numpy pulp matplotlib
```

## 7) Exécution (ordre recommandé)

Depuis `BE_optmisation/`.

### Étape A — Générer les arcs train/avion pré-traités

Le dépôt ne fournit pas de script CLI unique pour cette étape ; il faut appeler les fonctions Python puis sauvegarder les CSV :

1. Charger `train_lines.csv` et appliquer `train_co2_emissions.treatment(...)`.
2. Charger les vols bruts et appliquer `function_data.treatment(...)`.
3. Sauver respectivement :
	 - `trains_arcs_processed.csv`
	 - `planes_arcs_processed.csv`

### Étape B — Optimiser un trajet unique

```bash
python optim_pulp_trajet_unique.py
```

Le script demande une ville de départ et d'arrivée, puis affiche l'alternative multimodale la moins émettrice sous contrainte de temps.

### Étape C — Lancer une analyse batch

```bash
python optim_pulp_all_liaisons_plane.py
```

ou

```bash
python optim_pulp_all_liaisons_plane_gare.py
```

## 8) Fichiers de sortie (générés)

Selon le script exécuté :

- `optimisation_vols_existants.csv`
- `bilan_airport_train.csv`
- (éventuellement) `trains_arcs_processed.csv`, `planes_arcs_processed.csv`

Ces fichiers sont généralement non versionnés.

## 9) Limites et points d'attention

- Le cœur du workflow dépend d'un fichier vols brut absent du dépôt.
- Les paramètres de contrainte temporelle (`A`, `B`, `C`) varient selon les scripts : les résultats ne sont donc pas strictement comparables entre toutes les variantes.
- Les temps d'accès (`T_access`) et temps de transfert (`T_transf`) sont des hypothèses fixes.
- Certains scripts sont orientés expérimentation et non packaging production (pas de CLI unifiée, pas de tests automatisés fournis).

## 10) Résumé rapide

Ce dépôt sert à **quantifier et optimiser des alternatives train/avion** à l'échelle de liaisons européennes, en minimisant le CO₂ sous contrainte de temps. La logique est suffisamment modulaire pour tester différents paramètres de service, filtrages de vols et scénarios batch réseau.
