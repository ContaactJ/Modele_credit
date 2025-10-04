
"""
make_survival_from_assures.py
-----------------------------
Convertit 'assures_synthetiques_100k.csv' (généré précédemment) en un dataset pour modèle de Cox:

- Crée un temps-à-événement en mois (T) par simulation inverse avec une hazard base exponentielle.
- Censure administrative à 36 mois (modifiable).
- L'événement 'default' de la base est utilisé pour calibrer la baseline pour ~2% de défaut à 36 mois *en moyenne*,
  mais surtout on crée une dépendance aux covariables (risque proportionnel) via un prédicteur linéaire s(x).
- Produit un CSV 'data_cox.csv' avec colonnes: time, event, <variables numériques> (one-hot pour les catégorielles).

Utilisation:
    python make_survival_from_assures.py --input assures_synthetiques_100k.csv --horizon 36 --seed 42

Ensuite:
    python train_and_save.py --csv data_cox.csv
"""
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="assures_synthetiques_100k.csv")
    ap.add_argument("--horizon", type=int, default=36, help="Horizon de censure administrative (mois)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n = len(df)
    rng = np.random.default_rng(args.seed)

    # Encodages d'effets (cohérents avec la génération)
    income_effects = {
        "<20k":  0.70, "20-35k": 0.45, "35-50k": 0.20, "50-80k": 0.00, "80-120k": -0.15, "120k+":  -0.35,
    }
    wealth_effects = {
        "<10k":   0.35, "10-50k": 0.15, "50-150k": 0.00, "150-300k": -0.15, "300k+":   -0.30,
    }
    employment_effects = {
        "CDI": 0.00, "CDD": 0.35, "Indep": 0.25, "Chomage": 0.90, "Retraite": 0.10,
    }
    purpose_effects = {
        "Habitation": 0.00, "Auto": 0.05, "Conso": 0.30, "Rachat": -0.10, "Travaux": 0.10,
    }
    sex_effects = {"H": 0.02, "F": 0.00}

    rng2 = np.random.default_rng(123)
    geo_random = {z: g for z, g in zip(range(1, 11), rng2.normal(0.0, 0.05, size=10))}

    # Prédicteur linéaire s(x) (centrages sur les moyennes du dataset)
    age = df["age"].to_numpy()
    dti = df["dti"].to_numpy()
    ltv = df["ltv"].to_numpy()
    credit_hist = df["credit_history_years"].to_numpy()
    prev_lates = df["prev_lates"].to_numpy()
    has_coapp = df["has_coapplicant"].to_numpy()

    s = np.zeros(n, dtype=float)
    s += df["income_band"].map(income_effects).to_numpy()
    s += df["wealth_band"].map(wealth_effects).to_numpy()
    s += df["employment_status"].map(employment_effects).to_numpy()
    s += df["loan_purpose"].map(purpose_effects).to_numpy()
    s += df["sexe"].map(sex_effects).to_numpy()
    s += df["geo_zone"].map(geo_random).to_numpy()

    s += 3.0 * (dti - dti.mean())
    s += 1.8 * (ltv - ltv.mean())
    s += -0.05 * (credit_hist - credit_hist.mean())
    s += 0.20 * (prev_lates - prev_lates.mean())
    s += -0.20 * (has_coapp - has_coapp.mean())
    s += 0.0015 * (np.abs(age - 40) - np.abs(age - 40).mean())

    hr = np.exp(s)  # hazard ratio relatif

    # Calibrer lambda0 pour une incidence moyenne proche de 2% à 'horizon' mois:
    # E[ 1 - exp(-lambda0 * H * HR) ] ≈ 0.02  -> recherche par dichotomie
    target = 0.02
    H = args.horizon
    low, high = 1e-6, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        mean_def = (1.0 - np.exp(-mid * H * hr)).mean()
        if mean_def > target:
            high = mid
        else:
            low = mid
    lambda0 = 0.5 * (low + high)

    # Échantillonner les temps d'événement (exponentiel avec taux = lambda0 * hr)
    U = rng.uniform(size=n)
    T_event = -np.log(U) / (lambda0 * hr)

    # Censure administrative à H mois
    T = np.minimum(T_event, H)
    E = (T_event <= H).astype(int)

    # Construire X numérique (one-hot des catégorielles)
    cat_cols = ["sexe", "income_band", "wealth_band", "employment_status", "loan_purpose", "geo_zone"]
    num_cols = ["age", "ltv", "dti", "credit_history_years", "prev_lates", "has_coapplicant"]
    X = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
    X = pd.concat([X, df[num_cols].astype(float)], axis=1)

    out = pd.concat([pd.DataFrame({"time": T, "event": E}), X], axis=1)
    out_path = "data_cox.csv"
    out.to_csv(out_path, index=False)

    # Petit résumé
    print(f"lambda0 calibré: {lambda0:.6f}  | horizon: {H} mois")
    print("Taux de défaut simulé à l'horizon:", E.mean())
    print("T (quantiles):", np.quantile(T, [0.1, 0.5, 0.9]).round(2))
    print(f"Fichier écrit: {out_path}")
    # Export liste des colonnes (utile pour scénarios)
    with open("feature_names.txt", "w", encoding="utf-8") as f:
        for c in X.columns:
            f.write(c + "\n")
    print("Colonnes features enregistrées dans feature_names.txt")

if __name__ == "__main__":
    main()
