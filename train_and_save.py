
import argparse
import pickle
import numpy as np
import pandas as pd
from coxph import CoxPH

def main():
    parser = argparse.ArgumentParser(description="Calibrer un modèle CoxPH et sauvegarder les artefacts.")
    parser.add_argument("--csv", default="data.csv", help="CSV avec colonnes: time,event,x1, ...")
    parser.add_argument("--l2", type=float, default=0.0, help="Ridge L2")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
        print(f"Chargé: {args.csv} ({len(df)} lignes)")
        T = df["time"].to_numpy(float)
        E = df["event"].to_numpy(int)
        X = df.drop(columns=["time", "event"]).to_numpy(float)
        feature_names = [c for c in df.columns if c not in ("time", "event")]
    except FileNotFoundError:
        print(f"{args.csv} introuvable. Génération d'un dataset simulé -> data_simulated.csv")
        rng = np.random.default_rng(42)
        n, p = 600, 3
        X = rng.normal(size=(n, p))
        beta_true = np.array([0.7, -0.5, 0.0])
        lambda0 = 0.02
        U = rng.uniform(size=n)
        T_true = -np.log(U) / (lambda0 * np.exp(X @ beta_true))
        C = rng.exponential(scale=80, size=n)
        T = np.minimum(T_true, C)
        E = (T_true <= C).astype(int)
        feature_names = [f"x{j+1}" for j in range(X.shape[1])]
        df = pd.DataFrame(np.column_stack([T, E, X]), columns=["time", "event"] + feature_names)
        df.to_csv("data_simulated.csv", index=False)
        print("Dataset simulé écrit : data_simulated.csv")

    model = CoxPH(l2=args.l2, max_iter=args.max_iter, tol=args.tol, verbose=False).fit(X, T, E)

    with open("cox_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    model.baseline_cumhaz_.to_csv("baseline_h0.csv", index=False)
    pd.Series(model.beta_, index=feature_names, name="beta").to_csv("coefficients.csv")

    print("\n=== RÉSUMÉ ===")
    print(pd.Series(model.beta_, index=feature_names))
    print("\nFichiers créés: cox_model.pkl, baseline_h0.csv, coefficients.csv")

if __name__ == "__main__":
    main()
