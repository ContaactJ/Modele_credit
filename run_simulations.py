
import argparse
import pickle
import numpy as np
import pandas as pd

def load_model(pkl_path="cox_model.pkl"):
    with open(pkl_path, "rb") as f:
        pack = pickle.load(f)
    return pack["model"], pack.get("feature_names", None)

def invert_sample_times(hr, H0_df, nsim, rng):
    H0_times = H0_df["time"].values
    H0_vals = H0_df["H0"].values
    if len(H0_vals) == 0:
        raise RuntimeError("Baseline H0 vide.")
    T_samples = np.empty(nsim, dtype=float)
    for k in range(nsim):
        u = rng.uniform()
        thresh = -np.log(u) / hr
        idx = np.searchsorted(H0_vals, thresh, side="left")
        if idx >= len(H0_vals):
            T_samples[k] = H0_times[-1]
        else:
            T_samples[k] = H0_times[idx]
    return T_samples

def main():
    parser = argparse.ArgumentParser(description="Charger un modèle CoxPH calibré et lancer des simulations.")
    parser.add_argument("--model", default="cox_model.pkl", help="Modèle picklé")
    parser.add_argument("--scenarios", default=None, help="CSV avec colonnes features")
    parser.add_argument("--random_n", type=int, default=0, help="nb de scénarios aléatoires si pas de CSV")
    parser.add_argument("--p", type=int, default=None, help="dimension si aléatoire et pas de noms stockés")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsim", type=int, default=0, help="tirages de temps par scénario (0 = pas de tirage)")
    parser.add_argument("--times", default=None, help="CSV (col 'time') OU liste 't1,t2,...'")
    args = parser.parse_args()

    model, feature_names = load_model(args.model)
    rng = np.random.default_rng(args.seed)

    if args.scenarios is not None:
        dfX = pd.read_csv(args.scenarios)
        X_new = dfX.to_numpy(float)
        scenario_names = list(dfX.index.astype(str))
        if feature_names is not None:
            missing = set(feature_names) - set(dfX.columns)
            if missing:
                raise ValueError(f"Colonnes manquantes dans {args.scenarios}: {missing}")
        used_cols = feature_names if feature_names else dfX.columns.tolist()
    else:
        if args.random_n <= 0:
            raise ValueError("Spécifie --scenarios CSV ou --random_n > 0.")
        if feature_names is None and args.p is None:
            raise ValueError("Précise --p si le modèle ne stocke pas les noms de variables.")
        p = args.p if args.p is not None else len(feature_names)
        X_new = rng.normal(size=(args.random_n, p))
        scenario_names = [f"rnd_{i}" for i in range(args.random_n)]
        used_cols = feature_names if feature_names else [f"x{j+1}" for j in range(p)]
        dfX = pd.DataFrame(X_new, columns=used_cols)

    if args.times is None:
        times = model.baseline_cumhaz_["time"].values
    else:
        try:
            tdf = pd.read_csv(args.times)
            times = tdf["time"].values
        except Exception:
            times = np.array([float(t) for t in args.times.split(",")], dtype=float)

    S_df = model.predict_survival_function(X_new, times=times)
    S_df.columns = scenario_names
    S_df.to_csv("survival_curves.csv")
    print("Courbes de survie enregistrées: survival_curves.csv")

    if args.nsim > 0:
        hr = model.predict_partial_hazard(X_new)
        all_rows = []
        for j, name in enumerate(scenario_names):
            T_samples = invert_sample_times(hr[j], model.baseline_cumhaz_, args.nsim, rng)
            for k, val in enumerate(T_samples):
                all_rows.append((name, k, val))
        sim_df = pd.DataFrame(all_rows, columns=["scenario", "draw", "time"])
        sim_df.to_csv("simulated_times.csv", index=False)
        print("Simulations de temps enregistrées: simulated_times.csv")

    dfX.to_csv("scenarios_used.csv", index=False)
    print("Scénarios utilisés enregistrés: scenarios_used.csv")

if __name__ == "__main__":
    main()
