# ph_diagnostics.py
import argparse, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # pas d'affichage, on sauve des PNG
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

def km_like_S_at(df, t_star):
    d = df[["time","event"]].sort_values("time")
    at = len(d); S = 1.0
    for tt, g in d.groupby("time"):
        if tt > t_star: break
        dcount = g["event"].sum()
        S *= (1.0 - dcount / max(1, at))
        at -= len(g)
    return float(S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_cox_train.csv", help="Jeu de données (time,event,features...)")
    ap.add_argument("--penalizer", type=float, default=0.0, help="Ridge L2 pour le fit lifelines")
    ap.add_argument("--time_transform", default="log", choices=["log","rank","identity"],
                    help="Transformation du temps pour le test de Schoenfeld (log conseillé)")
    ap.add_argument("--plots", action="store_true", help="Sauver des graphes des résidus de Schoenfeld")
    ap.add_argument("--top_plots", type=int, default=12, help="Nombre max de variables à tracer")
    args = ap.parse_args()

    outdir = "ph_outputs"
    os.makedirs(outdir, exist_ok=True)

    # 1) Charger les données
    df = pd.read_csv(args.csv)
    if not {"time","event"}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes 'time' et 'event'.")
    feature_cols = [c for c in df.columns if c not in ("time","event")]
    print(f"Chargé: {args.csv} | n={len(df)} | p={len(feature_cols)} features")

    # 2) Fit Cox (lifelines) sur l'échantillon complet fourni
    cph = CoxPHFitter(penalizer=args.penalizer)
    cph.fit(df[["time","event"] + feature_cols], duration_col="time", event_col="event", show_progress=False)
    print("Fit lifelines OK.")
    cph.summary.to_csv(os.path.join(outdir, "lifelines_summary.csv"))
    print("-> Résumé des coef (lifelines_summary.csv)")

    # 3) TEST PH — Résidus de Schoenfeld (global + par variable)
    print("\n[PH TEST] Résidus de Schoenfeld (time_transform=%s)..." % args.time_transform)
    ph_test = proportional_hazard_test(cph, df, time_transform=args.time_transform)
    sch = ph_test.summary  # colonnes: test_statistic, p, -log2(p)
    sch = sch.rename_axis("variable").reset_index()
    sch.to_csv(os.path.join(outdir, "schoenfeld_test.csv"), index=False)
    print(sch[["variable","p"]].sort_values("p").head(10))
    # LIGNE GLOBALE ?
    # lifelines met souvent une ligne 'global' pour le test global si appelée via cph.check_assumptions ;
    # ici, on se concentre sur les p-val par variable.

    # 4) INTERACTIONS x*log(time) — une par une
    print("\n[PH TEST] Interactions x*log(time) (une par variable) ...")
    eps = 1e-9
    logt = np.log(df["time"].values + eps)
    rows = []
    for col in feature_cols:
        df_tmp = df.copy()
        inter_col = f"{col}_x_logt"
        df_tmp[inter_col] = df_tmp[col].astype(float) * logt
        # fit avec base + interaction
        cph2 = CoxPHFitter(penalizer=args.penalizer)
        cph2.fit(df_tmp[["time","event"] + feature_cols + [inter_col]],
                 duration_col="time", event_col="event", show_progress=False)
        pval = cph2.summary.loc[inter_col, "p"]
        rows.append({"variable": col, "p_interaction": pval})
    inter_df = pd.DataFrame(rows).sort_values("p_interaction")
    inter_df.to_csv(os.path.join(outdir, "interaction_logt_test.csv"), index=False)
    print(inter_df.head(10))

    # 5) Inspection visuelle — tracés des résidus de Schoenfeld lissés
    if args.plots:
        print("\n[PH PLOTS] Tracé des résidus de Schoenfeld (avec lissage) ...")
        # lifelines fournit les résidus de Schoenfeld
        resid = cph.compute_residuals(df, kind="schoenfeld")   # index = obs avec event, colonnes = vars
        # temps associés aux résidus (pour events uniquement)
        event_times = df.loc[df["event"]==1, "time"].values
        # On lisse avec une moyenne glissante simple (fenêtre ~10% des events)
        m = len(resid)
        win = max(5, int(0.1 * m))
        win = win if win % 2 == 1 else win+1  # fenêtre impaire

        # Sélectionner les variables "les plus suspectes" (p petites) pour tracer
        vars_to_plot = list(sch.sort_values("p")["variable"])
        # retirer la variable 'global' si elle apparait
        vars_to_plot = [v for v in vars_to_plot if v in resid.columns][:args.top_plots]

        for var in vars_to_plot:
            y = resid[var].values
            # tri par temps
            idx = np.argsort(event_times)
            t_sorted = event_times[idx]
            y_sorted = y[idx]
            # lissage glissant
            y_series = pd.Series(y_sorted)
            y_smooth = y_series.rolling(win, center=True, min_periods=max(3, win//3)).mean().values

            plt.figure(figsize=(6,3))
            plt.scatter(t_sorted, y_sorted, s=4, alpha=0.3)
            plt.plot(t_sorted, y_smooth, linewidth=2)
            plt.axhline(0, color="k", linewidth=1)
            plt.title(f"Schoenfeld residus vs temps — {var}")
            plt.xlabel("temps"); plt.ylabel("résidu (centré)")
            plt.tight_layout()
            outpng = os.path.join(outdir, f"schoenfeld_{var}.png")
            plt.savefig(outpng, dpi=150)
            plt.close()
        print(f"-> PNG enregistrés dans {outdir}/ (jusqu'à {args.top_plots} variables)")

    # 6) Récap & seuils de lecture
    print("\n=== RÉCAP ===")
    viol_schoenfeld = sch[sch["p"] < 0.05][["variable","p"]]
    viol_inter = inter_df[inter_df["p_interaction"] < 0.05]
    print(f"Variables suspectes (Schoenfeld p<0.05): {len(viol_schoenfeld)}")
    print(viol_schoenfeld.to_string(index=False) if len(viol_schoenfeld) else "Aucune")
    print(f"Variables suspectes (interaction x log(time) p<0.05): {len(viol_inter)}")
    print(viol_inter.head(20).to_string(index=False) if len(viol_inter) else "Aucune")
    print(f"\nSorties écrites dans: {outdir}/")
    print(" - schoenfeld_test.csv           (p-values par variable)")
    print(" - interaction_logt_test.csv     (p-values de l'interaction par variable)")
    print(" - lifelines_summary.csv         (coef, HR, IC, p)")
    if args.plots:
        print(" - schoenfeld_<var>.png          (résidus lissés vs temps)")

if __name__ == "__main__":
    main()
