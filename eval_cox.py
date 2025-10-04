# eval_cox.py
import sys, pickle, numpy as np, pandas as pd

MODEL_PKL = sys.argv[1]            # ex: cox_model.pkl
CSV_PATH  = sys.argv[2]            # ex: data_cox_valid.csv
TSTAR     = float(sys.argv[3]) if len(sys.argv)>3 else 36.0

# --- 1) Charger modèle + données ---
with open(MODEL_PKL, "rb") as f:
    pack = pickle.load(f)
model = pack["model"]
feature_names = pack.get("feature_names")

df = pd.read_csv(CSV_PATH)
time_col, event_col = "time", "event"
if feature_names is None:
    X = df.drop(columns=[time_col, event_col]).to_numpy(float)
    feature_names = list(df.drop(columns=[time_col, event_col]).columns)
else:
    X = df[feature_names].to_numpy(float)
t = df[time_col].to_numpy(float)
e = df[event_col].to_numpy(int)

# --- 2) Scores (hazard ratios) ---
hr = model.predict_partial_hazard(X)  # exp(xβ)

# --- 3) c-index (Harrell) ---
def c_index(time, event, score):
    order = np.argsort(time)
    time, event, score = time[order], event[order], score[order]
    n_pairs = n_conc = n_ties = 0
    for i in range(len(time)):
        if event[i]==0: 
            continue
        # comparer i avec tous j ayant time>time[i]
        j_idx = np.where(time[i] < time[i+1:])[0] + (i+1)
        n_pairs += len(j_idx)
        if len(j_idx)==0: 
            continue
        s_comp = score[j_idx]
        n_conc += np.sum(score[i] > s_comp)
        n_ties += np.sum(score[i] == s_comp)
    return (n_conc + 0.5*n_ties) / max(1, n_pairs)

cidx = c_index(t, e, hr)

# --- 4) Calibration à T* : S_pred vs S_obs (KM) par déciles ---
H0 = model.baseline_cumhaz_
H0_T = np.interp(TSTAR, H0["time"].values, H0["H0"].values, left=0.0, right=H0["H0"].values[-1])
S_pred = np.exp(-H0_T * hr)  # survie prédite à T*

valid = df.copy()
valid["S_pred"] = S_pred
# déciles de risque (1-S_pred)
valid["decile"] = pd.qcut(1 - valid["S_pred"], 10, labels=False, duplicates="drop")

# Kaplan-Meier (survie observée à T*)
def km_S_at(time, event, t_star):
    # estimateur KM discret minimal
    dt = pd.DataFrame({"t": time, "e": event}).sort_values("t")
    at_risk = len(dt)
    S = 1.0
    for tt, g in dt.groupby("t"):
        d = g["e"].sum()
        if at_risk>0 and tt <= t_star:
            S *= (1.0 - d/at_risk)
            at_risk -= len(g)
        else:
            break
    return float(S)

rows = []
for d in sorted(valid["decile"].dropna().unique()):
    g = valid[valid["decile"]==d]
    S_obs = km_S_at(g[time_col].values, g[event_col].values, TSTAR)
    S_bar = float(g["S_pred"].mean())
    rows.append({"decile": int(d), "n": len(g), f"S_obs({int(TSTAR)}m)": S_obs, f"S_pred({int(TSTAR)}m)": S_bar,
                 "abs_err": abs(S_obs - S_bar)})
calib = pd.DataFrame(rows).sort_values("decile")
mae = float(calib["abs_err"].mean())
mxe = float(calib["abs_err"].max())

# --- 5) Sorties & verdict ---
print(f"\n=== EVALUATION on {CSV_PATH} at T*={int(TSTAR)} months ===")
print(f"C-index: {cidx:.3f}")
print("\nCalibration by decile:")
print(calib.drop(columns=['abs_err']).to_string(index=False))
print(f"\nCalibration MAE: {mae:.4f}   Max error: {mxe:.4f}")

# Règle simple de verdict
disc_ok = cidx >= 0.65
cal_ok  = (mae <= 0.02) and (mxe <= 0.05)
print("\nVERDICT:",
      "OK (discrimination & calibration)" if (disc_ok and cal_ok)
      else "À améliorer (voir seuils: c-index>=0.65, MAE<=0.02, max<=0.05)")
