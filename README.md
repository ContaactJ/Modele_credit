# Cox Credit Risk Model (Cox Proportional Hazards)

Modèle de **Cox PH** pour estimer le risque de défaut (particuliers) avec :
- implémentation maison (Newton + line search, Breslow),
- scripts d’entraînement, d’évaluation (discrimination & calibration),
- diagnostics de l’hypothèse de **risques proportionnels** (résidus de Schoenfeld),
- simulations de temps d’événement à partir d’un modèle calibré.

## Sommaire
- [Installation](#installation)
- [Structure du dépôt](#structure-du-dépôt)
- [Format des données](#format-des-données)
- [Entraînement](#entraînement)
- [Évaluation (discrimination & calibration)](#évaluation-discrimination--calibration)
- [Diagnostics PH (Schoenfeld)](#diagnostics-ph-schoenfeld)
- [Simulations](#simulations)
- [Reproductibilité (hash dataset)](#reproductibilité-hash-dataset)
- [Artefacts de modèle](#artefacts-de-modèle)
- [Critères de validation](#critères-de-validation)
- [Dépannage rapide](#dépannage-rapide)
- [Licence](#licence)

---

## Installation

```bash
# Créer un environnement virtuel
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1

# Dépendances
pip install -r requirements.txt
```

> Recommandé : Python 3.10+.

---

## Structure du dépôt

```
.
├─ coxph.py                     # classe CoxPH (fit/predict, baseline H0)
├─ train_and_save.py            # entraînement + sauvegarde artefacts
├─ eval_cox.py                  # c-index + calibration déciles à T*
├─ ph_diagnostics.py            # tests PH (Schoenfeld), interactions x·log(time)
├─ run_simulations.py           # simulations de temps d'événement
├─ split_data_cox.py            # split train/valid/test (si besoin)
├─ baseline_h0.csv              # H0(t) cumulée (fonction en escalier)
├─ coefficients.csv             # coefficients β (lisible/audit)
├─ feature_names.txt            # ordre exact des colonnes de features
├─ .gitignore
├─ README.md
└─ requirements.txt
```

> Les jeux de **données réels** et les **modèles binaires** (`*.pkl`) ne sont pas versionnés (voir `.gitignore`).  
> Utiliser `data/samples/` pour des échantillons anonymisés si besoin.

---

## Format des données

CSV avec colonnes :

- `time` : temps (mois) jusqu’à événement ou censure (float)
- `event` : 1 = événement (défaut), 0 = censuré
- **features** : uniquement numériques (catégorielles → one-hot/dummies)

Exemple (tête de colonnes) :
```
time,event,age,sexe_H,dti,ltv,employment_status_CDI,employment_status_Indep,...
```

---

## Entraînement

Entraîne un Cox sur un CSV et sauvegarde le modèle + artefacts :

```bash
python train_and_save.py --csv data_cox_train.csv
```

**Sorties créées :**
- `cox_model.pkl` : modèle complet (β, H0, standardisation, features)
- `baseline_h0.csv` : H0(t) cumulée (time,H0)
- `coefficients.csv` : coefficients β
- `feature_names.txt` : ordre des features attendu

> **Note** : `cox_model.pkl` suffit pour prédire. Les CSV sont utiles pour audit/portabilité.

---

## Évaluation (discrimination & calibration)

Mesure **c-index** et **calibration** à un horizon \(T^*\) (ex. 36 mois) sur **validation/test**.

```bash
# Validation (T*=36)
python eval_cox.py cox_model.pkl data_cox_valid.csv 36

# Test (T*=36)
python eval_cox.py cox_model.pkl data_cox_test.csv 36
```

**Lecture recommandée :**
- **c-index** ≥ 0,65 (≥ 0,70 : bon)
- **Calibration** par déciles : MAE ≤ 2 points, erreur max ≤ 5 points

> Tu peux répéter pour 12 et 24 mois en changeant l’argument final.

---

## Diagnostics PH (Schoenfeld)

Vérifie l’hypothèse de **risques proportionnels** :
- Test de **Schoenfeld** (p-values par variable)
- Interactions **x·log(time)** (check simple)
- **Graphiques** des résidus (option `--plots`)

```bash
# p-values + CSV de résultats
python ph_diagnostics.py --csv data_cox_train.csv

# Avec graphes (PNG dans ph_outputs/)
python ph_diagnostics.py --csv data_cox_train.csv --plots
```

Interprétation rapide :
- p > 0,05 → pas d’évidence de violation PH
- p < 0,05 → suspicion (effet dépend du temps) → envisager `x·log(time)` ou stratification

---

## Simulations

Génère des **temps simulés** à partir du modèle calibré (inversion de H0) :

```bash
# 50 profils aléatoires, 500 tirages chacun
python run_simulations.py --model cox_model.pkl --random_n 50 --nsim 500

# …ou sur des scénarios fournis (mêmes colonnes que feature_names.txt)
python run_simulations.py --model cox_model.pkl --scenarios scenarios.csv --nsim 200
```

**Sorties :**
- `survival_curves.csv` : S(t) par scénario
- `simulated_times.csv` : tirages de temps d’événement
- `scenarios_used.csv` : X utilisés

> Pour censurer à 36 mois : `time_cens = min(time, 36)`, `event = 1{time ≤ 36}`.

---

## Reproductibilité (hash dataset)

Figer la base d’apprentissage avec un **hash SHA-256** (traçabilité) :

- **PowerShell** :
  ```powershell
  Get-FileHash .\data_cox.csv -Algorithm SHA256 | Format-List
  ```
- Sauvegarder le hash dans `data_cox.sha256.txt` (non versionné par défaut).

---

## Artefacts de modèle

- **`cox_model.pkl`** : objet Python sérialisé (β, H0(t), standardisation, features).  
- **`baseline_h0.csv`** : H0(t) (une ligne par **temps d’événement distinct**).
- **`coefficients.csv`** : β (lisible par un humain).
- **`feature_names.txt`** : ordre attendu des colonnes pour le scoring.

> En production, le **pickle seul** suffit pour scorer. Les CSV servent à l’audit et à la portabilité (hors Python).

---

## Critères de validation

- **Discrimination** : c-index ≥ **0,65** (≥0,70 : bon).
- **Calibration à T*** : MAE ≤ **0,02**, max ≤ **0,05** par déciles.
- **PH** : pas de violation majeure (Schoenfeld p>0,05 ; visuels ~horizontaux).  
- **Stabilité** : résultats cohérents à 12/24/36 mois.

---

## Dépannage rapide

- **“ModuleNotFoundError: coxph”**  
  Lancer depuis le dossier où se trouve `coxph.py` ou ajuster `PYTHONPATH`.

- **“ValueError: could not convert string to float”**  
  Une feature non numérique traîne → encoder les catégorielles (dummies) avant le fit.

- **Fichiers ignorés n’apparaissent pas**  
  Vérifier `.gitignore`. Si déjà suivis : `git rm --cached chemin/du/fichier` puis commit.



