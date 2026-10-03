
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

# Assumptions the dataset cannot supply. Stated here so they are easy to challenge.
LGD = 0.75   # loss given default for unsecured card debt; sensitivity run below
CCF = 0.50   # credit conversion factor: share of the unused limit drawn before default


def load():
    df = pd.read_csv(os.path.join(ROOT, "data", "UCI_Credit_Card.csv"))
    df = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    return df


def features(df):
    x = df.copy()
    bills = x[[f"BILL_AMT{i}" for i in range(1, 7)]]
    pays = x[[f"PAY_AMT{i}" for i in range(1, 7)]]
    x["utilisation"] = (x["BILL_AMT1"] / x["LIMIT_BAL"]).clip(-1, 3)
    x["months_late_6m"] = (x[[f"PAY_{i}" for i in range(1, 7)]] > 0).sum(axis=1)
    x["worst_status_6m"] = x[[f"PAY_{i}" for i in range(1, 7)]].max(axis=1)
    x["paid_share_6m"] = (pays.sum(axis=1) / bills.clip(lower=0).sum(axis=1).replace(0, np.nan)).clip(0, 5)
    return x


# SEX and MARRIAGE are excluded on purpose: US fair lending law (ECOA) prohibits using them
# in credit decisions, so a model meant to resemble a real scorecard should not lean on them.
CAT = ["EDUCATION"]
NUM = ["LIMIT_BAL", "AGE"] + [f"PAY_{i}" for i in range(1, 7)] + \
      [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] + \
      ["utilisation", "months_late_6m", "worst_status_6m", "paid_share_6m"]


def logistic():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def gbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=SEED, verbose=-1)


def scores(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    return {"accuracy": round(accuracy_score(y, yhat), 4),
            "roc_auc": round(roc_auc_score(y, p), 4) if len(np.unique(p)) > 1 else 0.5,
            "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "precision_at_0.5": round(precision_score(y, yhat, zero_division=0), 4),
            "recall_at_0.5": round(recall_score(y, yhat), 4)}


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    df = features(load())
    tr, te = train_test_split(df, test_size=0.2, stratify=df["default"], random_state=SEED)
    y_tr, y_te = tr["default"].to_numpy(), te["default"].to_numpy()
    m = {"n_accounts": len(df), "default_rate": round(float(df["default"].mean()), 4),
         "n_test": len(te), "assumptions": {"LGD": LGD, "CCF": CCF}}

    # ---- baselines
    base_rate = y_tr.mean()
    p_naive = np.full(len(te), base_rate)
    one_var = make_pipeline(StandardScaler(), LogisticRegression()).fit(tr[["PAY_1"]], y_tr)
    p_one = one_var.predict_proba(te[["PAY_1"]])[:, 1]

    # ---- models, with 5-fold CV AUC on training data for selection
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X_tr, X_te = tr[NUM + CAT], te[NUM + CAT]
    lr = logistic()
    cv_lr = cross_val_score(lr, X_tr, y_tr, cv=cv, scoring="roc_auc")
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    g = gbm()
    cv_g = cross_val_score(g, X_tr, y_tr, cv=cv, scoring="roc_auc")
    g.fit(X_tr, y_tr)
    p_g = g.predict_proba(X_te)[:, 1]

    m["holdout"] = {
        "always_no_default": scores(y_te, np.zeros(len(te))) | {"roc_auc": 0.5},
        "last_month_status_only": scores(y_te, p_one),
        "logistic_regression": scores(y_te, p_lr) | {"cv_auc_mean": round(cv_lr.mean(), 4),
                                                      "cv_auc_sd": round(cv_lr.std(), 4)},
        "lightgbm": scores(y_te, p_g) | {"cv_auc_mean": round(cv_g.mean(), 4),
                                         "cv_auc_sd": round(cv_g.std(), 4)},
    }

    # ---- expected loss: EL = PD x LGD x EAD
    drawn = te["BILL_AMT1"].clip(lower=0)
    undrawn = (te["LIMIT_BAL"] - drawn).clip(lower=0)
    ead = (drawn + CCF * undrawn).to_numpy()
    realised = y_te * LGD * ead                      # loss that actually occurred, same LGD/EAD
    el = {}
    for name, p in [("flat_pd", p_naive), ("logistic_regression", p_lr), ("lightgbm", p_g)]:
        pred = p * LGD * ead
        el[name] = {"predicted_el_ntd": round(float(pred.sum()), 0),
                    "pred_vs_realised": round(float(pred.sum() / realised.sum()), 3)}
    best = p_g
    te_el = te.assign(pd_=best, ead=ead, el=best * LGD * ead, realised=realised)
    te_el["decile"] = pd.qcut(te_el["pd_"].rank(method="first"), 10, labels=range(1, 11))
    dec = te_el.groupby("decile", observed=True).agg(accounts=("pd_", "size"), mean_pd=("pd_", "mean"),
                                                     default_rate=("default", "mean"),
                                                     predicted_el=("el", "sum"), realised_loss=("realised", "sum"))
    dec["share_of_realised_loss"] = dec["realised_loss"] / dec["realised_loss"].sum()
    dec.round(4).to_csv(os.path.join(ROOT, "results", "el_by_decile.csv"))
    top2 = float(dec.loc[[9, 10], "share_of_realised_loss"].sum())
    el["realised_loss_ntd"] = round(float(realised.sum()), 0)
    el["portfolio_ead_ntd"] = round(float(ead.sum()), 0)
    el["el_rate_of_ead"] = round(float(te_el["el"].sum() / ead.sum()), 4)
    el["riskiest_20pct_share_of_loss"] = round(top2, 3)
    el["top_decile_default_rate"] = round(float(dec.loc[10, "default_rate"]), 3)
    el["bottom_decile_default_rate"] = round(float(dec.loc[1, "default_rate"]), 3)
    el["lgd_sensitivity_el_ntd"] = {str(l): round(float((best * l * ead).sum()), 0) for l in [0.45, 0.60, 0.75, 0.90]}
    m["expected_loss"] = el

    with open(os.path.join(ROOT, "results", "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, p in [("Last month's status only", p_one), ("Logistic regression", p_lr), ("LightGBM", p_g)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], color="grey", lw=1, label="Always no default (AUC 0.500)")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC, holdout"); ax[0].legend(fontsize=8)
    for name, p in [("Logistic regression", p_lr), ("LightGBM", p_g)]:
        fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax[1].plot(mp, fr, "o-", label=name)
    ax[1].plot([0, 1], [0, 1], color="grey", lw=1)
    ax[1].set_xlabel("Predicted PD"); ax[1].set_ylabel("Observed default rate")
    ax[1].set_title("Calibration, holdout"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "roc_calibration.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.arange(1, 11)
    ax.bar(xs - 0.2, dec["predicted_el"] / 1e6, 0.4, label="Predicted EL")
    ax.bar(xs + 0.2, dec["realised_loss"] / 1e6, 0.4, label="Realised loss")
    ax.set_xticks(xs); ax.set_xlabel("PD decile (1 = safest)"); ax.set_ylabel("NT$ millions")
    ax.set_title("Expected vs realised loss by risk decile, holdout"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "el_by_decile.png"), dpi=150)

    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
