import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

fname = "Sample Results PD MSA.txt"
df = pd.read_csv(fname, sep="\t", comment="#", dtype=str)
df.columns = df.columns.str.strip()
df_full = df.copy()

target_col = next((c for c in df.columns if "target" in c.lower()), None)
group_col = next((c for c in df.columns if "biological" in c.lower() and "group" in c.lower()), None)
sample_col = next((c for c in df.columns if "sample" in c.lower() and "name" in c.lower()), None)

if not target_col or not group_col:
    raise ValueError("Colonne chiave mancanti (Target o Biological Group).")
if not sample_col:
    raise ValueError("Colonna Sample Name non trovata.")

valid_groups = ["HC", "MSA", "PD"]

def to_float_safe(s):
    s = (s.astype(str)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
         .str.replace(" ", "", regex=False))
    return pd.to_numeric(s, errors="coerce")

def prepare_block(df_block):
    num_cols = ["Mean DCq", "DDCq", "Rq"]
    for col in num_cols:
        if col in df_block.columns:
            df_block[col] = to_float_safe(df_block[col])
        else:
            df_block[col] = np.nan

    df_block["RQ_calc"] = df_block["Rq"]
    miss = df_block["RQ_calc"].isna() & df_block["DDCq"].notna()
    df_block.loc[miss, "RQ_calc"] = 2 ** (-df_block.loc[miss, "DDCq"])

    df_block = df_block[df_block[group_col].isin(valid_groups)].copy()
    df_block = df_block.dropna(subset=["RQ_calc"]).copy()
    df_block["RQ_calc"] = df_block["RQ_calc"].astype(float)
    df_block.loc[df_block["RQ_calc"] <= 0, "RQ_calc"] = 1e-4

    df_block[sample_col] = df_block[sample_col].astype(str).str.strip()
    df_block = df_block.drop_duplicates(subset=[sample_col], keep="first").copy()
    return df_block

def bootstrap_auc(y_true, y_scores, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
        aucs.append(auc(fpr, tpr))
    mean_auc = np.mean(aucs)
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return mean_auc, ci_low, ci_high

def compute_midrank(x):
    x = np.asarray(x)
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T)).item()
    pvalue = 2 * (1 - norm.cdf(z))
    return float(pvalue)

def delong_roc_test(y_true, pred1, pred2):
    y_true = np.asarray(y_true, dtype=int)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)

    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    preds_sorted = np.vstack([pred1[order], pred2[order]])

    label_1_count = int(np.sum(y_true_sorted))
    aucs, delongcov = fast_delong(preds_sorted, label_1_count)
    pvalue = calc_pvalue(aucs, delongcov)

    return {
        "AUC_1": float(aucs[0]),
        "AUC_2": float(aucs[1]),
        "p_delong": float(pvalue)
    }

def compute_and_plot_roc_combined(data_mt, data_tl, outname="ROC_combined.png"):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    panels = ['A', 'B', 'C', 'D']

    comparisons = [
        ("HC", "PD"),
        ("HC", "MSA"),
        ("HC", "PD"),
        ("HC", "MSA")
    ]
    data_list = [data_mt, data_mt, data_tl, data_tl]
    titles = ["mtDNA-CN", "mtDNA-CN", "Telomere Length", "Telomere Length"]

    roc_results = []

    for ax, (neg, pos), data, panel, title in zip(axes, comparisons, data_list, panels, titles):

        y_true = np.concatenate([np.zeros(len(data[neg])), np.ones(len(data[pos]))])
        y_scores = np.concatenate([data[neg], data[pos]])

        y_scores = -y_scores

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        mean_auc, ci_low, ci_high = bootstrap_auc(y_true, y_scores)

        ax.plot(fpr, tpr, color="orange", lw=2.2)
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.3)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.tick_params(axis='both', labelsize=10)

        ax.set_title(title, fontsize=12, pad=22)

        ax.text(-0.10, 1.12, panel, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

        header_y = 1.01

        ax.text(0.00, header_y,
                f"{pos} vs {neg}",
                transform=ax.transAxes,
                fontsize=11,
                ha='left', va='bottom')

        ax.text(1.00, header_y,
                f"AUC = {mean_auc:.2f} CI, [{ci_low:.2f}-{ci_high:.2f})]",
                transform=ax.transAxes,
                fontsize=11,
                ha='right', va='bottom')

        roc_results.append({
            "Marker": title,
            "Comparison": f"{pos} vs {neg}",
            "AUC_mean": mean_auc,
            "AUC_CI_low": ci_low,
            "AUC_CI_high": ci_high
        })

    plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.88)
    plt.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()

    return roc_results

df_mt = df_full[df_full[target_col].str.contains("MT", case=False, na=False)].copy()
df_mt = prepare_block(df_mt)
data_mt = {g: df_mt.loc[df_mt[group_col] == g, "RQ_calc"].values for g in valid_groups}

df_tl = df_full[df_full[target_col].str.contains("TELOMER", case=False, na=False)].copy()
df_tl = prepare_block(df_tl)
data_tl = {g: df_tl.loc[df_tl[group_col] == g, "RQ_calc"].values for g in valid_groups}

roc_results = compute_and_plot_roc_combined(data_mt, data_tl, outname="ROC_combined.png")
pd.DataFrame(roc_results).to_csv("ROC_results.csv", index=False)

merged = df_mt[[sample_col, group_col, "RQ_calc"]].rename(columns={"RQ_calc": "RQ_mt"})
merged = merged.merge(
    df_tl[[sample_col, group_col, "RQ_calc"]].rename(columns={"RQ_calc": "RQ_tl"}),
    on=[sample_col, group_col],
    how="inner"
)

delong_results = []

for neg, pos in [("HC", "PD"), ("HC", "MSA")]:
    sub = merged[merged[group_col].isin([neg, pos])].copy()

    y_true = (sub[group_col] == pos).astype(int).values

    pred_mt = -sub["RQ_mt"].astype(float).values
    pred_tl = -sub["RQ_tl"].astype(float).values

    res = delong_roc_test(y_true, pred_mt, pred_tl)

    delong_results.append({
        "Comparison": f"{pos} vs {neg}",
        "AUC_mtDNA_CN": res["AUC_1"],
        "AUC_TL": res["AUC_2"],
        "p_DeLong": res["p_delong"]
    })

delong_df = pd.DataFrame(delong_results)
delong_df.to_csv("ROC_AUC_comparison_DeLong.csv", index=False)

print("ROC_results.csv.")
print("ROC_AUC_comparison_DeLong.csv.")
print("N samples:", df_full[group_col].value_counts().to_dict())

print("\n=== DeLong ===")
print(delong_df.to_string(index=False))