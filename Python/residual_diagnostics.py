import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot

valid_groups = ["HC", "PD", "MSA"]
colors = {"HC": "#A5D6A7", "PD": "#EF9A9A", "MSA": "#90CAF9"}

def to_float_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def normalize_sex(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip().str.lower()
    mapping = {
        "m": "male", "male": "male", "man": "male", "maschio": "male",
        "f": "female", "female": "female", "woman": "female", "femmina": "female",
    }
    s = s.replace(mapping)
    s = s.where(s.isin(["male", "female"]), np.nan)
    return s

def cohens_d(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp

def pct_within_iqr(x, ref) -> float:
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    x = x[~np.isnan(x)]
    ref = ref[~np.isnan(ref)]
    if len(x) == 0 or len(ref) < 4:
        return np.nan
    q1, q3 = np.percentile(ref, [25, 75])
    return 100.0 * np.mean((x >= q1) & (x <= q3))

res_fname = "Sample Results PD MSA.txt"
cov_fname = "age gender PD MSA OLD ok.txt"

df = pd.read_csv(res_fname, sep="\t", comment="#", dtype=str)
df.columns = df.columns.str.strip()

cov = pd.read_csv(cov_fname, sep="\t", dtype=str)
cov.columns = cov.columns.str.strip()

sample_col = next((c for c in df.columns if "sample" in c.lower() and "name" in c.lower()), None)
group_col  = next((c for c in df.columns if "biological" in c.lower() and "group" in c.lower()), None)
target_col = next((c for c in df.columns if "target" in c.lower()), None)

cov_sample_col = next((c for c in cov.columns if "sample" in c.lower() and "name" in c.lower()), None)
cov_group_col  = next((c for c in cov.columns if "biological" in c.lower() and "group" in c.lower()), None)
cov_age_col    = next((c for c in cov.columns if c.lower().strip() == "age"), None)
cov_sex_col    = next((c for c in cov.columns if c.lower().strip() == "sex"), None)

if any(x is None for x in [sample_col, group_col, target_col, cov_sample_col, cov_group_col, cov_age_col, cov_sex_col]):
    raise ValueError("Missing key columns (Sample Name / Biological Group / Target Name / Age / Sex).")

df[sample_col] = df[sample_col].astype(str).str.strip()
df[group_col]  = df[group_col].astype(str).str.strip()

cov[cov_sample_col] = cov[cov_sample_col].astype(str).str.strip()
cov[cov_group_col]  = cov[cov_group_col].astype(str).str.strip()

samples_in_order = df[sample_col].drop_duplicates(keep="first").tolist()

if len(samples_in_order) != len(cov):
    raise ValueError(
        f"Unique samples in Sample Results ({len(samples_in_order)}) "
        f"!= rows in covariates file ({len(cov)})."
    )

map_df = pd.DataFrame({
    "SampleName_key": samples_in_order,
    "Age": pd.to_numeric(cov[cov_age_col], errors="coerce"),
    "Sex": normalize_sex(cov[cov_sex_col]),
    "Group_cov": cov[cov_group_col].astype(str).str.strip()
})

df = df.merge(map_df, left_on=sample_col, right_on="SampleName_key", how="left")

bad = df.loc[df["Group_cov"].notna() & (df[group_col] != df["Group_cov"]),
             [sample_col, group_col, "Group_cov"]].drop_duplicates()
if len(bad) > 0:
    print("WARNING: mismatch between Biological Group in the two files (first 10):")
    print(bad.head(10))

df = df.dropna(subset=["Age", "Sex"]).copy()

def prepare_block(df_block: pd.DataFrame) -> pd.DataFrame:
    for col in ["Mean DCq", "DDCq", "Rq"]:
        df_block[col] = to_float_safe(df_block[col]) if col in df_block.columns else np.nan

    df_block["RQ_calc"] = df_block["Rq"]
    miss = df_block["RQ_calc"].isna() & df_block["DDCq"].notna()
    df_block.loc[miss, "RQ_calc"] = 2 ** (-df_block.loc[miss, "DDCq"])

    df_block = df_block[df_block[group_col].isin(valid_groups)].copy()
    df_block = df_block.dropna(subset=["RQ_calc", "Age", "Sex"]).copy()

    df_block["RQ_calc"] = df_block["RQ_calc"].astype(float)
    df_block.loc[df_block["RQ_calc"] <= 0, "RQ_calc"] = 1e-4
    df_block = df_block.drop_duplicates(subset=[sample_col], keep="first").copy()
    return df_block

df_mt = prepare_block(df[df[target_col].str.contains("MT", case=False, na=False)].copy())
df_tl = prepare_block(df[df[target_col].str.contains("TELOMER", case=False, na=False)].copy())

print("MT rows:", len(df_mt), " | unique subjects:", df_mt[sample_col].nunique())
print("TL rows:", len(df_tl), " | unique subjects:", df_tl[sample_col].nunique())

def adj_pairwise(df_block: pd.DataFrame):
    d = df_block[[sample_col, group_col, "RQ_calc", "Age", "Sex"]].copy()
    d["Group"] = pd.Categorical(d[group_col], categories=["HC", "PD", "MSA"], ordered=True)
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["Sex"] = normalize_sex(d["Sex"])

    d["log2RQ"] = np.log2(d["RQ_calc"].astype(float))
    d = d.dropna(subset=["Group", "Age", "Sex", "log2RQ"]).copy()

    model = smf.ols("log2RQ ~ C(Group) + Age + C(Sex)", data=d).fit(cov_type="HC3")

    params = model.params.index.tolist()
    idx = {p: i for i, p in enumerate(params)}

    def p_contrast(label: str) -> float:
        w = np.zeros(len(params))
        if label == "HC vs PD":
            w[idx["C(Group)[T.PD]"]] = 1.0
        elif label == "HC vs MSA":
            w[idx["C(Group)[T.MSA]"]] = 1.0
        elif label == "PD vs MSA":
            w[idx["C(Group)[T.PD]"]] = 1.0
            w[idx["C(Group)[T.MSA]"]] = -1.0
        else:
            raise ValueError(label)
        return float(model.t_test(w).pvalue)

    comps = [
        {"groups": "HC vs PD",  "p_adj": p_contrast("HC vs PD")},
        {"groups": "HC vs MSA", "p_adj": p_contrast("HC vs MSA")},
        {"groups": "PD vs MSA", "p_adj": p_contrast("PD vs MSA")},
    ]
    pvals = [c["p_adj"] for c in comps]
    _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for i, c in enumerate(comps):
        c["p_adj_FDR"] = float(p_fdr[i])

    return comps, model, d

def model_diagnostics(model, label: str, out_prefix: str = "diagnostics"):

    resid = pd.Series(model.resid).astype(float)
    fitted = pd.Series(model.fittedvalues).astype(float)

    if len(resid) >= 3:
        shapiro_stat, shapiro_p = shapiro(resid)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid, model.model.exog)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    qqplot(
        resid, line="45", ax=axes[0],
        markerfacecolor="black", markeredgecolor="black", alpha=0.7
    )
    axes[0].set_title(f"{label}: Q-Q plot residuals", fontsize=11)

    axes[1].scatter(
        fitted, resid,
        edgecolor="black", facecolor="white", s=28, linewidth=0.8
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"{label}: Residuals vs fitted", fontsize=11)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(
        f"{out_prefix}_{label}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    normality_flag = "violated" if (pd.notna(shapiro_p) and shapiro_p < 0.05) else "not violated"
    hetero_flag = "present" if bp_f_p < 0.05 else "not detected"

    interpretation = (
        f"{label}: Residual normality {normality_flag} "
        f"(Shapiro p={shapiro_p:.3g}); "
        f"heteroskedasticity {hetero_flag} "
        f"(Breusch–Pagan p={bp_f_p:.3g}). "
        f"Inference based on HC3 robust standard errors."
    )

    print("\n=== Diagnostics for", label, "===")
    print(f"Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.4g}")
    print(f"Breusch-Pagan: LM p = {bp_lm_p:.4g}, F p = {bp_f_p:.4g}")
    print("Summary:", interpretation)

    return {
        "Model": label,
        "N": int(len(resid)),
        "Shapiro_W": float(shapiro_stat) if pd.notna(shapiro_stat) else np.nan,
        "Shapiro_p": float(shapiro_p) if pd.notna(shapiro_p) else np.nan,
        "BP_LM": float(bp_lm),
        "BP_LM_p": float(bp_lm_p),
        "BP_F": float(bp_f),
        "BP_F_p": float(bp_f_p),
        "Interpretation": interpretation
    }

comparisons_mt, model_mt, d_mt = adj_pairwise(df_mt)
comparisons_tl, model_tl, d_tl = adj_pairwise(df_tl)

diag_mt = model_diagnostics(model_mt, "mtDNA-CN", out_prefix="model_diagnostics")
diag_tl = model_diagnostics(model_tl, "Telomere_length", out_prefix="model_diagnostics")

diag_df = pd.DataFrame([diag_mt, diag_tl])
diag_df.to_csv("model_diagnostics_summary.csv", index=False)

with open("model_diagnostics_interpretation.txt", "w") as f:
    for row in diag_df["Interpretation"]:
        f.write(row + "\n")

print("\nSaved:")
print("- model_diagnostics_summary.csv")
print("- model_diagnostics_interpretation.txt")

def effects_and_overlap(d: pd.DataFrame, label: str):
    l2_hc  = d.loc[d["Group"] == "HC",  "log2RQ"].astype(float).values
    l2_pd  = d.loc[d["Group"] == "PD",  "log2RQ"].astype(float).values
    l2_msa = d.loc[d["Group"] == "MSA", "log2RQ"].astype(float).values

    d_pd_hc  = cohens_d(l2_pd,  l2_hc)
    d_msa_hc = cohens_d(l2_msa, l2_hc)
    d_pd_msa = cohens_d(l2_pd,  l2_msa)

    pct_pd_in_hc_iqr  = pct_within_iqr(l2_pd,  l2_hc)
    pct_msa_in_hc_iqr = pct_within_iqr(l2_msa, l2_hc)

    print(f"\n=== {label}: Cohen’s d (log2RQ) & overlap within HC IQR (log2RQ) ===")
    print(f"Cohen’s d: PD vs HC = {d_pd_hc:.2f}; MSA vs HC = {d_msa_hc:.2f}; PD vs MSA = {d_pd_msa:.2f}")
    print(f"% within HC IQR: PD = {pct_pd_in_hc_iqr:.1f}%; MSA = {pct_msa_in_hc_iqr:.1f}%")

    return {
        "d_log2": {"PD_vs_HC": d_pd_hc, "MSA_vs_HC": d_msa_hc, "PD_vs_MSA": d_pd_msa},
        "pct_in_hc_iqr_log2": {"PD": pct_pd_in_hc_iqr, "MSA": pct_msa_in_hc_iqr},
    }

effects_mt = effects_and_overlap(d_mt, "mtDNA-CN")
effects_tl = effects_and_overlap(d_tl, "Telomere length")
data_mt = {g: df_mt.loc[df_mt[group_col] == g, "RQ_calc"].astype(float).values for g in valid_groups}
data_tl = {g: df_tl.loc[df_tl[group_col] == g, "RQ_calc"].astype(float).values for g in valid_groups}

def plot_panel(ax, data_dict, comps, ylabel, log_scale=False, jitter_seed=123):
    rng = np.random.default_rng(jitter_seed)
    positions = np.arange(len(valid_groups)) + 1

    for i, g in enumerate(valid_groups):
        x = rng.normal(positions[i], 0.06, len(data_dict[g]))
        ax.scatter(
            x, data_dict[g],
            facecolor='black' if g == "HC" else 'white',
            edgecolor='black',
            s=28,
            linewidth=0.6,
            zorder=4
        )

        ax.boxplot(
            data_dict[g],
            positions=[positions[i]],
            widths=0.55,
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.3),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2),
            medianprops=dict(color=colors[g], linewidth=2.0),
            flierprops=dict(
                marker='o',
                markerfacecolor='black' if g == "HC" else 'white',
                markeredgecolor='black',
                markersize=6,
                linestyle='none'
            )
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(valid_groups, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.6)

    if log_scale:
        ax.set_yscale("log")

    all_vals = np.concatenate([np.asarray(v) for v in data_dict.values() if len(v) > 0])
    ymax = float(np.max(all_vals)) if len(all_vals) else 1.0
    y_levels = [ymax * 1.6, ymax * 2.4, ymax * 3.6]
    ax.set_ylim(ax.get_ylim()[0], ymax * 5.0)

    label_map = {"HC vs PD": (1, 2), "HC vs MSA": (1, 3), "PD vs MSA": (2, 3)}
    order = ["HC vs PD", "HC vs MSA", "PD vs MSA"]
    comp_by_name = {c["groups"]: c for c in comps}

    for k, name in enumerate(order):
        comp = comp_by_name[name]
        x1, x2 = label_map[name]
        y = y_levels[k]
        h = y * 1.08
        ax.plot([x1, x1, x2, x2], [y, h, h, y], color="black", lw=0.8, zorder=4)

        txt = f"$p_{{FDR}}$={comp['p_adj_FDR']:.2e}"
        text_y = h * 1.02
        if name == "HC vs PD":
            text_y *= 0.96
        if name == "HC vs MSA":
            text_y *= 0.97

        ax.text((x1 + x2) / 2, text_y, txt, ha="center", va="bottom", fontsize=8, zorder=5)

def plot_rq_panel(ax, rq_dict, ylabel):
    groups = ["HC", "PD", "MSA"]
    vals = [rq_dict[g] for g in groups]
    x = np.arange(len(groups))

    ax.bar(
        x, vals,
        edgecolor="black",
        linewidth=1.1,
        color=[colors[g] for g in groups],
        width=0.75
    )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.6)

    ax.set_ylim(0, 1.05)

    for i, g in enumerate(groups):
        v = vals[i]
        label = "1.000" if g == "HC" else f"{v:.3f}"
        ax.text(i, v + 0.03, label, ha="center", va="bottom", fontsize=11)

rq_mt = {"HC": 1.0, "PD": 0.038, "MSA": 0.193}
rq_tl = {"HC": 1.0, "PD": 0.135, "MSA": 0.183}
fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), gridspec_kw={"height_ratios": [3.2, 1.6]})

plot_panel(axes[0, 0], data_mt, comparisons_mt, ylabel="Relative ND1-CN levels", log_scale=True, jitter_seed=123)
plot_panel(axes[0, 1], data_tl, comparisons_tl, ylabel="Relative TL (T/S ratio)", log_scale=True, jitter_seed=456)

plot_rq_panel(axes[1, 0], rq_mt, ylabel="RQ vs HC (mtDNA-CN)")
plot_rq_panel(axes[1, 1], rq_tl, ylabel="RQ vs HC (TL)")

panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
for ax, label in zip(panel_axes, ["A", "B", "C", "D"]):
    ax.text(-0.25, 1.05, label, transform=ax.transAxes, fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig("Figure1.png", dpi=300)
plt.show()

print("\nMT model (age+sex adjusted):\n", model_mt.summary())
print("\nTL model (age+sex adjusted):\n", model_tl.summary())