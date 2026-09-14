import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

res_fname = "Sample Results PD MSA.txt"
cov_fname = "age gender PD MSA OLD ok.txt"

valid_groups = ["HC", "PD", "MSA"]
sex_levels = ["female", "male"]

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

def age_adjusted_pairwise_within_sex(df_block: pd.DataFrame, sex_value: str):
    d = df_block[df_block["Sex"] == sex_value].copy()
    d["Group"] = pd.Categorical(d[group_col], categories=["HC", "PD", "MSA"], ordered=True)
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["log2RQ"] = np.log2(d["RQ_calc"].astype(float))
    d = d.dropna(subset=["Group", "Age", "log2RQ"]).copy()

    counts = d["Group"].value_counts().reindex(valid_groups, fill_value=0)
    if counts.min() < 2:
        return None, None, d

    model = smf.ols("log2RQ ~ C(Group) + Age", data=d).fit(cov_type="HC3")

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
        {"groups": "HC vs PD",  "p": p_contrast("HC vs PD")},
        {"groups": "HC vs MSA", "p": p_contrast("HC vs MSA")},
        {"groups": "PD vs MSA", "p": p_contrast("PD vs MSA")},
    ]

    pvals = [c["p"] for c in comps]
    _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for i, c in enumerate(comps):
        c["p_FDR"] = float(p_fdr[i])

    return comps, model, d

def sex_difference_test(df_block: pd.DataFrame, biomarker_name: str, adjust_for_group: bool = True):
    d = df_block[[sample_col, group_col, "RQ_calc", "Age", "Sex"]].copy()
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["Sex"] = normalize_sex(d["Sex"])
    d = d.dropna(subset=["RQ_calc", "Age", "Sex"]).copy()

    d["log2RQ"] = np.log2(d["RQ_calc"].astype(float))

    d["Sex"] = pd.Categorical(d["Sex"], categories=["female", "male"])
    d["Group"] = pd.Categorical(d[group_col], categories=["HC", "PD", "MSA"], ordered=True)

    if adjust_for_group:
        formula = 'log2RQ ~ C(Sex, Treatment(reference="female")) + Age + C(Group)'
        coef_name = 'C(Sex, Treatment(reference="female"))[T.male]'
    else:
        formula = 'log2RQ ~ C(Sex, Treatment(reference="female")) + Age'
        coef_name = 'C(Sex, Treatment(reference="female"))[T.male]'

    model = smf.ols(formula, data=d).fit(cov_type="HC3")

    beta = float(model.params[coef_name])
    pval = float(model.pvalues[coef_name])
    ci_low, ci_high = model.conf_int().loc[coef_name].tolist()

    print(f"\n=== Male vs Female — {biomarker_name} ===")
    print("Model:", "log2RQ ~ Sex + Age + Group (HC3)" if adjust_for_group else "log2RQ ~ Sex + Age (HC3)")
    print(f"beta_male (log2 units) = {beta:.4f}  (95% CI {ci_low:.4f} to {ci_high:.4f})")
    print(f"P-value (HC3 Wald) = {pval:.3e}")

    return model

def plot_sex_panel(
    ax,
    df_block,
    sex_value,
    comps,
    ylabel,
    title,
    log_scale=True,
    jitter_seed=123,
    special_spacing=False,
):
    rng = np.random.default_rng(jitter_seed)
    positions = np.arange(len(valid_groups)) + 1

    data_dict = {
        g: df_block[(df_block[group_col] == g) & (df_block["Sex"] == sex_value)]["RQ_calc"].astype(float).values
        for g in valid_groups
    }

    for i, g in enumerate(valid_groups):
        vals = np.asarray(data_dict[g], dtype=float)

        if len(vals) > 0:
            x = rng.normal(positions[i], 0.06, len(vals))
            ax.scatter(
                x, data_dict[g],
                facecolor='black' if g == "HC" else 'white',
                edgecolor='black',
                s=28,
                linewidth=0.6,
                zorder=4
            )
            ax.boxplot(
                vals,
                positions=[positions[i]],
                widths=0.55,
                patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.3),
                whiskerprops=dict(color="black", linewidth=1.2),
                capprops=dict(color="black", linewidth=1.2),
                medianprops=dict(color=colors[g], linewidth=2.0),
                flierprops=dict(
                    marker="o",
                    markerfacecolor="black" if g == "HC" else "white",
                    markeredgecolor="black",
                    markersize=6,
                    linestyle="none"
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

    if comps is not None:
        all_vals = np.concatenate([np.asarray(v) for v in data_dict.values() if len(v) > 0])
        ymax = float(np.max(all_vals)) if len(all_vals) else 1.0

        y_levels = [ymax * 1.6, ymax * 2.4, ymax * 3.6]
        y_top = ymax * 5.0

        if special_spacing:
            base = ymax * 1.7
            step = ymax * 1.8
            y_levels = [base + i * step for i in range(3)]
            y_top = base + 4.2 * step

        ax.set_ylim(ax.get_ylim()[0], y_top)

        label_map = {"HC vs PD": (1, 2), "HC vs MSA": (1, 3), "PD vs MSA": (2, 3)}
        order = ["HC vs PD", "HC vs MSA", "PD vs MSA"]
        comp_by_name = {c["groups"]: c for c in comps}

        for k, name in enumerate(order):
            comp = comp_by_name[name]
            x1, x2 = label_map[name]
            y = y_levels[k]
            h = y * 1.08

            ax.plot([x1, x1, x2, x2], [y, h, h, y], color="black", lw=0.8, zorder=4)

            pval = comp["p_FDR"]
            if pval <= 0.0001:
                label = "****"
            elif pval <= 0.001:
                label = "***"
            elif pval <= 0.01:
                label = "**"
            elif pval <= 0.05:
                label = "*"
            else:
                label = "ns"

            ax.text((x1 + x2) / 2, h * 1.02, label,
                    ha="center", va="bottom",
                    fontsize=7, zorder=5)

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
df[target_col] = df[target_col].astype(str).str.strip()

cov[cov_sample_col] = cov[cov_sample_col].astype(str).str.strip()
cov[cov_group_col]  = cov[cov_group_col].astype(str).str.strip()

samples_in_order = df[sample_col].drop_duplicates(keep="first").tolist()

if len(samples_in_order) != len(cov):
    raise ValueError(
        f"Unique samples in Sample Results ({len(samples_in_order)}) != rows in covariates file ({len(cov)})."
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
    for col in ["DDCq", "Rq"]:
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

def sex_by_group_contrast_tests(df_block: pd.DataFrame, biomarker_name: str):

    d = df_block[[sample_col, group_col, "RQ_calc", "Age", "Sex"]].copy()
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["Sex"] = normalize_sex(d["Sex"])
    d = d.dropna(subset=["RQ_calc", "Age", "Sex"]).copy()

    d["log2RQ"] = np.log2(d["RQ_calc"].astype(float))

    d["Sex"] = pd.Categorical(d["Sex"], categories=["female", "male"])
    d["Group"] = pd.Categorical(d[group_col], categories=["HC", "PD", "MSA"], ordered=True)

    formula = (
        'log2RQ ~ '
        'C(Group, Treatment(reference="HC")) * '
        'C(Sex, Treatment(reference="female")) + '
        'Age'
    )
    model = smf.ols(formula, data=d).fit(cov_type="HC3")

    params = model.params.index.tolist()
    idx = {p: i for i, p in enumerate(params)}

    int_PD  = 'C(Group, Treatment(reference="HC"))[T.PD]:C(Sex, Treatment(reference="female"))[T.male]'
    int_MSA = 'C(Group, Treatment(reference="HC"))[T.MSA]:C(Sex, Treatment(reference="female"))[T.male]'

    missing = [t for t in [int_PD, int_MSA] if t not in idx]
    if missing:
        raise KeyError(
            "Missing expected interaction term(s) in model params: "
            + ", ".join(missing)
            + "\nAvailable params:\n"
            + "\n".join(params)
        )

    def ttest_pvalue(w):
        return float(model.t_test(w).pvalue)

    w1 = np.zeros(len(params)); w1[idx[int_PD]] = 1.0
    p_hc_pd = ttest_pvalue(w1)
    w2 = np.zeros(len(params)); w2[idx[int_MSA]] = 1.0
    p_hc_msa = ttest_pvalue(w2)
    w3 = np.zeros(len(params))
    w3[idx[int_PD]] = 1.0
    w3[idx[int_MSA]] = -1.0
    p_pd_msa = ttest_pvalue(w3)

    comps = [
        {"contrast": "Sex-difference in (HC vs PD)",  "p": p_hc_pd},
        {"contrast": "Sex-difference in (HC vs MSA)", "p": p_hc_msa},
        {"contrast": "Sex-difference in (PD vs MSA)", "p": p_pd_msa},
    ]

    # FDR across these 3 interaction contrasts
    pvals = [c["p"] for c in comps]
    _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for i, c in enumerate(comps):
        c["p_FDR"] = float(p_fdr[i])

    print(f"\n=== Sex×Group interaction contrasts — {biomarker_name} ===")
    print("Model: log2RQ ~ Group*Sex + Age (HC3)")
    for c in comps:
        print(f"{c['contrast']}: p = {c['p']:.3e} | p_FDR = {c['p_FDR']:.3e}")

    return comps, model

df_mt = prepare_block(df[df[target_col].str.contains("MT", case=False, na=False)].copy())
df_tl = prepare_block(df[df[target_col].str.contains("TELOMER", case=False, na=False)].copy())

print("MT subjects:", df_mt[sample_col].nunique(), "| TL subjects:", df_tl[sample_col].nunique())
print("Sex counts (MT):\n", df_mt["Sex"].value_counts(dropna=False))
print("Sex counts (TL):\n", df_tl["Sex"].value_counts(dropna=False))

sex_difference_test(df_mt, biomarker_name="mtDNA-CN (ND1)", adjust_for_group=True)
sex_difference_test(df_tl, biomarker_name="Telomere length (T/S)", adjust_for_group=True)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
plt.subplots_adjust(wspace=0.35, hspace=0.35)

for j, sx in enumerate(sex_levels):
    comps, model, d = age_adjusted_pairwise_within_sex(df_mt, sx)
    plot_sex_panel(
        axes[0, j],
        df_mt,
        sx,
        comps,
        ylabel="Relative ND1-CN levels (RQ)",
        title=None,
        log_scale=True,
        jitter_seed=100 + j,
        special_spacing=(j == 1),
    )

for j, sx in enumerate(sex_levels):
    comps, model, d = age_adjusted_pairwise_within_sex(df_tl, sx)
    plot_sex_panel(
        axes[1, j],
        df_tl,
        sx,
        comps,
        ylabel="Relative TL (T/S ratio, RQ)",
        title=None,
        log_scale=True,
        jitter_seed=200 + j,
        special_spacing=False
    )

for ax, lab in zip([axes[0,0], axes[0,1], axes[1,0], axes[1,1]], ["A", "B", "C", "D"]):
    ax.text(-0.22, 1.07, lab, transform=ax.transAxes, fontsize=16, fontweight="bold")

axes[0, 0].annotate("Female",
                    xy=(-0.32, 0.0001),   # move as you like
                    xycoords="axes fraction",
                    rotation=90,
                    va="center", ha="center",
                    fontsize=14, fontweight="bold")

axes[0, 1].annotate("Male",
                    xy=(1.10, 0.0001),    # move as you like
                    xycoords="axes fraction",
                    rotation=-90,
                    va="center", ha="center",
                    fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0.06, 0.04, 0.94, 0.96])
plt.savefig("Figure_SexStratified_Boxplots_MT_TL.png", dpi=300)
plt.show()

print("\nSaved: Figure_SexStratified_Boxplots_MT_TL.png")
