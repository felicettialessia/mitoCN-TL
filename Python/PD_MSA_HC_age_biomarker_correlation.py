import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

DEMOGRAPHIC_FILE = "age gender PD MSA OLD ok.txt"
QPCR_FILE = "Sample Results PD MSA.txt"

CQ_COL = "Mean Adjusted Equivalent Cq"
TARGETS = ["MT-ND1", "TELOMER"]
REFERENCE = "b-actin"

def to_numeric_decimal(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .str.strip(),
        errors="coerce"
    )

demo = pd.read_csv(
    DEMOGRAPHIC_FILE,
    sep="\t",
    dtype={"Sample Name": str}
)

qpcr = pd.read_csv(
    QPCR_FILE,
    sep="\t",
    dtype={"Sample Name": str}
)

demo.columns = demo.columns.str.strip()
qpcr.columns = qpcr.columns.str.strip()

demo["Sample Name"] = demo["Sample Name"].astype(str).str.strip()
qpcr["Sample Name"] = qpcr["Sample Name"].astype(str).str.strip()

demo["Biological Group"] = (
    demo["Biological Group"].astype(str).str.strip()
)

qpcr["Biological Group"] = (
    qpcr["Biological Group"].astype(str).str.strip()
)

qpcr["Target Name"] = (
    qpcr["Target Name"].astype(str).str.strip()
)

demo["Sex"] = demo["Sex"].astype(str).str.strip()
demo["Age"] = to_numeric_decimal(demo["Age"])

qpcr[CQ_COL] = to_numeric_decimal(qpcr[CQ_COL])

groups = ["HC", "PD", "MSA"]

demo = demo[
    demo["Biological Group"].isin(groups)
].copy()

qpcr = qpcr[
    qpcr["Biological Group"].isin(groups)
].copy()

duplicates = qpcr.duplicated(
    subset=["Sample Name", "Target Name"],
    keep=False
)

if duplicates.any():

    print("\nWARNING: duplicate Sample Name / Target Name:")
    print(
        qpcr.loc[
            duplicates,
            ["Sample Name", "Biological Group", "Target Name", CQ_COL]
        ].sort_values(["Sample Name", "Target Name"])
    )

    raise ValueError(
        "There are duplicates."
        "Check the file before going on."
    )

wide = qpcr.pivot(
    index="Sample Name",
    columns="Target Name",
    values=CQ_COL
).reset_index()

required_targets = TARGETS + [REFERENCE]

missing_targets = [
    target
    for target in required_targets
    if target not in wide.columns
]

if missing_targets:
    raise ValueError(
        f"Missing targets: {missing_targets}"
    )

data = demo[
    ["Sample Name", "Biological Group", "Sex", "Age"]
].merge(
    wide,
    on="Sample Name",
    how="inner"
)

data = data.dropna(
    subset=[
        "Age",
        "Biological Group",
        "Sex",
        "MT-ND1",
        "TELOMER",
        "b-actin"
    ]
).copy()

for target in TARGETS:

    data[f"dCq_{target}"] = (
        data[target] - data[REFERENCE]
    )

    data[f"Norm_{target}"] = (
        2 ** (-data[f"dCq_{target}"])
    )

    data[f"log2Norm_{target}"] = (
        -data[f"dCq_{target}"]
    )

print("\n==========================================")
print("Overall cohort PD-MSA-HC")
print("==========================================")

print(f"\nN tot = {len(data)}")

print("\nN samples by group:")
print(
    data["Biological Group"]
    .value_counts()
    .reindex(["HC", "PD", "MSA"])
)

print("\nAge by group:")
print(
    data.groupby("Biological Group")["Age"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reindex(["HC", "PD", "MSA"])
)

results = []

for target in TARGETS:

    outcome = f"log2Norm_{target}"

    tmp = data[
        ["Age", outcome]
    ].dropna()

    rho, p_spearman = spearmanr(
        tmp["Age"],
        tmp[outcome]
    )

    r, p_pearson = pearsonr(
        tmp["Age"],
        tmp[outcome]
    )

    results.append({
        "Target": target,
        "N": len(tmp),
        "Spearman_rho": rho,
        "Spearman_p": p_spearman,
        "Pearson_r": r,
        "Pearson_p": p_pearson
    })


results = pd.DataFrame(results)

results["Spearman_p_FDR"] = multipletests(
    results["Spearman_p"],
    method="fdr_bh"
)[1]

results["Pearson_p_FDR"] = multipletests(
    results["Pearson_p"],
    method="fdr_bh"
)[1]


print("\n==========================================")
print("Age-biomarkers correlation")
print("Overall cohort PD + MSA + HC")
print("Normalization: b-actin")
print("Variable: -DeltaCq")
print("==========================================\n")

print(results.to_string(index=False))

adjusted_results = []

for target in TARGETS:

    outcome = f"log2Norm_{target}"

    model = smf.ols(
        formula=(
            f'Q("{outcome}") ~ Age '
            f'+ C(Q("Biological Group"), Treatment(reference="HC")) '
            f'+ C(Sex)'
        ),
        data=data
    ).fit(
        cov_type="HC3"
    )

    beta = model.params["Age"]
    p_age = model.pvalues["Age"]

    ci = model.conf_int().loc["Age"]

    adjusted_results.append({
        "Target": target,
        "N": int(model.nobs),
        "Beta_Age": beta,
        "CI95_low": ci.iloc[0],
        "CI95_high": ci.iloc[1],
        "p_Age": p_age,
        "R_squared": model.rsquared,
        "Adjusted_R_squared": model.rsquared_adj
    })


adjusted_results = pd.DataFrame(adjusted_results)

adjusted_results["p_Age_FDR"] = multipletests(
    adjusted_results["p_Age"],
    method="fdr_bh"
)[1]


print("\n==========================================")
print("Age association")
print("Model: biomarker ~ Age + Diagnosis + Sex")
print("OLS with robust SE HC3")
print("==========================================\n")

print(adjusted_results.to_string(index=False))

stratified_results = []

for group in ["HC", "PD", "MSA"]:

    subset = data[
        data["Biological Group"] == group
    ].copy()

    for target in TARGETS:

        outcome = f"log2Norm_{target}"

        tmp = subset[
            ["Age", outcome]
        ].dropna()

        rho, p = spearmanr(
            tmp["Age"],
            tmp[outcome]
        )

        stratified_results.append({
            "Group": group,
            "Target": target,
            "N": len(tmp),
            "Spearman_rho": rho,
            "p": p
        })


stratified_results = pd.DataFrame(
    stratified_results
)

print("\n==========================================")
print("Group-stratified Spearman")
print("==========================================\n")

print(stratified_results.to_string(index=False))

data.to_csv(
    "PD_MSA_HC_normalized_data.csv",
    index=False
)

results.to_csv(
    "PD_MSA_HC_age_correlations.csv",
    index=False
)

adjusted_results.to_csv(
    "PD_MSA_HC_age_adjusted_HC3.csv",
    index=False
)

stratified_results.to_csv(
    "PD_MSA_HC_age_correlations_by_group.csv",
    index=False
)

for target in TARGETS:

    outcome = f"log2Norm_{target}"

    tmp = data[
        ["Age", outcome]
    ].dropna()

    rho, p = spearmanr(
        tmp["Age"],
        tmp[outcome]
    )

    plt.figure(figsize=(6, 5))

    plt.scatter(
        tmp["Age"],
        tmp[outcome],
        alpha=0.70
    )

    slope, intercept = np.polyfit(
        tmp["Age"],
        tmp[outcome],
        1
    )

    x_line = np.linspace(
        tmp["Age"].min(),
        tmp["Age"].max(),
        100
    )

    plt.plot(
        x_line,
        slope * x_line + intercept
    )

    plt.xlabel("Age (years)")

    plt.ylabel(
        f"{target} normalized to b-actin (-Delta Cq)"
    )

    plt.title(
        f"PD + MSA + HC: Age vs {target}\n"
        f"Spearman rho = {rho:.3f}, p = {p:.4g}"
    )

    plt.tight_layout()

    safe_target = target.replace("-", "_")

    plt.savefig(
        f"PD_MSA_HC_Age_vs_{safe_target}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("\nAnalysis complete.")
print("Output:")
print("- PD_MSA_HC_normalized_data.csv")
print("- PD_MSA_HC_age_correlations.csv")
print("- PD_MSA_HC_age_adjusted_HC3.csv")
print("- PD_MSA_HC_age_correlations_by_group.csv")
print("- PNG graphs for MT-ND1 e TELOMER")
