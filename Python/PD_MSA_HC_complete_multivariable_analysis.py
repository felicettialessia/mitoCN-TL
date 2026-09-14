import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

DEMOGRAPHIC_FILE = "age gender PD MSA OLD ok.txt"
QPCR_FILE = "Sample Results PD MSA.txt"

TARGETS = ["MT-ND1", "TELOMER"]
GROUPS = ["HC", "PD", "MSA"]

def to_numeric_decimal(series):
    return pd.to_numeric(
        series.astype(str)
              .str.strip()
              .str.replace(",", ".", regex=False),
        errors="coerce"
    )

def bh_fdr(pvalues):
    return multipletests(
        np.asarray(pvalues, dtype=float),
        method="fdr_bh"
    )[1]

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

for df in (demo, qpcr):
    df["Sample Name"] = (
        df["Sample Name"]
        .astype(str)
        .str.strip()
    )

    df["Biological Group"] = (
        df["Biological Group"]
        .astype(str)
        .str.strip()
    )

demo["Sex"] = (
    demo["Sex"]
    .astype(str)
    .str.strip()
    .str.upper()
)

demo["Age"] = to_numeric_decimal(
    demo["Age"]
)

qpcr["Target Name"] = (
    qpcr["Target Name"]
    .astype(str)
    .str.strip()
)

qpcr["Rq_numeric"] = to_numeric_decimal(
    qpcr["Rq"]
)

demo = demo[
    demo["Biological Group"].isin(GROUPS)
].copy()

qpcr_biomarkers = qpcr[
    qpcr["Biological Group"].isin(GROUPS)
    &
    qpcr["Target Name"].isin(TARGETS)
].copy()

duplicates = qpcr_biomarkers.duplicated(
    subset=["Sample Name", "Target Name"],
    keep=False
)

if duplicates.any():

    print("\nATTENZIONE: duplicati Sample/Target:")
    print(
        qpcr_biomarkers.loc[
            duplicates,
            [
                "Sample Name",
                "Biological Group",
                "Target Name",
                "Rq_numeric"
            ]
        ]
    )

    raise ValueError(
        "Sono presenti duplicati Sample Name / Target Name."
    )

wide = qpcr_biomarkers.pivot(
    index="Sample Name",
    columns="Target Name",
    values="Rq_numeric"
).reset_index()

missing = [
    target for target in TARGETS
    if target not in wide.columns
]

if missing:
    raise ValueError(
        f"Target mancanti: {missing}"
    )

data = demo[
    [
        "Sample Name",
        "Biological Group",
        "Sex",
        "Age"
    ]
].merge(
    wide,
    on="Sample Name",
    how="inner"
)

data = data.dropna(
    subset=[
        "Age",
        "Sex",
        "Biological Group",
        "MT-ND1",
        "TELOMER"
    ]
).copy()

data = data[
    data["Sex"].isin(["F", "M"])
].copy()

# RQ must be > 0 before log2
for target in TARGETS:
    if (data[target] <= 0).any():
        raise ValueError(
            f"Valori RQ <= 0 trovati per {target}."
        )

    data[f"log2RQ_{target}"] = np.log2(
        data[target]
    )

data["DiseaseStatus"] = np.where(
    data["Biological Group"] == "HC",
    "HC",
    "Synucleinopathy"
)

main_rows = []
omnibus_rows = []
pairwise_rows = []

for target in TARGETS:

    outcome = f"log2RQ_{target}"

    model = smf.ols(
        formula=(
            f'Q("{outcome}") ~ '
            f'C(Q("Biological Group"), Treatment(reference="HC")) '
            f'+ Age + C(Sex)'
        ),
        data=data
    ).fit(
        cov_type="HC3"
    )

    age_ci = model.conf_int().loc["Age"]

    sex_term = [
        x for x in model.params.index
        if x.startswith("C(Sex)[T.")
    ][0]

    sex_ci = model.conf_int().loc[
        sex_term
    ]

    main_rows.append({
        "Target": target,
        "N": int(model.nobs),

        "Beta_Age": model.params["Age"],
        "Age_CI95_low": age_ci.iloc[0],
        "Age_CI95_high": age_ci.iloc[1],
        "p_Age": model.pvalues["Age"],

        "Sex_contrast": "Male vs Female",
        "Beta_Sex": model.params[sex_term],
        "Sex_CI95_low": sex_ci.iloc[0],
        "Sex_CI95_high": sex_ci.iloc[1],
        "p_Sex": model.pvalues[sex_term],

        "R2": model.rsquared,
        "Adjusted_R2": model.rsquared_adj
    })

    group_terms = [
        x for x in model.params.index
        if 'C(Q("Biological Group")' in x
    ]

    for term in group_terms:

        ci = model.conf_int().loc[
            term
        ]

        if "[T.PD]" in term:
            comparison = "PD vs HC"
        elif "[T.MSA]" in term:
            comparison = "MSA vs HC"
        else:
            comparison = term

        pairwise_rows.append({
            "Target": target,
            "Comparison": comparison,
            "Beta_log2": model.params[term],
            "CI95_low": ci.iloc[0],
            "CI95_high": ci.iloc[1],
            "p": model.pvalues[term]
        })

    R = np.zeros(
        (
            len(group_terms),
            len(model.params)
        )
    )

    for i, term in enumerate(
        group_terms
    ):
        j = list(
            model.params.index
        ).index(term)

        R[i, j] = 1

    wald = model.wald_test(
        R,
        scalar=True
    )

    omnibus_rows.append({
        "Target": target,
        "N": int(model.nobs),
        "Wald_statistic": float(
            np.asarray(
                wald.statistic
            ).squeeze()
        ),
        "df": len(group_terms),
        "p_DiagnosticGroup": float(
            np.asarray(
                wald.pvalue
            ).squeeze()
        )
    })


main_results = pd.DataFrame(
    main_rows
)

omnibus_results = pd.DataFrame(
    omnibus_rows
)

pairwise_results = pd.DataFrame(
    pairwise_rows
)

pd_msa_rows = []

for target in TARGETS:

    outcome = f"log2RQ_{target}"

    model = smf.ols(
        formula=(
            f'Q("{outcome}") ~ '
            f'C(Q("Biological Group"), Treatment(reference="MSA")) '
            f'+ Age + C(Sex)'
        ),
        data=data
    ).fit(
        cov_type="HC3"
    )

    term = [
        x for x in model.params.index
        if 'C(Q("Biological Group")' in x
        and "[T.PD]" in x
    ][0]

    ci = model.conf_int().loc[
        term
    ]

    pd_msa_rows.append({
        "Target": target,
        "Comparison": "PD vs MSA",
        "Beta_log2": model.params[term],
        "CI95_low": ci.iloc[0],
        "CI95_high": ci.iloc[1],
        "p": model.pvalues[term]
    })


pd_msa_results = pd.DataFrame(
    pd_msa_rows
)

all_pairwise = pd.concat(
    [
        pairwise_results,
        pd_msa_results
    ],
    ignore_index=True
)

all_pairwise["p_FDR"] = np.nan

for target in TARGETS:

    mask = (
        all_pairwise["Target"]
        == target
    )

    all_pairwise.loc[
        mask,
        "p_FDR"
    ] = bh_fdr(
        all_pairwise.loc[
            mask,
            "p"
        ]
    )

main_results["p_Age_FDR"] = bh_fdr(
    main_results["p_Age"]
)

main_results["p_Sex_FDR"] = bh_fdr(
    main_results["p_Sex"]
)

omnibus_results[
    "p_DiagnosticGroup_FDR"
] = bh_fdr(
    omnibus_results[
        "p_DiagnosticGroup"
    ]
)

sensitivity_rows = []

for target in TARGETS:

    outcome = f"log2RQ_{target}"

    model = smf.ols(
        formula=(
            f'Q("{outcome}") ~ '
            f'C(Q("DiseaseStatus"), Treatment(reference="HC")) '
            f'+ Age + C(Sex)'
        ),
        data=data
    ).fit(
        cov_type="HC3"
    )

    status_term = [
        x for x in model.params.index
        if 'C(Q("DiseaseStatus")' in x
    ][0]

    ci = model.conf_int().loc[
        status_term
    ]

    beta = model.params[
        status_term
    ]

    sensitivity_rows.append({
        "Target": target,
        "N": int(model.nobs),
        "Comparison": "PD+MSA vs HC",
        "Beta_log2": beta,
        "CI95_low": ci.iloc[0],
        "CI95_high": ci.iloc[1],
        "p_DiseaseStatus": model.pvalues[
            status_term
        ],
        "RQ_ratio_PD_MSA_vs_HC": 2 ** beta,
        "R2": model.rsquared,
        "Adjusted_R2": model.rsquared_adj
    })


sensitivity_results = pd.DataFrame(
    sensitivity_rows
)

sensitivity_results[
    "p_DiseaseStatus_FDR"
] = bh_fdr(
    sensitivity_results[
        "p_DiseaseStatus"
    ]
)

correlation_rows = []

for group in GROUPS:

    subset = data[
        data["Biological Group"]
        == group
    ].copy()

    for target in TARGETS:

        outcome = f"log2RQ_{target}"

        rho, p = spearmanr(
            subset["Age"],
            subset[outcome]
        )

        correlation_rows.append({
            "Group": group,
            "Target": target,
            "N": len(subset),
            "Spearman_rho": rho,
            "p": p
        })


age_correlations = pd.DataFrame(
    correlation_rows
)

age_correlations[
    "p_FDR_within_group"
] = np.nan

for group in GROUPS:

    mask = (
        age_correlations["Group"]
        == group
    )

    age_correlations.loc[
        mask,
        "p_FDR_within_group"
    ] = bh_fdr(
        age_correlations.loc[
            mask,
            "p"
        ]
    )

pd.set_option(
    "display.max_columns",
    None
)

print("\n==========================================")
print("MAIN MODEL")
print("log2RQ ~ diagnosis + age + sex")
print("OLS + HC3")
print("==========================================\n")

print(
    main_results.to_string(
        index=False
    )
)


print("\n==========================================")
print("DIAGNOSTIC GROUP OMNIBUS")
print("==========================================\n")

print(
    omnibus_results.to_string(
        index=False
    )
)


print("\n==========================================")
print("PAIRWISE GROUP COMPARISONS")
print("BH-FDR within each biomarker")
print("==========================================\n")

print(
    all_pairwise.to_string(
        index=False
    )
)


print("\n==========================================")
print("SENSITIVITY: PD + MSA vs HC")
print("==========================================\n")

print(
    sensitivity_results.to_string(
        index=False
    )
)


print("\n==========================================")
print("AGE CORRELATIONS BY GROUP")
print("==========================================\n")

print(
    age_correlations.to_string(
        index=False
    )
)

data.to_csv(
    "PD_MSA_HC_log2RQ_data.csv",
    index=False
)

main_results.to_csv(
    "Main_model_age_sex_diagnosis_HC3.csv",
    index=False
)

omnibus_results.to_csv(
    "Diagnostic_group_omnibus_HC3.csv",
    index=False
)

all_pairwise.to_csv(
    "Diagnostic_group_pairwise_HC3_FDR.csv",
    index=False
)

sensitivity_results.to_csv(
    "Sensitivity_PD_MSA_combined_vs_HC_HC3.csv",
    index=False
)

age_correlations.to_csv(
    "Age_correlations_by_diagnostic_group.csv",
    index=False
)


print("\n==========================================")
print("ANALYSIS COMPLETED")
print("==========================================")

restricted_data = data[
    data["Age"].between(
        58,
        78,
        inclusive="both"
    )
].copy()

print("\n==========================================")
print("FINAL SENSITIVITY ANALYSIS")
print("RESTRICTED COMMON AGE RANGE: 58-78 YEARS")
print("==========================================")

print(f"\nN restricted cohort = {len(restricted_data)}")

print("\nN per group:")
print(
    restricted_data["Biological Group"]
    .value_counts()
    .reindex(GROUPS)
)

restricted_rows = []
restricted_omnibus_rows = []

for target in TARGETS:

    outcome = f"log2RQ_{target}"

    model = smf.ols(
        formula=(
            f'Q("{outcome}") ~ '
            f'C(Q("Biological Group"), Treatment(reference="HC")) '
            f'+ Age + C(Sex)'
        ),
        data=restricted_data
    ).fit(
        cov_type="HC3"
    )

    age_ci = model.conf_int().loc["Age"]
    sex_term = [
        x for x in model.params.index
        if x.startswith("C(Sex)[T.")
    ][0]

    sex_ci = model.conf_int().loc[
        sex_term
    ]

    restricted_rows.append({
        "Target": target,
        "N": int(model.nobs),

        "Beta_Age": model.params["Age"],
        "Age_CI95_low": age_ci.iloc[0],
        "Age_CI95_high": age_ci.iloc[1],
        "p_Age": model.pvalues["Age"],

        "Sex_contrast": "Male vs Female",
        "Beta_Sex": model.params[sex_term],
        "Sex_CI95_low": sex_ci.iloc[0],
        "Sex_CI95_high": sex_ci.iloc[1],
        "p_Sex": model.pvalues[sex_term],

        "R2": model.rsquared,
        "Adjusted_R2": model.rsquared_adj
    })
   group_terms = [
        x for x in model.params.index
        if 'C(Q("Biological Group")' in x
    ]

    R = np.zeros(
        (
            len(group_terms),
            len(model.params)
        )
    )

    for i, term in enumerate(group_terms):

        j = list(
            model.params.index
        ).index(term)

        R[i, j] = 1

    wald = model.wald_test(
        R,
        scalar=True
    )

    restricted_omnibus_rows.append({
        "Target": target,
        "N": int(model.nobs),
        "Wald_statistic": float(
            np.asarray(
                wald.statistic
            ).squeeze()
        ),
        "df": len(group_terms),
        "p_DiagnosticGroup": float(
            np.asarray(
                wald.pvalue
            ).squeeze()
        )
    })


restricted_results = pd.DataFrame(
    restricted_rows
)

restricted_omnibus_results = pd.DataFrame(
    restricted_omnibus_rows
)

restricted_results["p_Age_FDR"] = bh_fdr(
    restricted_results["p_Age"]
)

restricted_results["p_Sex_FDR"] = bh_fdr(
    restricted_results["p_Sex"]
)

restricted_omnibus_results[
    "p_DiagnosticGroup_FDR"
] = bh_fdr(
    restricted_omnibus_results[
        "p_DiagnosticGroup"
    ]
)

print("\n==========================================")
print("RESTRICTED AGE 58-78: MAIN MODEL")
print("log2RQ ~ diagnosis + age + sex")
print("OLS + HC3")
print("==========================================\n")

print(
    restricted_results.to_string(
        index=False
    )
)

print("\n==========================================")
print("RESTRICTED AGE 58-78:")
print("DIAGNOSTIC GROUP OMNIBUS")
print("==========================================\n")

print(
    restricted_omnibus_results.to_string(
        index=False
    )
)

restricted_results.to_csv(
    "Restricted_age_58_78_main_model_HC3.csv",
    index=False
)

restricted_omnibus_results.to_csv(
    "Restricted_age_58_78_diagnostic_group_omnibus_HC3.csv",
    index=False
)

print("\nFinal restricted-age sensitivity analysis completed.")
