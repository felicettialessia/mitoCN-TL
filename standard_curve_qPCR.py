import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("Sample Results_dilution1.txt", sep="\t")

for col in ["Mean Equivalent Cq", "Sigma Equivalent Cq"]:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["Sample Name"].str.contains(r"S\d")]

conc_map = {
    "S1": 5, "S2":0.5,"S3": 0.05, "S4": 0.005, "S5": 0.0005
}
df["Sample_ID"] = df["Sample Name"].str.extract(r"(S\d)")
df["Concentration_ng"] = df["Sample_ID"].map(conc_map)

def fit_standard_curve(conc, Ct):
    log_conc = np.log10(conc)
    X = sm.add_constant(log_conc)
    model = sm.OLS(Ct, X).fit()
    slope, intercept = model.params[1], model.params[0]
    r2 = model.rsquared
    efficiency = 10 ** (-1 / slope)
    return slope, intercept, r2, efficiency, model

plt.figure(figsize=(8, 6))
plt.style.use('seaborn-v0_8-white')

genes = {
    "ND1 (mitochondrial)": "MT-ND1",
    "Telomere (telomeric repeats)": "TELOMER",
    "β-actin (nuclear)": "b-actin"
}

colors = {
    "ND1 (mitochondrial)": "black",
    "Telomere (telomeric repeats)": "firebrick",
    "β-actin (nuclear)": "dimgray"
}

markers = {
    "ND1 (mitochondrial)": "o",
    "Telomere (telomeric repeats)": "D",
    "β-actin (nuclear)": "s"
}

annotation_lines = []

for label, key in genes.items():
    data = df[df["Target Name"].str.contains(key, case=False)]

    conc = data["Concentration_ng"].values
    Ct = data["Mean Equivalent Cq"].values
    sigma = data["Sigma Equivalent Cq"].values

    slope, intercept, r2, eff, model = fit_standard_curve(conc, Ct)

    log_conc = np.log10(conc)
    Ct_fit = model.predict(sm.add_constant(log_conc))

    plt.errorbar(
        conc, Ct, yerr=sigma,
        fmt=markers[label], color=colors[label], ecolor=colors[label],
        markersize=6, elinewidth=1, capsize=3,
        label=label
    )

    plt.plot(conc, Ct_fit, color=colors[label], linewidth=1)

    annotation_lines.append(
        f"{label}: slope = {slope:.3f}, E = {eff:.2f}, R² = {r2:.4f}"

    )

start_y = 0.97
step = 0.05

for i, text in enumerate(annotation_lines):
    plt.text(
        0.97,                 # x position (relative axes coords)
        start_y - i * step,   # y position
        text,
        fontsize=10,
        color="black",
        ha="right",
        va="top",
        transform=plt.gca().transAxes
    )

plt.xscale("log")
plt.xlim(1e-4, 10)
plt.xlabel("log DNA input (ng)", fontsize=12)
plt.ylabel("Cycle threshold (Ct)", fontsize=12)
plt.title("qPCR standard curves (ND1, Telomere, β-actin)", fontsize=14)
plt.tick_params(which="both", direction="in", top=True, right=True)
plt.grid(False)
plt.legend(frameon=False, fontsize=10, loc="lower left")

plt.tight_layout()
plt.show()

