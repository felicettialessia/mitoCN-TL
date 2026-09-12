import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


df_data = pd.read_csv("Sample Results PD MSA.txt", sep="\t", comment="#")
df_meta = pd.read_csv("age gender PD MSA OLD ok.txt", sep="\t", comment="#")

df_data.columns = df_data.columns.str.strip()
df_meta.columns = df_meta.columns.str.strip()

df = df_data.merge(df_meta[['Sample Name', 'Sex', 'Age']], on='Sample Name', how='left')

def to_float_safe(s):
    s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

df['DDCq'] = to_float_safe(df['DDCq'])
df['Rq'] = to_float_safe(df['Rq'])
df['RQ_calc'] = df['Rq']
mask = df['RQ_calc'].isna() & df['DDCq'].notna()
df.loc[mask, 'RQ_calc'] = 2 ** (-df.loc[mask, 'DDCq'])
df['RQ_calc'] = df['RQ_calc'].clip(lower=1e-4)

bins = [57, 63, 69, 75, 81]
labels = ["58-63", "64-69", "70-75", "76-81"]
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

colors = {"HC": "#A5D6A7", "PD": "#EF9A9A", "MSA": "#90CAF9"}

def pairwise_tests(data_dict):
    comps = [("HC vs PD", "HC", "PD"),
             ("HC vs MSA", "HC", "MSA"),
             ("PD vs MSA", "PD", "MSA")]
    results = []
    for label, g1, g2 in comps:
        if len(data_dict[g1]) > 0 and len(data_dict[g2]) > 0:
            _, p = mannwhitneyu(data_dict[g1], data_dict[g2], alternative="two-sided")
        else:
            p = np.nan
        results.append({"groups": label, "mw_p": p})
    return results

def add_fdr(comps):
    pvals = [c["mw_p"] for c in comps]
    _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for i in range(len(comps)):
        comps[i]["mw_p_FDR"] = pvals_fdr[i]
    return comps

def p_to_asterisk(p):
    if np.isnan(p):
        return 'ns'
    elif p <= 0.001:
        return '***'
    elif p <= 0.01:
        return '**'
    elif p <= 0.05:
        return '*'
    else:
        return 'ns'

def scatter_by_age_on_axes(df_subset, target_name, ylabel, axes, panel_labels):
    groups = ["HC", "PD", "MSA"]
    age_groups = pd.Categorical(df_subset['AgeGroup'], categories=labels, ordered=True).categories
    line_shift_log10_by_panel = {"B": 0.08}

    label_map = {"HC vs PD": (1, 2), "HC vs MSA": (1, 3), "PD vs MSA": (2, 3)}

    for idx, (ax, age) in enumerate(zip(axes, age_groups)):
        panel_id = panel_labels[idx]

        df_age = df_subset[df_subset['AgeGroup'] == age]

        data_dict = {
            g: df_age[
                (df_age['Biological Group'] == g) &
                (df_age['Target Name'].str.contains(target_name, case=False, na=False))
            ]['RQ_calc'].values
            for g in groups
        }

        positions = np.arange(len(groups)) + 1

        for i, g in enumerate(groups):
            x = np.random.normal(positions[i], 0.05, len(data_dict[g]))
            ax.scatter(
                x, data_dict[g],
                facecolors='black' if g == "HC" else 'white',
                edgecolors='black',
                s=50, linewidth=0.9,
                zorder=1
            )

        for i, g in enumerate(groups):
            if len(data_dict[g]) > 0:
                med = np.median(data_dict[g])
                ax.hlines(
                    med,
                    positions[i] - 0.2,
                    positions[i] + 0.2,
                    colors=colors[g],
                    linewidth=2,
                    zorder=2
                )

        comps = add_fdr(pairwise_tests(data_dict))

        if any(len(v) > 0 for v in data_dict.values()):
            all_values = np.concatenate([v for v in data_dict.values() if len(v) > 0])
            ymax_data = float(np.max(all_values))
        else:
            ymax_data = 1.0

        log_max = np.log10(ymax_data)
        step = 0.15
        extra_log = line_shift_log10_by_panel.get(panel_id, 0.0)

        for j, comp in enumerate(comps):
            comp_name = comp["groups"]
            x1, x2 = label_map[comp_name]

            y_bottom = 10 ** (log_max + (j + 1) * step + extra_log)
            y_top = y_bottom * 1.05

            ax.plot([x1, x1, x2, x2], [y_bottom, y_top, y_top, y_bottom],
                    lw=0.8, c='black', zorder=3)

            label = p_to_asterisk(comp['mw_p_FDR'])
            fs = 6 if label == "ns" else 9

            if label == "ns":
                y_text = y_top * 1.10
            else:
                y_text = y_top

            ax.text((x1 + x2) / 2, y_text, label,
                    ha='center', va='center',
                    fontsize=fs, zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(groups)
        ax.set_title(f"Age {age}")
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.text(-0.25, 1.05, panel_id, transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='left', va='bottom')

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey='row')

scatter_by_age_on_axes(df, "MT", "RQ ND1", axes[0, :], panel_labels=list("ABCD"))
scatter_by_age_on_axes(df, "TELOMER", "RQ TL", axes[1, :], panel_labels=list("EFGH"))

plt.tight_layout()
plt.savefig("Figure_A_to_H_AgeStratified_ND1_TL.png", dpi=300)
plt.show()
