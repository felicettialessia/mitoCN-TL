import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq


def read_metadata(file_path: str) -> pd.DataFrame:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        comment="#",
        dtype=str
    )

    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def anova_power_from_cohens_f(
    effect_size_f: float,
    n_total: int,
    k_groups: int,
    alpha: float
) -> float:

    df1 = k_groups - 1
    df2 = n_total - k_groups

    if df2 <= 0:
        raise ValueError("Total sample size is too small relative to the number of groups.")

    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    noncentrality = n_total * effect_size_f**2

    power = stats.ncf.sf(f_crit, df1, df2, noncentrality)
    return power


def solve_mde_anova(
    n_total: int,
    k_groups: int,
    alpha: float,
    target_power: float
) -> float:

    def objective(f):
        return anova_power_from_cohens_f(
            f,
            n_total,
            k_groups,
            alpha
        ) - target_power

    lower = 1e-8
    upper = 5.0

    while objective(upper) < 0:
        upper *= 2

    return brentq(objective, lower, upper)


def eta_squared_from_cohens_f(f: float) -> float:

    return f**2 / (1 + f**2)


def pairwise_power_from_cohens_d(
    d: float,
    n1: int,
    n2: int,
    alpha: float,
    alternative: str = "two-sided",
) -> float:

    df = n1 + n2 - 2

    if df <= 0:
        raise ValueError("Sample sizes are insufficient for the t-test.")

    noncentrality = d / np.sqrt(1 / n1 + 1 / n2)

    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        power = stats.nct.sf(t_crit, df, noncentrality) + stats.nct.cdf(
            -t_crit, df, noncentrality
        )

    elif alternative == "larger":
        t_crit = stats.t.ppf(1 - alpha, df)
        power = stats.nct.sf(t_crit, df, noncentrality)

    elif alternative == "smaller":
        t_crit = stats.t.ppf(alpha, df)
        power = stats.nct.cdf(t_crit, df, noncentrality)

    else:
        raise ValueError(
            "alternative must be one of: two-sided, larger, or smaller"
        )

    return power


def solve_mde_pairwise(
    n1: int,
    n2: int,
    alpha: float,
    target_power: float,
    alternative: str = "two-sided",
) -> float:

    def objective(d):
        return pairwise_power_from_cohens_d(
            d=d,
            n1=n1,
            n2=n2,
            alpha=alpha,
            alternative=alternative,
        ) - target_power

    lower = 1e-8
    upper = 5.0

    while objective(upper) < 0:
        upper *= 2

    return brentq(objective, lower, upper)


def cohen_f_interpretation(f: float) -> str:

    if f < 0.10:
        return "below small"
    elif f < 0.25:
        return "small"
    elif f < 0.40:
        return "medium"
    else:
        return "large"


def cohen_d_interpretation(d: float) -> str:

    if d < 0.20:
        return "below small"
    elif d < 0.50:
        return "small"
    elif d < 0.80:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser(
        description="Power analysis and minimum detectable effect size by Biological Group."
    )

    parser.add_argument(
        "input_file",
        help="Metadata file containing the Biological Group column.",
    )

    parser.add_argument(
        "--group-col",
        default="Biological Group",
        help='Name of the column containing the biological groups. Default: "Biological Group".',
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level. Default: 0.05.",
    )

    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Desired power. Default: 0.80.",
    )

    parser.add_argument(
        "--no-pairwise",
        action="store_true",
        help="Disable pairwise MDE calculations.",
    )

    args = parser.parse_args()

    df = read_metadata(args.input_file)

    if args.group_col not in df.columns:
        raise ValueError(
            f'Column "{args.group_col}" not found. '
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.group_col]).copy()

    group_counts = (
        df[args.group_col]
        .astype(str)
        .str.strip()
        .value_counts()
        .sort_index()
    )

    k_groups = group_counts.shape[0]
    n_total = int(group_counts.sum())

    if k_groups < 2:
        raise ValueError("At least two biological groups are required.")

    print("\n==============================")
    print("POWER ANALYSIS BY BIOLOGICAL GROUP")
    print("==============================")

    print("\nIdentified groups:")
    for group, n in group_counts.items():
        print(f"  {group}: n = {n}")

    print(f"\nTotal N = {n_total}")
    print(f"Number of groups = {k_groups}")
    print(f"Alpha = {args.alpha}")
    print(f"Desired power = {args.power}")

    mde_f = solve_mde_anova(
        n_total=n_total,
        k_groups=k_groups,
        alpha=args.alpha,
        target_power=args.power,
    )

    mde_eta2 = eta_squared_from_cohens_f(mde_f)

    print("\n------------------------------")
    print("OVERALL COMPARISON BETWEEN GROUPS")
    print("------------------------------")
    print("Test considered: one-way omnibus ANOVA")
    print(f"Minimum Detectable Effect Size, Cohen's f = {mde_f:.3f}")
    print(f"Equivalent eta-squared = {mde_eta2:.3f}")
    print(f"Conventional interpretation = {cohen_f_interpretation(mde_f)}")

    achieved_power = anova_power_from_cohens_f(
        effect_size_f=mde_f,
        n_total=n_total,
        k_groups=k_groups,
        alpha=args.alpha,
    )

    print(f"Verified power at the MDE = {achieved_power:.3f}")

    if not args.no_pairwise and k_groups > 1:
        print("\n------------------------------")
        print("OPTIONAL PAIRWISE COMPARISONS")
        print("------------------------------")
        print("Test considered: independent two-sample t-test")
        print("Effect size: Cohen's d")
        print(
            "Note: these comparisons are descriptive/post-hoc "
            "relative to the overall test.\n"
        )

        groups = list(group_counts.index)
        n_comparisons = int(k_groups * (k_groups - 1) / 2)
        alpha_bonf = args.alpha / n_comparisons

        print(f"Number of pairwise comparisons = {n_comparisons}")
        print(f"Uncorrected alpha = {args.alpha}")
        print(f"Bonferroni alpha = {alpha_bonf:.5f}\n")

        header = (
            f"{'Comparison':<20}"
            f"{'n1':>6}"
            f"{'n2':>6}"
            f"{'MDE d':>12}"
            f"{'Interp.':>14}"
            f"{'MDE d Bonf.':>16}"
            f"{'Interp. Bonf.':>16}"
        )
        print(header)
        print("-" * len(header))

        for g1, g2 in itertools.combinations(groups, 2):
            n1 = int(group_counts[g1])
            n2 = int(group_counts[g2])

            mde_d = solve_mde_pairwise(
                n1=n1,
                n2=n2,
                alpha=args.alpha,
                target_power=args.power,
                alternative="two-sided",
            )

            mde_d_bonf = solve_mde_pairwise(
                n1=n1,
                n2=n2,
                alpha=alpha_bonf,
                target_power=args.power,
                alternative="two-sided",
            )

            comparison = f"{g1} vs {g2}"

            print(
                f"{comparison:<20}"
                f"{n1:>6}"
                f"{n2:>6}"
                f"{mde_d:>12.3f}"
                f"{cohen_d_interpretation(mde_d):>14}"
                f"{mde_d_bonf:>16.3f}"
                f"{cohen_d_interpretation(mde_d_bonf):>16}"
            )

    print("\n==============================")
    print("End of analysis")
    print("==============================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
