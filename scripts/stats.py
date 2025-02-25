import numpy as np
import pandas as pd
import scipy.stats


def groupwise_chi_squared(
    df, value_col, group_col="Group", ref_group="Test Set"
):
    # Get groups
    groups = np.sort(df[group_col].unique())
    nonreference_groups = [x for x in groups if x != ref_group]

    # Get groupwise count table
    freqs = df.groupby(group_col)[value_col].value_counts().unstack()
    freqs = {c: freqs.loc[c].values for c in freqs.index}

    # Calculate proportions in reference group
    ref_prop = freqs[ref_group] / sum(freqs[ref_group])

    # Test for equal proportions in non-reference groups
    for group in nonreference_groups:
        pval = scipy.stats.chisquare(
            freqs[group], f_exp=ref_prop * sum(freqs[group])
        ).pvalue
        print(
            f"Chi-squared p-value for {value_col} in group {group} vs. {ref_group}: {pval:2g}"
        )

