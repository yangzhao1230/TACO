import numpy as np


def cell_type_specificity(df, on_target_col, off_target_cols, log=False):
    """
    Calculate cell type specificity from predicted or measured output
    """
    # Stack off-target values
    off_target = df[off_target_cols].values.T

    # Multiply the on-target values to get the same shape
    on_target = np.tile(df[on_target_col].values, (off_target.shape[0], 1))

    # Get the log2 fold change between on-target values and each off-target cell type
    if log:
        lfc = np.log2(on_target / off_target)
    else:
        lfc = on_target - off_target

    # Calculate mingap
    mingap = lfc.min(0).tolist()
    return mingap


def target_specificity(df, label_col, label_map, pred_cols, log=False):
    mingap = np.zeros(len(df))
    for i, col in enumerate(pred_cols):
        other_cols = [x for x in pred_cols if x != col]
        mingap_i = cell_type_specificity(
            df, on_target_col=col, off_target_cols=other_cols, log=log
        )
        label_i = label_map[i]
        mingap[df.label == label_i] = np.array(mingap_i)[df.label == label_i]
    return mingap
