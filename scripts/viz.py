import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from mizani.palettes import hue_pal
from plotnine import *
from polygraph.sequence import gc
from polygraph.embedding import differential_analysis
from scipy.stats import mannwhitneyu


def boxplot(df, group_col="Group", value_col="value", fill_col=None):
    if fill_col is None:
        p = ggplot(df, aes(x=group_col, y=value_col))
    else:
        p = ggplot(df, aes(x=group_col, y=value_col, fill=fill_col))
    return (
        p + geom_boxplot(outlier_size=0.1) + theme_classic() + theme(figure_size=(3, 3))
    )


def pointdensityplot(df, true_col="exp", pred_col="pred", corrx=None, corry=None):
    corr = np.round(df[[true_col, pred_col]].corr().iloc[0, 1], 2)
    if corrx is None:
        corrx = df[pred_col].min()
    if corry is None:
        corry = df[pred_col].max()

    return (
        ggplot(df, aes(x=true_col, y=pred_col))
        + geom_pointdensity(size=0.1)
        + geom_abline(aes(intercept=0, slope=1), color="blue")
        + ylab("Predicted activity")
        + xlab("Measured activity")
        + theme_classic()
        + theme(figure_size=(3.5, 3))
        + annotate("text", x=corrx, y=corry, label=f"Pearson rho={corr}", size=9)
    )


def histplot(df, label_col="label"):
    return (
        ggplot(df[label_col].value_counts().reset_index(), aes(x=label_col, y="count"))
        + geom_col()
        + theme_classic()
        + theme(figure_size=(6, 3.5))
    )


def plot_label_freq(df, label_col="label", highlights=None):
    p = (
        histplot(df, label_col=label_col)
        + xlab("Label")
        + ylab("Frequency in regLM training set")
    )
    if highlights is not None:
        for h in highlights:
            idx = np.where(np.sort(df[label_col].unique()) == h)[0][0]
            p = p + geom_rect(
                xmin=idx + 0.5,
                xmax=idx + 1.5,
                ymin=0,
                ymax=df[label_col].value_counts().max(),
                alpha=0.02,
                fill="mistyrose",
            )
    return p


def plot_gc_content(df, group_col="Group", label_col=None):
    df["GC Content"] = gc(df)
    p = boxplot(df, group_col=group_col, value_col="GC Content") + xlab("Method")
    if label_col is not None:
        p = p + facet_wrap(label_col) + theme(figure_size=(5, 3))
    return p


def plot_accuracy_by_label(df, labels=None, label_col="label", acc_col="acc_mean"):
    to_plot = df[[label_col, acc_col]]
    if labels is not None:
        to_plot = to_plot[to_plot.label.isin(labels)]

    return (
        boxplot(to_plot, group_col=label_col, value_col=acc_col)
        + ylab("Nucleotide prediction accuracy")
        + xlab("Label")
    )


def pca_plot(ad, group_col="Group", label_col="label", ref_group="Test Set"):
    # Create dataframe
    to_plot = pd.DataFrame(ad.obsm["X_pca"][:, :2], columns=["PC1", "PC2"])
    to_plot[group_col] = ad.obs[group_col].tolist()
    to_plot[label_col] = ad.obs[label_col].tolist()

    # Calculate variance explained
    pc1_var = np.round(ad.uns["pca"]["variance_ratio"][0] * 100, 2)
    pc2_var = np.round(ad.uns["pca"]["variance_ratio"][1] * 100, 2)

    # Format label and group columns
    groups = to_plot[group_col].unique()
    nonreference_groups = [x for x in groups if x != ref_group]
    to_plot[group_col] = pd.Categorical(
        to_plot[group_col], categories=[ref_group] + nonreference_groups
    )

    # Get colors
    gp_colors = ["lightgray"] + scale_color_hue().palette(len(nonreference_groups))

    # Make plot labeled by group
    return (
        ggplot(to_plot, aes(x="PC1", y="PC2", color=group_col, alpha=group_col))
        + scale_alpha_manual(values=[0.1] + [1] * len(nonreference_groups))
        + geom_jitter(size=0.1)
        + scale_color_manual(values=gp_colors)
        + xlab(f"PC1 ({pc1_var}%)")
        + ylab(f"PC2 ({pc2_var}%)")
        + theme_classic()
        + theme(figure_size=(5, 3.5))
        + guides(color=guide_legend(override_aes={"size": 3}))
    )


def umap_plot(
    ad, group_col="Group", label_col="label", filter_labels=None, ref_group="Test Set"
):
    # Create dataframe
    to_plot = pd.DataFrame(ad.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    to_plot[group_col] = ad.obs[group_col].tolist()
    to_plot[label_col] = ad.obs[label_col].tolist()

    # Format label and group columns
    groups = to_plot[group_col].unique()
    syn_groups = [x for x in groups if x != ref_group]
    to_plot[group_col] = pd.Categorical(
        to_plot[group_col], categories=syn_groups + ["Test Set"]
    )
    if filter_labels is not None:
        to_plot.loc[~to_plot[label_col].isin(filter_labels), label_col] = "other"

    # Make plot labeled by group
    group_plot = (
        ggplot(to_plot, aes(x="UMAP1", y="UMAP2", color=group_col))
        + geom_point(size=0.6)
        + scale_color_manual(values=hue_pal()(len(syn_groups)) + ["lightgray"])
        + theme_classic()
        + theme(figure_size=(5, 3.5))
        + guides(color=guide_legend(override_aes={"size": 3}))
    )
    label_plot = (
        ggplot(to_plot, aes(x="UMAP1", y="UMAP2", color=label_col))
        + geom_point(size=0.6)
        + scale_color_manual(
            values=hue_pal()(len(to_plot[label_col].unique()) - 1) + ["lightgray"]
        )
        + theme_classic()
        + theme(figure_size=(5, 3.5))
        + guides(color=guide_legend(override_aes={"size": 3}))
    )
    return group_plot, label_plot


def cluster_dist(ad, group_col="Group", ref_group="Test Set", **kwargs):
    # cluster
    sc.tl.leiden(ad, **kwargs)

    # Make UMAP plot of clusters
    umap = pd.DataFrame(ad.obsm["X_umap"])
    umap.columns = ["UMAP1", "UMAP2"]
    umap["leiden"] = ad.obs.leiden.astype(str).tolist()
    n_clusters = int(ad.obs.leiden.cat.categories[-1]) + 1
    umap["leiden"] = pd.Categorical(
        umap["leiden"], categories=[str(x) for x in range(n_clusters)]
    )
    umap_plot = (
        ggplot(umap, aes(x="UMAP1", y="UMAP2", color="leiden"))
        + geom_point(size=0.2)
        + theme_classic()
        + theme(figure_size=(4.5, 3.5))
        + guides(color=guide_legend(override_aes={"size": 2.5}))
    )

    # Calculate fraction of each group in each cluster
    df = ad.obs[[group_col, "leiden"]].value_counts().reset_index()
    df = df.pivot_table(index="leiden", columns=group_col, values="count").fillna(0)
    df = df.div(df.sum(axis=0), axis=1)
    df = df.reset_index().melt(id_vars="leiden")

    # Plot stacked barplot
    cluster_frac_plot = (
        ggplot(df, aes(fill="leiden", y="value", x=group_col))
        + geom_col(color="black")
        + ylab("Fraction in cluster")
        + theme_classic()
        + theme(figure_size=(3.5, 3.5))
        + guides(fill=guide_legend(override_aes={"size": 1}))
    )

    # Calculate heatmap of features enriched in each cluster
    ad = differential_analysis(ad, group_col="leiden", reference_group="rest")
    sig = ad.uns["DE_test"][ad.uns["DE_test"].padj < 0.05].sort_values('log2FC')
    heatmap_motifs = set(sig.groupby('leiden').tail(4)['value'].tolist())
    df = ad.uns["DE_test"][ad.uns["DE_test"]["value"].isin(heatmap_motifs)]

    heatmap = sns.clustermap(
        df.pivot_table(index="value", columns="leiden", values="log2FC"),
        figsize=(6, 11),
        center=0,
        vmin=-8,
        vmax=8,
        cmap="bwr",
    )

    return umap_plot, cluster_frac_plot, heatmap


def plot_motif_freq_by_label(
    ad,
    motifs,
    labels=None,
    label_col="label",
    label_map=None,
):
    df = pd.DataFrame({"Motif": ad.var_names})

    if labels is None:
        labels = ad.obs[label_col].unique()
    for l in labels:
        df[l] = (ad[ad.obs[label_col] == l].X > 0).mean(0)

    if label_map is not None:
        label_map["Motif"] = "Motif"
        df.columns = df.columns.map(label_map)

    if isinstance(motifs, dict):
        df = df[df.Motif.isin(np.concatenate([v for k, v in motifs.items()]))].copy()
        for k, v in motifs.items():
            df.loc[df.Motif.isin(v), "Category"] = k
    else:
        df = df[df.Motif.isin(motifs)]

    df.Motif = df.Motif.apply(lambda x: x.split('.')[-1])

    if isinstance(motifs, dict):
        p = ggplot(
            df.melt(id_vars=["Motif", "Category"], var_name=label_col),
            aes(x="Motif", fill=label_col, y="value"),
        ) + facet_wrap("Category", scales="free")
    else:
        p = ggplot(
            df.melt(id_vars="Motif", var_name=label_col),
            aes(x="Motif", fill=label_col, y="value"),
        )

    return (
        p
        + geom_col(stat="identity", position="dodge")
        + ylab("Fraction with motif")
        + theme_classic()
        + theme(figure_size=(10, 3))
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )


def plot_motif_pos(sites, motifs):
    return (
        ggplot(sites[sites.Matrix_id.isin(motifs)], aes(x="pos", color="Group"))
        + geom_density()
        + facet_wrap("Matrix_id", ncol=3, scales="free_y")
        + xlab("Start position")
        + theme_classic()
        + theme(figure_size=(5.5, 3))
    )


def plot_directed_evolution(df, pred_cols, var_name):
    to_plot = df[["seq", "iter"] + pred_cols]
    to_plot = to_plot.melt(id_vars=["seq", "iter"], var_name=var_name)
    (
        ggplot(to_plot, aes(x="iter", y="value", fill=var_name))
        + geom_boxplot()
        + xlab("Iteration")
        + ylab("Predicted Expression")
    )


def plot_exp_match(gen, matched):
    return (
        ggplot(
            pd.concat([gen, matched]).melt(id_vars=["Sequence", "Group"]),
            aes(x="variable", y="Predicted Expression", fill="Group"),
        )
        + geom_boxplot()
    )


def plot_training_curve(log):
    curve = pd.read_csv(log)

    return (
        ggplot(
            curve[["step", "train_loss_step", "val_loss"]]
            .rename(columns={"train_loss_step": "Training", "val_loss": "Validation"})
            .melt(id_vars="step", var_name="Loss")
            .dropna(),
            aes(x="step", y="value", color="Loss"),
        )
        + geom_line()
        + scale_y_log10()
        + xlab("Step")
        + ylab("Cross-entropy loss")
        + theme_classic()
        + theme(figure_size=(7, 4))
    )


def plot_expression_by_label(
    df,
    var_name="variable",
    label_col="label",
    seq_col="Sequence",
    var_map=None,
    label_map=None,
    label_name="Label",
    group_col=None,
):
    if group_col is None:
        to_plot = df.melt(id_vars=[seq_col, label_col], var_name=var_name)
    else:
        to_plot = df.melt(id_vars=[seq_col, label_col, group_col], var_name=var_name)
    to_plot[label_name] = to_plot.label.map(label_map)
    to_plot[var_name] = to_plot[var_name].map(var_map)
    if group_col is None:
        p = ggplot(to_plot, aes(x=var_name, y="value"))
    else:
        p = ggplot(to_plot, aes(x=var_name, y="value", fill=group_col))
    return (
        p
        + facet_wrap(label_name)
        + geom_boxplot(outlier_size=0.1)
        + theme_classic()
        + theme(figure_size=(6, 3))
        + ylab("Predicted Expression")
    )


def plot_accuracy_by_motif(df, labels=None):
    to_plot = df[["label", "In Motif", "Out of Motif"]].dropna().copy()
    if labels is not None:
        to_plot = to_plot[to_plot.label.isin(labels)]
    maxv = to_plot.iloc[:, 1:].values.max()
    return (
        ggplot(
            to_plot.melt(id_vars="label"), aes(fill="variable", y="value", x="label")
        )
        + geom_boxplot(outlier_size=0.1)
        + theme_classic()
        + theme(figure_size=(4, 3))
        + ylab("Nucleotide Prediction Accuracy")
        + theme(axis_title_x=element_blank())
    )


def plot_n_motifs_by_label(seqs, ad, labels=None):
    ad.obs["n_motifs"] = ad.X.sum(1)
    to_plot = ad.obs.copy()
    if labels is not None:
        to_plot = to_plot[to_plot["label"].isin(labels)]
    return (
        ggplot(to_plot, aes(x="label", y="n_motifs"))
        + geom_boxplot(outlier_size=0.1)
        + theme_classic()
        + theme(figure_size=(3, 3))
        + ylab("Number of motifs")
        + xlab("Label")
    )


def plot_accuracy_with_shuffle(seqs, labels=None):
    to_plot = seqs[["label", "acc_mean", "acc_shuf_mean"]].copy()
    if labels is not None:
        to_plot = to_plot[to_plot.label.isin(labels)].copy()
        
    to_plot = to_plot.melt(id_vars="label", var_name="Data")
    to_plot["Data"] = to_plot["Data"].map(
        {"acc_mean": "True Labels", "acc_shuf_mean": "Shuffled Labels"}
    )
    to_plot["Data"] = pd.Categorical(
        to_plot["Data"], categories=["True Labels", "Shuffled Labels"]
    )
    return (
        ggplot(to_plot, aes(x="label", y="value", fill="Data"))
        + geom_boxplot(outlier_size=0.1)
        + theme_classic()
        + theme(figure_size=(4, 3))
        + xlab("True Label")
        + ylab("Nucleotide Prediction Accuracy")
    )


def plot_likelihood_ratio(df, ratio_label="44/00"):
    return (
        ggplot(df, aes(x="Motif", y="LL_ratio"))
        + geom_boxplot(outlier_size=0.1)
        + geom_hline(yintercept=0, linetype="dashed")
        + ylab(f"Log-likelihood\nratio ({ratio_label})")
        + theme_classic()
        + theme(figure_size=(5, 2.5))
        + facet_wrap("Category", scales="free", ncol=2)
        + theme(axis_text_x=element_text(rotation=50, hjust=1))
        + theme(axis_title_x=element_blank())
    )


def plot_attributions(attrs, start_pos=0, end_pos=-1, figsize=(20, 2),
    ticks=20,
    highlight_pos=[],
    highlight_width=[],
    ylim=None,
    annotations=None
):

    import logomaker
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    attrs = attrs.squeeze()[start_pos:end_pos, :]
    df = pd.DataFrame(attrs, columns=["A", "C", "G", "T"])
    df.index.name = "pos"
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    #ax.xaxis.set_ticks(np.arange(1, len(attrs) + 1, ticks))
    #ax.set_xticklabels(range(start_pos, len(attrs) + start_pos + 1, ticks))

    for pos, w in zip(highlight_pos, highlight_width):
        ax.add_patch(
            Rectangle(
                xy=[pos - start_pos - (w // 2), -10],
                width=w,
                height=20,
                facecolor=(1, 1, 0, 0.15),
            )
        )

    if annotations is not None:
        for k, v in annotations.items():
            ax.annotate(k, v)

    logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)

    if ylim is not None:
        logo.ax.set_ylim(ylim)
    logo.ax.set_ylabel('Importance Score')

    return logo
