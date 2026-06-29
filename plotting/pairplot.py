import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_color_mapping(df: pd.DataFrame, col: str) -> dict:
    """unique color mapping only good to 10 categories"""
    colors = [
        'red', 'green', 'blue', 'orange', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan'
    ]

    categories = sorted(df[col].dropna().unique())
    if len(categories) > len(colors):
        print("Warning: color mapping is not unique")

    return {
        category: colors[i % len(colors)]
        for i, category in enumerate(categories)
    }


def pairplotter(
    data: pd.DataFrame,
    features: list,
    versus: str,
    save_fig=False
    ):
    """displays pairplot of a list of features
    saves as PDF if specified"""

    color_map = create_color_mapping(data, versus)

    # Build pairplot for all features
    pairplot = sns.pairplot(
        data[features + [versus]],
        hue=versus,
        palette=color_map,
        plot_kws={"alpha": 0.6, "s": 20},
        height=1.5,
        aspect=1
    )

    # Adjust title
    pairplot.fig.subplots_adjust(top=0.95)
    pairplot.fig.suptitle('Pairwise Scatterplots of features', y=0.98)

    # Save high-res figure
    if save_fig:
        pairplot.savefig("pairplot_features.pdf", dpi=600)

    plt.show()
