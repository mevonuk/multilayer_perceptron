# plots histograms of data and determines if house scores are homogeneous
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../tools")
from load_data import load, label_data
import seaborn as sns


def pairplotter(data, subjects, diagnosis, save_fig=0):
    """displays pairplot"""

    # Names of diagnoses
    diagnose_colors = {
        'M': 'red',
        'B': 'green',
    }

    # Build pairplot for all features
    pairplot = sns.pairplot(
        data[subjects + [diagnosis]],
        hue=diagnosis,
        palette=diagnose_colors,
        plot_kws={"alpha": 0.6, "s": 20},
        height=1.5,
        aspect=1
    )

    # Adjust title
    plt.suptitle('Pairwise Scatterplots of features', y=1.02)

    # Save high-res figure
    if save_fig == 1:
        pairplot.savefig("pairplot_subjects.pdf", dpi=600)

    plt.show()


def main():
    """Loads data, displays histograms,
    and calculates the max diff in Means compared to mean Std
    to indicate if distributions are homogeneous"""

    data = None
    try:
        dataset = "../data/data.csv"
        data = load(dataset)
        data = label_data(data)

        # Identify features, drop ID column
        data_noid = data.drop(columns=['ID'])

        # just the mean features:
        mean_features = [
            'radius_mean',  # good
            'texture_mean',   # good
            'perimeter_mean',  # good
            'area_mean',
            'smoothness_mean',
            'compactness_mean',
            'concavity_mean',
            'concave_pts_mean',
            'symmetry_mean',
            'fractal_dim_mean',  # good
        ]

        # just the std features:
        std_features = [
            'radius_std',  # okay
            'texture_std',
            'perimeter_std',  # good
            'area_std',
            'smoothness_std',  # okay
            'compactness_std',
            'concavity_std',
            'concave_pts_std',  # good
            'symmetry_std',  # good
            'fractal_dim_std',
        ]

        # just the worst features:
        worst_features = [
            'radius_worst',  # good
            'texture_worst',  # okay
            'perimeter_worst',
            'area_worst',
            'smoothness_worst',
            'compactness_worst',
            'concavity_worst',  # good
            'concave_pts_worst',  # good
            'symmetry_worst',  # good
            'fractal_dim_worst',
        ]

        # Build pairplot for all features
        pairplotter(data, worst_features, 'diagnosis', save_fig=1)

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
