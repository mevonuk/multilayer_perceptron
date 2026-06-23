from tools.load_data import load
from tools.preprocessing import label_data
from plotting.pairplot import pairplotter
import argparse


def main():
    """Loads data, plots pairplot of chosen features"""
    parser = argparse.ArgumentParser(description="display pairplots of data")

    parser.add_argument(
        "--subset",
        type=str,
        default="mean",
        choices=["mean", "std", "worst"],
        help="Subset of data"
    )

    args = parser.parse_args()

    subset = args.subset

    data = None
    try:
        dataset = "data/data.csv"
        data = load(dataset)
        labels = [
            'ID',
            'diagnosis',
            'radius_mean',
            'radius_std',
            'radius_worst',
            'texture_mean',
            'texture_std',
            'texture_worst',
            'perimeter_mean',
            'perimeter_std',
            'perimeter_worst',
            'area_mean',
            'area_std',
            'area_worst',
            'smoothness_mean',
            'smoothness_std',
            'smoothness_worst',
            'compactness_mean',
            'compactness_std',
            'compactness_worst',
            'concavity_mean',
            'concavity_std',
            'concavity_worst',
            'concave_pts_mean',
            'concave_pts_std',
            'concave_pts_worst',
            'symmetry_mean',
            'symmetry_std',
            'symmetry_worst',
            'fractal_dim_mean',
            'fractal_dim_std',
            'fractal_dim_worst',
        ]
        data = label_data(data, labels)

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
        plot_features = mean_features
        if subset == 'worst': plot_features = worst_features
        if subset == 'std': plot_features = std_features
        pairplotter(data, plot_features, 'diagnosis', save_fig=0)

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
