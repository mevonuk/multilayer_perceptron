import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../tools")
from load_data import load, label_data
import itertools

def main():

    data = None
    try:
        dataset = "../data/data.csv"
        data = load(dataset)
        data = label_data(data)

        diagnoses = ['M', 'B']
        diagnose_colors = {
            'M': 'red',
            'B': 'green',
        }

        # Identify features, drop ID column
        data_noid = data.drop(columns=['ID'])
        # grab only numeric features (should be all but diagnosis)
        feature_names = data_noid.select_dtypes(include='number').columns

        # just the mean features:
        mean_features = [
            'radius_mean',
            'texture_mean',
            'perimeter_mean',
            'area_mean',  # not great
            'smoothness_mean',
            'compactness_mean',  # not great
            'concavity_mean',  # not great
            'concave_pts_mean', # not great
            'symmetry_mean',
            'fractal_dim_mean',
        ]

        # Generate all combinations of features
        # feature_combinations = list(itertools.combinations(feature_names, 2))
        feature_combinations = list(itertools.combinations(mean_features, 2))

        # loop over feature combinations
        for (feature1, feature2) in feature_combinations:

            # Plot all pairs of subjects
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(6, 6))

            # Scatterplot
            for diagnose in diagnoses:
                h_data = data_noid[data_noid['diagnosis'] == diagnose]

                ax.scatter(
                    h_data[feature1],
                    h_data[feature2],
                    label=f"{diagnose}",
                    color=diagnose_colors[diagnose], alpha=0.7
                )
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_title(f"{feature1} vs {feature2}")
            ax.legend(title="diagnosis")

            plt.show()

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
