# plots histograms of data and determines if house scores are homogeneous
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, "../tools")
from load_data import load, label_data
from math_tools import range_of_means, mean_column, std_column


def main():
    """Loads data, displays histograms,
    and calculates the max diff in Means compared to mean Std
    to indicate if distributions are homogeneous"""

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

        # Plot one histogram per subject sequentially
        stats = {}
        for subject in feature_names:
            plt.figure(figsize=(8, 5))
            for house in diagnoses:
                h_data = data_noid[data_noid['diagnosis'] == house]
                scores = h_data[subject].dropna()
                plt.hist(
                    scores,
                    bins=10,
                    alpha=0.6,
                    label=house,
                    color=diagnose_colors[house]
                )
                stats[house] = [mean_column(scores), std_column(scores)]

            # check means and stds for each subject to see if they are similar
            stats_df = pd.DataFrame.from_dict(stats).T
            stats_df.columns = ['Mean', 'Std']
            if range_of_means(stats_df):
                print()
                print(subject)
                print('Range of means is within the average std: HOMOGENOUS')

            plt.title(f'{subject} Score Distribution by diagnosis')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
