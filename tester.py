from tools.load_data import load
from tools.split import split_data
from tools.preprocessing import hot_code, label_data
from tools.preprocessing import normalized_features
from plotting.pairplot import pairplotter
from model_tools.MLP import MLP, binary_cross_entropy
import numpy as np


def main():

    data = None
    try:
        dataset = "data/data.csv"

        # load dataset
        data = load(dataset)
        # label columns
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

        # chosen features
        features = [
            'radius_mean',
            'texture_mean',
            'perimeter_mean',
            'fractal_dim_mean',
            'radius_std',  # okay
            'perimeter_std',  # good
            'concave_pts_std',  # good
            'symmetry_std',  # good
            'radius_worst',  # good
            'texture_worst',  # okay
            'concavity_worst',  # good
            'concave_pts_worst',  # good
            'symmetry_worst',  # good
        ]

        # one-hot code the diagnosis
        diagnoses = ['M', 'B']
        data = hot_code(data, 'M', 'diagnosis', 'one_hot')

        # feature normalization
        data, new_feature_names, f_means, f_stds = normalized_features(
            data, features)

        # plot normalized features if necessary
        # pairplotter(data, new_feature_names, 'diagnosis')

        # extracting X and y data arrays
        # y is the diagnosis one-hot coded
        # X contains the normalized features: chosen based on the graph analysis
        X = data.loc[:, data.columns.intersection(new_feature_names)]
        y = data.loc[:, data.columns.intersection(['one_hot'])]

        # Split the dataset into test and train sets
        X_train, X_test, y_train, y_test = split_data(X, y, random_seed=42)

        mp_test = MLP(len(new_feature_names), 10, 1)

        epochs = 10000
        learning_rate = 0.0001
        mp_test.train(X_train.to_numpy(), y_train.to_numpy(), epochs, learning_rate)


        output_prediction = mp_test.predict(X_test.to_numpy())

        df = output_prediction - y_test
        print(df.value_counts())

        # compare output to real
        y_real = y_test.to_numpy()
        print(binary_cross_entropy(y_real, output_prediction))
  

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
