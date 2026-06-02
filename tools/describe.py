# routine to display the statistics of a dataset by feature

import sys
from load_data import load, label_data
from math_tools import count_items, mean_column, std_column
from math_tools import min_item, max_item, median_item
from math_tools import first_quartile, third_quartile
from math_tools import skew_column, kurtosis_column
from math_tools import excess_kurtosis
import pandas as pd


def main():
    """Loads data and displays statistics"""

    data = None
    try:
        # check number of arguments
        if len(sys.argv) != 2:
            print("Wrong number of arguments")
            return

        # load dataset
        data = load(sys.argv[1])
        data = label_data(data)

        # select features that are numeric
        numeric_data = data.select_dtypes(include='number')

        # make a dictionary of functions
        functions = {
            'Count': count_items,
            'Mean': mean_column,
            'Std': std_column,
            'Skew': skew_column,
            'Kurtosis': kurtosis_column,
            'Excess kurtosis': excess_kurtosis,
            'Min': min_item,
            '25%': first_quartile,
            '50%': median_item,
            '75%': third_quartile,
            'Max': max_item
        }

        # apply each function to the features
        rows = {}
        for name, func in functions.items():
            rows[name] = numeric_data.apply(func)

        # create a new DataFrame
        result = pd.DataFrame(rows).T

        print(result)

        # check against pandas describe()
        # print(numeric_data.describe())

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
