from tools.load_data import load, label_data


def main():

    data = None
    try:
        dataset = "data/data.csv"

        # load dataset
        data = load(dataset)
        # label columns
        data = label_data(data)

        print(data)


  

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == "__main__":
    main()
