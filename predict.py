from tools.load_save_data import load_split_data
from tools.math_tools import my_metrics, binary_cross_entropy
from tools.MLP_momentum import MLP_momentum


def predict(optimizer, X, y, verbose):
        mp_test = MLP_momentum(X.shape[1], optimizer=optimizer)

        # load weights
        if verbose: print("Loading saved weights...")
        mp_test.load_weights()

        # make a prediction using the trained weights and the test data
        if verbose: print("Making prediction...")
        output_prediction, output_prob = mp_test.predict(X)

        loss = binary_cross_entropy(y.to_numpy(), output_prob)

        # compare prediction to real values
        accuracy, precision, recall, F1 = my_metrics(
            output_prediction, y.to_numpy())

        return accuracy, precision, recall, F1, loss


def predicting(optimizer, verbose):

    if verbose: print("\nLoading the split datasets...")
    X_test, y_test = load_split_data(data_type='test')
    if verbose: print("Test data loaded")

    print("\nPredicting using test dataset...")
    try:
        # make the MLP specifying size of hidden and output layers
        if optimizer == 'compare':
            accuracy_gd, precision_gd, recall_gd, F1_gd, loss_gd = predict(
                 'gd', X_test, y_test, verbose)
            accuracy_nest, precision_nest, recall_nest, F1_nest, loss_nest = predict(
                 'nest', X_test, y_test, verbose)
            accuracy_adam, precision_adam, recall_adam, F1_adam, loss_adam = predict(
                 'adam', X_test, y_test, verbose)
            accuracy_rms, precision_rms, recall_rms, F1_rms, loss_rms = predict(
                 'rms', X_test, y_test, verbose)

            print("\nComparison of predictions")
            print("               GD     Nest     Adam  RMSprop")
            print(f'Accuracy:  {accuracy_gd:.4f}   {accuracy_nest:.4f}   {accuracy_adam:.4f}   {accuracy_rms:.4f}')
            print(f'Precision: {precision_gd:.4f}   {precision_nest:.4f}   {precision_adam:.4f}   {precision_rms:.4f}')
            print(f'Recall:    {recall_gd:.4f}   {recall_nest:.4f}   {recall_adam:.4f}   {recall_rms:.4f}')
            print(f'F1:        {F1_gd:.4f}   {F1_nest:.4f}   {F1_adam:.4f}   {F1_rms:.4f}')
            print(f'Loss:      {loss_gd:.4f}   {loss_nest:.4f}   {loss_adam:.4f}   {loss_rms:.4f}')
            
        else:
            accuracy, precision, recall, F1, loss = predict(
                 optimizer, X_test, y_test, verbose)

            # compare prediction to real values
            print("\nMetrics of the final", optimizer, "prediction:")
            print(f'Accuracy:  {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall:    {recall:.4f}')
            print(f'F1:        {F1:.4f}')
            print(f'Loss:      {loss:.4f}')

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)
