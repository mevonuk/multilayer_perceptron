from tools.load_save_data import load_split_data
from tools.plot_loss import plot2, plot_compare
from tools.MLP_momentum import MLP_momentum


def train_model(max_epochs, learn_factor, optimizer, hidden_size, v):
    """load data, set learning rate, normalize data, train, save weights"""
    if v: print("\nLoading the split datasets...")
    X_train, y_train = load_split_data(data_type='train')
    if v: print("Training data loaded")
    X_test, y_test = load_split_data(data_type='test')

    # default values of the learning rate
    if optimizer == 'gd':
        learning_rate = 0.01
    elif optimizer == 'nest':
        learning_rate = 0.01
    elif optimizer == 'adam':
        learning_rate = 0.001
    elif optimizer == 'rms':
        learning_rate = 0.001
    
    # scaling of the learning rate by the learn_factor
    learning_rate = learning_rate * learn_factor

    if v:
        print("Validation data loaded")
        print("\nStarting training of model...")
        print("maximum number of epochs:", max_epochs)
    print("Learning rate:", learning_rate)

    # make the MLP specifying size of hidden and output layers
    mp_test = MLP_momentum(X_train.shape[1], hidden_size, 1, optimizer)

    # feature normalization
    if v: print("Normalizing training set...")
    X_train_norm = mp_test.normalize_data(X_train, set_norm=True)

    y_validation = y_test.copy()
    # normalize validation data using norm values of train set
    X_validation_norm = mp_test.normalize_data(X_test, set_norm=False)

    # train while tracking performance on validation set
    train_loss, val_loss, train_acc, val_acc = mp_test.train_with_validation(
        X_train_norm.to_numpy(), y_train.to_numpy(),
        X_validation_norm.to_numpy(), y_validation.to_numpy(),
        max_epochs, learning_rate, v=v)

    # save the weights and biases
    mp_test.save_weights()
    if v: print("\ntraining weights and normalization saved.")

    return train_loss, val_loss, train_acc, val_acc


def training(max_epochs, learn_factor, optimizer, hidden_size, verbose):
    """train according to optimizer type and plot training history"""
    try:

        # make the MLP specifying size of hidden and output layers
        if optimizer in ['gd', 'nest', 'adam', 'rms']:

            print("\nInitializing and training MLP model using", optimizer, "...")

            train_loss, val_loss, train_acc, val_acc = train_model(
                max_epochs, learn_factor, optimizer, hidden_size, 1)

            plot2(train_loss, val_loss, 'Loss')
            plot2(train_acc, val_acc, 'Accuracy')

        else:

            print("\nInitializing and training MLP models for comparison...")

            print("\nGradient descent...")
            train_loss_gd, val_loss_gd, train_acc_gd, val_acc_gd = train_model(
                max_epochs, learn_factor, 'gd', hidden_size, 0)
            print("\nNesterov momentum...")
            train_loss_nest, val_loss_nest, train_acc_nest, val_acc_nest = train_model(
                max_epochs, learn_factor, 'nest', hidden_size, 0)
            print("\nAdam...")
            train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam = train_model(
                max_epochs, learn_factor, 'adam', hidden_size, 0)
            print("\nRMSprop...")
            train_loss_rms, val_loss_rms, train_acc_rms, val_acc_rms = train_model(
                max_epochs, learn_factor, 'rms', hidden_size, 0)
            
            plot_compare(
                train_loss_gd, train_loss_adam, train_loss_nest, train_loss_rms,
                'Training Loss')
            plot_compare(
                val_loss_gd, val_loss_adam, val_loss_nest, val_loss_rms,
                'Validation Loss')
            plot_compare(
                train_acc_gd, train_acc_adam, train_acc_nest, train_acc_rms,
                'Training accuracy')
            plot_compare(
                val_acc_gd, val_acc_adam, val_acc_nest, val_acc_rms,
                'Validation accuracy')


    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)
