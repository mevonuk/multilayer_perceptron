from tools.load_save_data import load_split_data
from plotting.plot_loss import plot2, plot_compare
from tools.MLP import MLP


def train_model(max_epochs, learn_factor, optimizer, hidden_layers, activation, v):
    """load data, set learning rate, normalize data, train, save weights"""
    if v: print("\nLoading the split datasets...")
    X_train, y_train, act = load_split_data(data_type='train')
    if act != activation:
        raise TypeError('saved datasets do not match activation type')
    if v: print("Training data loaded")
    X_test, y_test, act = load_split_data(data_type='test')
    if act != activation:
        raise TypeError('saved datasets do not match activation type')

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
    mp_test = MLP(X_train.shape[1], hidden_layers=hidden_layers, optimizer=optimizer, activation=activation)

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


def training(
        max_epochs, learn_factor,
        optimizer, hidden_layers,
        activation, save_figs, verbose):
    """train according to optimizer type and plot training history"""
    try:

        # make the MLP specifying size of hidden and output layers
        if optimizer in ['gd', 'nest', 'adam', 'rms']:

            print("\nInitializing and training MLP model using", optimizer, "...")

            train_loss, val_loss, train_acc, val_acc = train_model(
                max_epochs, learn_factor, optimizer, hidden_layers, activation, 1)

            plot2(train_loss, val_loss, optimizer + ' Loss', save_fig=save_figs)
            plot2(train_acc, val_acc, optimizer + ' Accuracy', save_fig=save_figs)

        else:

            print("\nInitializing and training MLP models for comparison...")

            print("\nGradient descent...")
            train_loss_gd, val_loss_gd, train_acc_gd, val_acc_gd = train_model(
                max_epochs, learn_factor, 'gd', hidden_layers, activation, 0)
            print("\nNesterov momentum...")
            train_loss_nest, val_loss_nest, train_acc_nest, val_acc_nest = train_model(
                max_epochs, learn_factor, 'nest', hidden_layers, activation, 0)
            print("\nAdam...")
            train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam = train_model(
                max_epochs, learn_factor, 'adam', hidden_layers, activation, 0)
            print("\nRMSprop...")
            train_loss_rms, val_loss_rms, train_acc_rms, val_acc_rms = train_model(
                max_epochs, learn_factor, 'rms', hidden_layers, activation, 0)
            
            plot_compare(
                train_loss_gd, train_loss_adam, train_loss_nest, train_loss_rms,
                'Compare Training Loss', save_fig=save_figs)
            plot_compare(
                val_loss_gd, val_loss_adam, val_loss_nest, val_loss_rms,
                'Compare Validation Loss', save_fig=save_figs)
            plot_compare(
                train_acc_gd, train_acc_adam, train_acc_nest, train_acc_rms,
                'Compare Training accuracy', save_fig=save_figs)
            plot_compare(
                val_acc_gd, val_acc_adam, val_acc_nest, val_acc_rms,
                'Compare Validation accuracy', save_fig=save_figs)


    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)
