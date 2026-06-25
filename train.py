from tools.load_save_data import load_split_data
from tools.plot_loss import plot2, plot_compare
from tools.MLP_momentum import MLP_momentum


def train_model(max_epochs, learn_rate, optimizer, hidden_size, v):

    if v: print("\nLoading the split datasets...")
    X_train, y_train = load_split_data(data_type='train')
    if v: print("Training data loaded")
    X_test, y_test = load_split_data(data_type='test')
    if v:
        print("Validation data loaded")
        print("\nStarting training of model...")
        print("maximum number of epochs:", max_epochs)
        print("Learning rate:", learn_rate)

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
        max_epochs, learn_rate, v=v)

    # save the weights and biases
    mp_test.save_weights()
    if v: print("\ntraining weights and normalization saved.")

    return train_loss, val_loss, train_acc, val_acc


def training(max_epochs, learn_rate, optimizer, hidden_size, verbose):

    try:

        # make the MLP specifying size of hidden and output layers
        if optimizer in ['gd', 'nest', 'adam']:

            print("initializing and training MLP model using", optimizer, "...")

            train_loss, val_loss, train_acc, val_acc = train_model(
                max_epochs, learn_rate, optimizer, hidden_size, 1)

            plot2(train_loss, val_loss, 'Loss')
            plot2(train_acc, val_acc, 'Accuracy')

        else:

            print("initializing and training MLP models for comparison...")

            print("\ngradient descent")
            train_loss_gd, val_loss_gd, train_acc_gd, val_acc_gd = train_model(
                max_epochs, learn_rate, 'gd', hidden_size, 0)
            print("\nNesterov momentum")
            train_loss_nest, val_loss_nest, train_acc_nest, val_acc_nest = train_model(
                max_epochs, learn_rate, 'nest', hidden_size, 0)
            print("\nAdam")
            train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam = train_model(
                max_epochs, learn_rate, 'adam', hidden_size, 0)
            
            plot_compare(train_loss_gd, train_loss_adam, train_loss_nest, 'Training Loss')
            plot_compare(val_loss_gd, val_loss_adam, val_loss_nest, 'Validation Loss')
            plot_compare(train_acc_gd, train_acc_adam, train_acc_nest, 'Training accuracy')
            plot_compare(val_acc_gd, val_acc_adam, val_acc_nest, 'Validation accuracy')


    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)



