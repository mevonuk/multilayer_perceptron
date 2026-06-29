import matplotlib.pyplot as plt


def plot2(training, validation, title):
    """plot training and validation history"""
    plt.figure(figsize=(8, 5))

    plt.plot(training, label="Training")
    plt.plot(validation, label="Validation")

    plt.title(title + " vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()

    plt.show()


def plot_compare(
        validation_gd, validation_adam,
        validation_nest, validation_rms,
        title):
    """compare validation histories for different optimizers"""
    plt.figure(figsize=(8, 5))

    plt.plot(validation_gd, label="Gradient descent")
    plt.plot(validation_nest, label="Nesterov momentum")
    plt.plot(validation_adam, label="Adam")
    plt.plot(validation_rms, label="RMSprop")

    plt.title(title + " vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()

    plt.show()
