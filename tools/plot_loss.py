import matplotlib.pyplot as plt


def plot_loss(training_loss):
    """plot the loss history"""
    plt.plot(training_loss)
    plt.title("Loss Function vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


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


def plot_metrics(accuracy, percision, recall, F1, title):
    """plot metrics history"""
    plt.figure(figsize=(8, 5))

    plt.plot(accuracy, label="accuracy")
    plt.plot(percision, label="precision")
    plt.plot(recall, label="recall")
    plt.plot(F1, label="F1")

    plt.title(title + " vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()

    plt.show()
