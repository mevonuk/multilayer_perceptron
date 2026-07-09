import matplotlib.pyplot as plt


def plot2(training, validation, title, save_fig=0):
    """plot training and validation history"""
    fig_title = title + " vs Epoch"
    plt.figure(figsize=(8, 5))

    plt.plot(training, label="Training")
    plt.plot(validation, label="Validation")

    plt.title(fig_title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()

    # Save high-res figure
    if save_fig:
        plt.savefig(fig_title.replace(' ', '_') + ".pdf", dpi=600)
    else:
        plt.show()

def plot_compare(
        validation_gd, validation_adam,
        validation_nest, validation_rms,
        title, save_fig=0):
    """compare validation histories for different optimizers"""
    fig_title = title + " vs Epoch"
    plt.figure(figsize=(8, 5))

    plt.plot(validation_gd, label="Gradient descent")
    plt.plot(validation_nest, label="Nesterov momentum")
    plt.plot(validation_adam, label="Adam")
    plt.plot(validation_rms, label="RMSprop")

    plt.title(fig_title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()

    # Save high-res figure
    if save_fig:
        plt.savefig(fig_title.replace(' ', '_') + ".pdf", dpi=600)
    else:
        plt.show()    
