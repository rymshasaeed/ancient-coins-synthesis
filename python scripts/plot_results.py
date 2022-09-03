import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_images(images):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images.detach(), nrow=12).permute(1, 2, 0))

def show_batch(dl):
    for images, _ in dl:
        show_images(images)
        break

def performance(H):
    fig = plt.figure(figsize = (8, 6))
    plt.plot(H["D_losses"], label="discriminator_loss")
    plt.plot(H["G_losses"], label="generator_loss")
    plt.title("Model Performance")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('../results/performance.png', dpi=1000, bbox_inches='tight')
    plt.show()