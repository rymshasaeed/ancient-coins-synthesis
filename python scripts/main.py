import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from coins_dataloader import Coins_Dataloader
from plot_results import show_batch, performance
from cgan_model import Generator, Discriminator, weights_init

# Parameters
num_epochs = 5
latent_dim = 100
batch_size = 128
learning_rate = 0.0002

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transformations
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor()])

# Create training set and load it as torch tensors
TrainDS = Coins_Dataloader(csv_file='../data/Train.csv', transforms=transform)
train_loader = DataLoader(TrainDS,
                          shuffle = True,
                          batch_size = batch_size)

# Initialize the cgan network
G = Generator().to(device)
G.apply(weights_init)
D = Discriminator().to(device)
D.apply(weights_init)

# Binary Cross Entropy loss
loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Initialize a varaible to store training history
training_history = {'D_losses': [],
                    'G_losses': [],
                    'per_epoch_time': []}

print('Training starts!')
for epoch in range(1, num_epochs+1):
    # Start the timer
    start = time.time()

    # Initialize loss
    discriminator_loss, generator_loss = [], []

    # Loop over the training set
    for index, (real_images, labels) in enumerate(train_loader):
        # Zero out any previously accumulated discriminator gradients
        D_optimizer.zero_grad()

        # Send the input to the device and perform forward pass
        real_images = real_images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long().abs()

        # Initialize real and fake target tensors
        real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))

        # Compute discriminator loss with real images
        D_real_loss = loss(D((real_images, labels)), real_target)

        # Noise vectors
        noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)
        noise_vector = noise_vector.to(device)

        # Get generated image
        generated_image = G((noise_vector, labels))

        # Train with generated images
        output = D((generated_image.detach(), labels))

        # Compute discriminator loss with generated images
        D_fake_loss = loss(output,  fake_target)

        # Total discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        discriminator_loss.append(D_total_loss)

        # Backward propagate
        D_total_loss.backward()

        # Update discriminator parameters
        D_optimizer.step()

        # Zero out any previously accumulated generator gradients
        G_optimizer.zero_grad()

        # Train generator with real labels and compute total generator loss
        G_loss = loss(D((generated_image, labels)), real_target)
        generator_loss.append(G_loss)

        # Backward propagate
        G_loss.backward()

        # Update generator parameters
        G_optimizer.step()

    # Print iteration information
    print('Epoch: [%d/%d]: D_loss: %.3f | G_loss: %.3f' % ((epoch),
                                                          num_epochs,
                                                          torch.mean(torch.FloatTensor(discriminator_loss)),
                                                          torch.mean(torch.FloatTensor(generator_loss))))

    # Update per epoch losses
    training_history['D_losses'].append(torch.mean(torch.FloatTensor(discriminator_loss)))
    training_history['G_losses'].append(torch.mean(torch.FloatTensor(generator_loss)))

    # Save generated coin images
    save_image(generated_image.data, '../results/generated_images/sample_%d'%epoch + '.png', nrow=12, normalize=True)

    # Save training weights
    torch.save(G.state_dict(), '../results/training_weights/generator_epoch_%d.pth' % (epoch))
    torch.save(D.state_dict(), '../results/training_weights/discriminator_epoch_%d.pth' % (epoch))

    # Stop the timer
    end = time.time()

    # Update per epoch training time
    elapsed = end - start
    training_history['per_epoch_time'].append(time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed)))
print('Training ends!')

# Plot the training loss
performance(training_history)

## Evaluation
# Load test images
TestDS = Coins_Dataloader(csv_file='../data/Test.csv', transforms=transform)
test_loader = DataLoader(TestDS,
                         shuffle = False,
                         batch_size = 128)

# Evaluate the model on test set
print('\nEvaluating on test images!')
for index, (test_images, labels) in enumerate(test_loader):
    # Send the input to the device
    test_images = test_images.to(device)
    labels = labels.to(device)
    labels = labels.unsqueeze(1).long().abs()

    # Initialize real and fake target tensors
    real_target = Variable(torch.ones(test_images.size(0), 1).to(device))
    fake_target = Variable(torch.zeros(test_images.size(0), 1).to(device))

    # Compute discriminator loss with test images
    D_test_loss = loss(D((test_images, labels)), real_target)

    # Noise vectors
    noise_vector = torch.randn(test_images.size(0), latent_dim, device=device)
    noise_vector = noise_vector.to(device)

    # Get generated image from test images
    generated_image = G((noise_vector, labels))

    # Test with generated images
    output = D((generated_image.detach(), labels))

    # Compute discriminator loss with generated images
    D_fake_loss = loss(output,  fake_target)

    # Total discriminator loss on test set
    D_loss = (D_test_loss + D_fake_loss) / 2

    # Test generator with real labels and compute total generator loss
    G_loss = loss(D((generated_image, labels)), real_target)

    # Save generated coin images
    save_image(generated_image.data, '../results/generated_images/test_sample_%d'%index + '.png', nrow=12, normalize=True)

    print('Batch: [%d/%d]: D_loss: %.3f | G_loss: %.3f' % ((index),
                                                          len(test_loader)-1,
                                                          D_loss,
                                                          G_loss))
