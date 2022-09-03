import torch
import torch.nn as nn

# Define a CGANs generator class
class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        self.label_conditioned_generator = nn.Sequential(nn.Embedding(10, 100),
                                                         nn.Linear(100, 16))
        self.latent = nn.Sequential(nn.Linear(100, 4*4*512),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(nn.ConvTranspose2d(513, 64*8, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1,bias=False),
                                   nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1,bias=False),
                                   nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1,bias=False),
                                   nn.BatchNorm2d(64*1, momentum=0.1,  eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64*1, 3, 4, 2, 1, bias=False),
                                   nn.Tanh())
    # Forward method
    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)

        return image

# Define a CGANs discriminator class
class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_condition_disc = nn.Sequential(nn.Embedding(10, 100),
                                                  nn.Linear(100, 3*128*128))
        self.model = nn.Sequential(nn.Conv2d(6, 64, 4, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 64*2, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64*2, 64*4, 4, 3,2, bias=False),
                                   nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64*4, 64*8, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(4608, 1),
                                   nn.Sigmoid())
    # Forward method
    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        output = self.model(concat)

        return output

# Define custom weights to be called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)