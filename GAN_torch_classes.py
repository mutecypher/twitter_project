import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Swish(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Swish(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.Swish(),
            nn.Linear(4096, 256 * 256),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 256, 256)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Swish(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Swish(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_epochs = 100
batch_size = 64
lr = 0.0002

# Create the generator and discriminator
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Dataset and dataloader (adjust path as per your data)
dataset = ImageFolder(root='/Volumes/Elements/GitHub/cats_with_birds/For_Training', transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Adversarial ground truths
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)

        real_images = real_images.to(device)

        # Train the generator
        generator_optimizer.zero_grad()

        z = torch.randn(real_images.size(0), latent_dim).to(device)
        generated_images = generator(z)
        validity = discriminator(generated_images)
        generator_loss = adversarial_loss(validity, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator
        discriminator_optimizer.zero_grad()

        real_loss = adversarial_loss(discriminator(real_images), real_labels)
        fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake_labels)
        discriminator_loss = 0.5 * (real_loss + fake_loss)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Print training progress
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], "
                  f"Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

    # Save generated images
    save_image(generated_images.data[:25], f"gan_aug{epoch + 1}.jpg", nrow=5, normalize=True)
