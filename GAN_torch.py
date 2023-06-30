import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import glob

print()
print(torch.__version__)
print()

# Define the path to your image dataset
dataset_path = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Fo_torch'

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def count_files_in_folder(folder_path):
    file_pattern = os.path.join(folder_path, '*')  # Specify the pattern to match files in the folder
    file_paths = glob.glob(file_pattern)  # Get a list of file paths that match the pattern
    file_count = len(file_paths)  # Count the number of files

    return file_count


# Create the ImageFolder dataset
dataset = ImageFolder(dataset_path, transform=transform)

# Create the dataloader
batch_size = 3
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print()
print(f"Number of batches in the dataloader: {len(dataloader)}")

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.SELU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.SELU(),
      ##      nn.Linear(2048, 4096),
       ##     nn.ReLU(),
            nn.Linear(2048, 256 * 256 * 3),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 256, 256)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ##nn.Linear(256 * 256 * 3, 4096),
            ##nn.SELU(),
            ##nn.Linear(4096, 2048),
           ## nn.ReLU(),
            nn.Linear(256 * 256 * 3, 2048),
            nn.SELU(),
            nn.Linear(2048, 1024),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

num_epochs = 200



dataset_path_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Fo_torch'


file_count = count_files_in_folder(dataset_path_folder)
print()
print(f"Number of files in the folder: {file_count}")
print()
print(f"another count {len(dataloader)} batch size times length = {len(dataloader) * batch_size} ")
print()

generator_path = '/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator'

# Check if the folder exists, create it if it doesn't
if not os.path.exists(generator_path):
    os.makedirs(generator_path)


discriminator_path = '/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator'

# Check if the folder exists, create it if it doesn't
if not os.path.exists(discriminator_path):
    os.makedirs(discriminator_path)



if os.path.exists('/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth') and os.path.exists('/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth'):
    # Load the saved model parameters
    print("yes, path exists")
    generator.load_state_dict(torch.load('/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth'))
    discriminator.load_state_dict(torch.load('/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth'))

# Training loop
for epoch in range(num_epochs):
    print()
    print(f"running epoch {epoch}")
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        valid = torch.ones(real_images.size(0), 1).to(device)
        fake = torch.zeros(real_images.size(0), 1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim).to(device)
        generated_images = generator(z)
        g_loss = criterion(discriminator(generated_images), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_images), valid)
        fake_loss = criterion(discriminator(generated_images.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        save_image(generated_images.data[:batch_size], f"/Volumes/Elements/GitHub/cats_with_birds/For_Training/gan_gened/ghn_aug{epoch + 1}.jpg", nrow=5, normalize=True)
    
    if os.path.exists('/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth'):
    # Delete the file
        os.remove('/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth')
        print(f"The file '{'/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth'}' has been deleted.")
    else:
        print(f"The file '{'/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth'}' does not exist.")
    
    
    if os.path.exists('/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth'):
    # Delete the file
        os.remove('/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth')
        print(f"The file '{'/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth'}' has been deleted.")
    else:
        print(f"The file '{'/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth'}' does not exist.")
    
    
       # Save the model parameters at the end of each epoch
    torch.save(generator.state_dict(), '/Volumes/Elements/GitHub/cats_with_birds/Torchy/generator/generator.pth')
    torch.save(discriminator.state_dict(), '/Volumes/Elements/GitHub/cats_with_birds/Torchy/discriminator/descriminator.pth')


    print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
    