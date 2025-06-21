import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=10):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.init_dim = 256
        self.fc = nn.Linear(latent_dim + label_dim, self.init_dim * 7 * 7)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_dim),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_dim, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat((z, self.label_emb(labels)), dim=1)
        x = self.fc(x).view(x.size(0), self.init_dim, 7, 7)
        return self.conv_blocks(x)

def load_generator(path, latent_dim=100):
    generator = Generator(latent_dim=latent_dim)
    generator.load_state_dict(torch.load(path, map_location="cpu"))
    generator.eval()
    return generator

def generate_digit_images(generator, digit, num_images=5, latent_dim=100):
    z = torch.randn(num_images, latent_dim)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    with torch.no_grad():
        generated = generator(z, labels)
    return generated
