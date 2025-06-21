
import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

def load_generator(path):
    generator = Generator()
    generator.load_state_dict(torch.load(path, map_location="cpu"))
    generator.eval()
    return generator

def generate_digit_images(generator, digit, num_images=5, latent_dim=100):
    z = torch.randn(num_images, latent_dim)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    with torch.no_grad():
        generated = generator(z, labels)
    return generated
