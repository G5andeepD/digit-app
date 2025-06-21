
import streamlit as st
import torch
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import load_generator, generate_digit_images

st.title("Handwritten Digit Generator (0–9)")

digit = st.selectbox("Select a digit to generate", list(range(10)))
generate = st.button("Generate Images")

if generate:
    st.write(f"Generating images for digit: {digit}")
    generator = load_generator("generator.pth")
    images = generate_digit_images(generator, digit, num_images=5)

    grid_img = make_grid(images, nrow=5, normalize=True).permute(1, 2, 0).numpy()
    st.image(grid_img, caption="Generated Digits", use_column_width=True)
