# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 07:15:50 2024

@author: stefano
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def generate_lattice(n, p):
    """Generate an n x n lattice with site vacancy probability p."""
    return np.random.rand(n, n) < p

def percolates(lattice):
    """Check if the lattice percolates."""
    intersection = get_intersection(lattice)
    return intersection.size > 1

def get_intersection(lattice):
    """Check if the bottom labels or top labels have more than 1 value equal    """
    labeled_lattice, num_features = scipy.ndimage.label(lattice)
    top_labels = np.unique(labeled_lattice[0, :])
    bottom_labels = np.unique(labeled_lattice[-1, :])
    print(top_labels, bottom_labels, np.intersect1d(top_labels, bottom_labels).size)
    return np.intersect1d(top_labels, bottom_labels)

def plot_lattice(lattice):
    """Plot the lattice."""
    plt.imshow(lattice, cmap='binary')
    plt.show()
    
def lattice_to_image(lattice):
    """Convert the lattice in an RGB image (even if will be black and white)"""
    r_ch = np.where(lattice, 0, 255)
    g_ch = np.where(lattice, 0, 255)
    b_ch = np.where(lattice, 0, 255)
    
    image = np.concatenate((
        np.expand_dims(r_ch, axis=2),
        np.expand_dims(g_ch, axis=2),
        np.expand_dims(b_ch, axis=2),
        ), axis=2)
    
    return image

def get_path(lattice):
    intersection = get_intersection(lattice)
    labeled_lattice, num_features = scipy.ndimage.label(lattice)
    print(intersection)
    return labeled_lattice == intersection[1]

def add_path_img(image, path):
    blank_image = np.zeros(image.shape)
    # set the red channel to 255-> get a red percolation path
    image[path, 0] = 255
    image[path, 1] = 20
    return image
    


# Parameters
n = 100  # Lattice size
p_values = [0.2, 0.3, 0.4, 0.5, 0.58, 0.6]  # Site vacancy probabilities

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()


# Plot lattices for different p values
for i, p in enumerate(p_values):
    lattice = generate_lattice(n, p)
    lattice_img = lattice_to_image(lattice)
    if percolates(lattice):
        axs[i].set_title(f"p = {p} (Percolates)")
        path = get_path(lattice)
        lattice_img = add_path_img(lattice_img, path)
    else:
        axs[i].set_title(f"p = {p} (Does not percolate)")
    axs[i].imshow(lattice_img)
    axs[i].axis('off')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()

# Show the plot
plt.show()
plt.savefig("images/percolation_plot.png", dpi=300)