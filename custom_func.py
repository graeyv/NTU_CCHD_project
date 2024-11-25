import os
import numpy as np

# Function to load images from a directory given a list of image filenames
def load_img(img_dir, img_list):
    """
    Load images stored in .npy format from a specified directory.

    Parameters:
    - img_dir (str): The directory containing the images.
    - img_list (list): List of image filenames to be loaded.

    Returns:
    - np.ndarray: A numpy array containing the loaded images.
    """
    images = []  # Initialize an empty list to store the images

    # Iterate over the image filenames in the list
    for i, image_name in enumerate(img_list):    
        # Check if the file has the .npy extension
        if image_name.split('.')[1] == 'npy':
            # Load the .npy file and append it to the images list
            image = np.load(img_dir + image_name)
            images.append(image)

    # Convert the list of images into a numpy array
    images = np.array(images)
    return images


# Generator to progressively provide batches of images and masks
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    Generator that yields batches of images and masks.

    Parameters:
    - img_dir (str): Directory containing the input images.
    - img_list (list): List of input image filenames.
    - mask_dir (str): Directory containing the corresponding masks.
    - mask_list (list): List of mask filenames.
    - batch_size (int): Number of images and masks per batch.

    Yields:
    - tuple: A tuple containing:
        - X (np.ndarray): A batch of input images.
        - Y (np.ndarray): A batch of corresponding masks.
    """
    L = len(img_list)  # Total number of images

    # Keras generators need to loop infinitely, so use `while True`
    while True:
        # Initialize batch indices
        batch_start = 0 
        batch_end = batch_size  

        # Loop through the dataset in batches
        while batch_start < L:
            limit = min(batch_end, L)  # Ensure the batch does not exceed the dataset size

            # Load the batch of images and masks
            X = load_img(img_dir, img_list[batch_start:limit])  # Load images
            Y = load_img(mask_dir, mask_list[batch_start:limit])  # Load masks

            # Convert to float32
            X = X.astype('float32')  
            Y = Y.astype('float32') 

            # Yield a tuple of image and mask batches
            yield (X, Y)  # A tuple with two numpy arrays, each containing `batch_size` samples

            # Update the indices for the next batch
            batch_start += batch_size   
            batch_end += batch_size
