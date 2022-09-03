# Import the necessary packages
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from preprocessing import preprocess

class Coins_Dataloader(Dataset):
    def __init__(self, csv_file, transforms):
        # Read filepaths from csv file
        self.Dataset = pd.read_csv(csv_file)

        # Augmentation transforms
        self.transforms = transforms

    def __len__(self):
        # Return total samples in the dataset
        return len(self.Dataset['images'].to_list())

    def __getitem__(self, idx):
        # Image paths and denomination labels
        imagePaths = self.Dataset['images'].to_list()
        labels = self.Dataset['labels'].to_list()

        # Get image path from the current index
        imagePath = imagePaths[idx]

        # Read image from disk in grayscale
        image = cv2.imread(imagePath)

        # Scale the image dimensions
        scale = 512.0 / np.amin(image.shape[0:2])
        image = cv2.resize(image, (int(np.ceil(image.shape[1] * scale)), int(np.ceil(image.shape[0] * scale))))

        # Create preprocessing object
        p = preprocess()

        # Segment the image
        p.segment(image)

        # Get dimensions of the area bounding the coin
        rect = cv2.boundingRect(p.edge)
        x,y,w,h = rect

        # Crop the image within the bounding region
        image = image[y:y+h, x:x+w]

        # Get the associated denomination
        label = labels[idx]

		# Applying transformations, if any
        if self.transforms is not None:
            image = self.transforms(image)

		# Return a tuple of the image and its mask
        return (image, label)