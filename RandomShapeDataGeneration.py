import cv2
import numpy as np
import os
import pandas as pd
import random


class ShapeGenerator:
    def __init__(self, n_):
        self.n_ = n_
        self.shapes = ['circle', 'square', 'rectangle']

    def generate_image(self):
        data = []
        label = []

        for i in range(self.n_):

            # Choose a random shape
            shape = random.choice(self.shapes)

            # Generate a random position (x, y)
            x = random.randint(0, 63)
            y = random.randint(0, 63)

            # Generate a random size
            size = random.randint(8, 16)

            # Create a black image (64x64)
            img = np.zeros((64, 64), np.uint8)

            # Draw the shape on the image
            if shape == 'circle':
                # Ensure that the circle stays inside the image
                x = min(max(x, size), 63 - size)
                y = min(max(y, size), 63 - size)
                cv2.circle(img, (x, y), size, (255, 255, 255), -1)
            elif shape == 'square':
                # Ensure that the square stays inside the image
                x = min(max(x, size), 63 - 2 * size)
                y = min(max(y, size), 63 - 2 * size)
                cv2.rectangle(img, (x - size, y - size),
                              (x + size, y + size), (255, 255, 255), -1)
            elif shape == 'rectangle':
                # Ensure that the rectangle stays inside the image
                x = min(max(x, size), 63 - 2 * size)
                y = min(max(y, 2 * size), 63 - 2 * size)
                cv2.rectangle(img, (x - size, y - 2 * size),
                              (x + size, y + 2 * size), (255, 255, 255), -1)

            data.append(img.reshape(-1))
            label.append(self.shapes.index(shape))

        return data, label


def saveData(n_instant, outname):

    # Convert to pandas DataFrame
    shape = ShapeGenerator(n_instant)
    data, labels = shape.generate_image()
    # plt.imshow(np.reshape(data[1],(64,64)))
    df = pd.DataFrame(data=data, columns=[
                      f"pixel_{i}" for i in range(len(data[1]))])
    df.insert(0, 'label', labels)

    # Save to CSV without index
    df.to_csv(outname, index=False)
    print('Data distribution %', 'Rectangles =', np.sum(df['label'] == 2) / n_instant * 100, ', Square =', np.sum(
        df['label'] == 1) / n_instant * 100, 'Circle =', sum(df['label'] == 0) / n_instant * 100)


def create_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def test_train_data():
    create_dir('./data/shapes')
    saveData(n_instant=10000, outname='./data/shapes/train.csv')
    saveData(n_instant=2000, outname='./data/shapes/test.csv')
