import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
from keras.models import save_model


DATA_FOLDER = 'data'
NUMBER_OF_BINS = 25
MAX_IMAGES_PER_BIN = 400
BATCH_SIZE = 64
EPOCHS = 20


def validateDataFolder(data_folder):
    csv_path = os.path.join(data_folder, 'driving_log.csv')
    image_folder = os.path.join(data_folder, 'IMG')

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f'Missing driving log: {csv_path}'
        )

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f'Missing image folder: {image_folder}'
        )

    print('Dataset structure verified.')

def loadData(data_folder):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(data_folder, 'driving_log.csv'), names=columns)

    for i in range(len(data)):
        image_name = str(data.loc[i, 'Center']).replace('\\', '/').split('/')[-1]
        data.loc[i, 'Center'] = os.path.join(data_folder, 'IMG', image_name)

    data['Steering'] = pd.to_numeric(data['Steering'], errors='coerce')
    data = data.dropna(subset=['Center', 'Steering'])
    return data


def balanceData(data):
    histogram, bins = np.histogram(data['Steering'], NUMBER_OF_BINS)
    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, histogram, width=0.05)
    plt.title('Steering Data Before Balancing')
    plt.xlabel('Steering Angle')
    plt.ylabel('Number of Images')
    plt.savefig('steering_before_balancing.png')
    plt.close()

    remove_list = []
    for i in range(NUMBER_OF_BINS):
        indices = []
        for j in range(len(data)):
            steering = data.iloc[j]['Steering']
            if steering >= bins[i] and steering <= bins[i + 1]:
                indices.append(j)

        if len(indices) > MAX_IMAGES_PER_BIN:
            indices = np.random.choice(
                indices,
                len(indices) - MAX_IMAGES_PER_BIN,
                replace=False
            )
            remove_list.extend(indices)

    data = data.drop(data.index[remove_list])
    data = data.reset_index(drop=True)

    plt.hist(data['Steering'], bins=NUMBER_OF_BINS)
    plt.title('Steering Data After Balancing')
    plt.xlabel('Steering Angle')
    plt.ylabel('Number of Images')
    plt.savefig('steering_after_balancing.png')
    plt.close()
    return data


def panImage(image, steering):
    x_translation = np.random.uniform(-0.1, 0.1) * image.shape[1]
    y_translation = np.random.uniform(-0.1, 0.1) * image.shape[0]
    matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    steering = steering + x_translation * 0.002
    return image, steering


def zoomImage(image):
    scale = np.random.uniform(1, 1.3)
    matrix = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2),
        0,
        scale
    )
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def brightnessImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = image[:, :, 2].astype(float)
    brightness = brightness * np.random.uniform(0.4, 1.2)
    image[:, :, 2] = np.clip(brightness, 0, 255)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def rotateImage(image, steering):
    angle = np.random.uniform(-5, 5)
    matrix = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2),
        angle,
        1
    )
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    steering = steering - angle * 0.01
    return image, steering


def augmentImage(image, steering):
    if np.random.rand() < 0.5:
        image, steering = panImage(image, steering)
    if np.random.rand() < 0.5:
        image = zoomImage(image)
    if np.random.rand() < 0.5:
        image = brightnessImage(image)
    if np.random.rand() < 0.3:
        image, steering = rotateImage(image, steering)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def preProcessing(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image


def batchGenerator(image_paths, steering_values, batch_size, training):
    while True:
        image_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = np.random.randint(0, len(image_paths))
            image = cv2.imread(image_paths[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            steering = steering_values[index]

            if training:
                image, steering = augmentImage(image, steering)

            image = preProcessing(image)
            image_batch.append(image)
            steering_batch.append(steering)

        yield np.asarray(image_batch), np.asarray(steering_batch)


def createModel():
    model = Sequential([
        layers.Input(shape=(66, 200, 3)),
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.Flatten(),
        layers.Dense(100, activation='elu'),
        layers.Dense(50, activation='elu'),
        layers.Dense(10, activation='elu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model


if __name__ == '__main__':
    np.random.seed(42)
    
    validateDataFolder(DATA_FOLDER)

    data = loadData(DATA_FOLDER)
    data = balanceData(data)

    image_paths = data['Center'].values
    steering_values = data['Steering'].values

    X_train, X_validation, y_train, y_validation = train_test_split(
        image_paths,
        steering_values,
        test_size=0.2,
        random_state=42
    )

    model = createModel()
    model.summary()

    history = model.fit(
        batchGenerator(X_train, y_train, BATCH_SIZE, True),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=batchGenerator(
            X_validation,
            y_validation,
            BATCH_SIZE,
            False
        ),
        validation_steps=max(1, len(X_validation) // BATCH_SIZE)
    )

    plt.plot(np.arange(0, EPOCHS), history.history['loss'], label='loss')
    plt.plot(
        np.arange(0, EPOCHS),
        history.history['val_loss'],
        label='validation loss'
    )
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

    save_model(model, 'model.h5')
    print('Model saved as model.h5')
