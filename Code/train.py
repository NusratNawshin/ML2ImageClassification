import os
import pickle
import numpy as np
import preprocessing
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout,BatchNormalization,Activation,Reshape,LSTM
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def check_preprocess():
    if os.path.exists('train') and os.path.exists('test') and os.path.exists('val'):
        print("validation Exists")
    else:
        preprocessing.preprocessing()


def masked_model():
    model = Sequential()

    model.add(Conv2D(128, kernel_size=5, strides=1, padding='same', input_shape=(35, 35, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    model.add(Conv2D(72, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=5, strides=4, padding='same'))

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=5, strides=4, padding='same'))

    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=4, strides=4, padding='same'))

    model.add(Reshape((-1, 64)))
    # LSTM
    model.add(LSTM(32))
    model.add(Dense(units=1, activation='sigmoid'))
    #
    model.summary()
    return model


# Data Augmentation for train images
def create_datagen():
    "Generates images with random augmentations"
    return ImageDataGenerator(
        rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
        height_shift_range=0.1, rotation_range=4, vertical_flip=False
    )


# Data Augmentation for validation and test images
def create_valDatagen():
    "Rescales images "
    return ImageDataGenerator(
        rescale=1.0 / 255
    )


def create_dataset_generator(datagen, batch_size, path,shuffle):
    "Data generator: reads images inn batches, encodes the target, reshapes the images"
    return datagen.flow_from_directory(
        directory=str(os.getcwd()) + path,
        target_size=(32, 32),
        class_mode="binary", batch_size=batch_size, shuffle=shuffle

    )
def prediction_label(predicts):
    temp = []
    for i in predicts:
        if i[0] < 0.5:
            temp.append(0)
        else:
            temp.append(1)
    return temp

def train():
    check_preprocess()
    batch_size = 8
    epochs = 150
    model = masked_model()
    datagen = create_datagen()
    val_datagen = create_valDatagen()
    train_generator = create_dataset_generator(datagen, batch_size, '/train',True)
    # Validation data
    val_generator = create_dataset_generator(val_datagen, batch_size, '/val',True)
    test_generator = create_dataset_generator(val_datagen, batch_size, '/test',False)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes),y=train_generator.classes)

    train_class_weights = dict(enumerate(class_weights))
    print("####################################")
    print(train_class_weights)

    data_size = len(train_generator)
    steps_per_epoch = int(data_size / batch_size)
    print(f"steps_per_epoch: {steps_per_epoch}")
    val_steps = int(len(val_generator) // batch_size)
    print(f"val_steps: {val_steps}")
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=['accuracy', 'Recall', 'Precision']
        # metrics=['accuracy']
    )
    # early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    # lrr = ReduceLROnPlateau(monitor='val_loss', patience=8, verbose=1, factor=0.5, min_lr=0.00001)
    model_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_data=val_generator,
        validation_steps=val_steps,
        class_weight=train_class_weights,
        # callbacks=[early_stopping, lrr]
    )

    # model_loss, model_acc, recall, precision= model.evaluate(test_generator)
    model_loss, model_acc, recall, precision= model.evaluate(test_generator)
    print("Test Scores:")
    print(
        f'Loss: {model_loss:.3f} || Accuracy: {model_acc * 100:.3f} || Recall: {recall * 100:.3f} || Precision: {precision * 100:.3f}')

    model.save_weights("model/masked_detection_model.h5")
    with open('model/history', 'wb') as file_pi:
        pickle.dump(model_history.history, file_pi)


if __name__ == "__main__":
    train()


# Reference Code
# https://www.kaggle.com/code/xiehf355023/face-mask-detection-cnn


