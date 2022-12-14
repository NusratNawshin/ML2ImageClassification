import os
import pickle

import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


def check_preprocess():
    if os.path.exists('train') and os.path.exists('test') and os.path.exists('val'):
        print("validation Exists")
    else:
        preprocessing.preprocessing()

#developed my Mahtab
def masked_model():
    model = Sequential()
    model.add(Conv2D(filters=24, kernel_size=5, padding='same', activation='relu', input_shape=(35, 35, 3)))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=2, activation='softmax'))
    #
    model.summary()
    return model
#done

#developing by Mahtab

# Data Augmentation for train images
def create_datagen():
    return ImageDataGenerator(
        rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
        height_shift_range=0.1, rotation_range=4, vertical_flip=False
    )


# Data Augmentation for validation and test images
def create_valDatagen():
    return ImageDataGenerator(
        rescale=1.0 / 255
    )


def create_dataset_generator(datagen, batch_size,path):
    return datagen.flow_from_directory(
        directory=str(os.getcwd()) + path,
        target_size=(35, 35),
        class_mode="categorical", batch_size=batch_size, shuffle=True

    )
#done

def train():
    check_preprocess()
    batch_size = 8
    epochs = 100
    model = masked_model()
    datagen = create_datagen()
    val_datagen = create_valDatagen()
    train_generator = create_dataset_generator(datagen, batch_size,'/train')
    # Validation data
    val_generator=create_dataset_generator(val_datagen,batch_size,'/val')
    test_generator= create_dataset_generator(val_datagen,batch_size,'/test')

    # val_generator = val_datagen.flow_from_directory(
    #     directory=str(os.getcwd()) + '/val',
    #     target_size=(35, 35),
    #     class_mode="categorical", batch_size=batch_size, shuffle=True
    # )
    # Test data
    # test_generator = val_datagen.flow_from_directory(
    #     directory=str(os.getcwd()) + '/test',
    #     target_size=(35, 35),
    #     class_mode="categorical", batch_size=batch_size, shuffle=False
    # )
    data_size = len(train_generator)
    steps_per_epoch = int(data_size / batch_size)
    print(f"steps_per_epoch: {steps_per_epoch}")
    val_steps = int(len(val_generator) // batch_size)
    print(f"val_steps: {val_steps}")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=['accuracy', 'Recall', 'Precision', 'AUC']
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    lrr = ReduceLROnPlateau(monitor='val_loss', patience=8, verbose=1, factor=0.5, min_lr=0.00001)
    model_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        shuffle=True,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[early_stopping, lrr]
    )
    model_loss, model_acc, recall, precision, auc = model.evaluate(test_generator)
    print("Test Scores:")
    print(
        f'Loss: {model_loss:.3f} || Accuracy: {model_acc * 100:.3f} || Recall: {recall * 100:.3f} || Precision: {precision * 100:.3f} || AUC: {auc * 100:.3f}')
    model.save_weights("model/masked_detection_model.h5")
    with open('model/history', 'wb') as file_pi:
        pickle.dump(model_history.history, file_pi)


if __name__ == "__main__":
    train()
