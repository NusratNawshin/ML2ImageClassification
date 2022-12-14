import pickle
import train
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_loss_and_accuracy(history):
    "Plots loss and accuracy vs epochs"
    history_df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    history_df.loc[0:, ['loss', 'val_loss']].plot(ax=ax[0])
    ax[0].set(xlabel='epoch number', ylabel='loss')

    history_df.loc[0:, ['accuracy', 'val_accuracy']].plot(ax=ax[1])
    ax[1].set(xlabel='epoch number', ylabel='accuracy')
    plt.show()


def evaluation(y, y_hat, title='Confusion Matrix'):
    "Plots the confusion matrix"
    cm = confusion_matrix(y, y_hat)
    sns.heatmap(cm, cmap='PuBu', annot=True, fmt='g', annot_kws={'size': 20})
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(title, fontsize=18)

    plt.show()

def prediction_label(predicts):
    "Set threshold to predict labels"
    temp = []
    for i in predicts:
        if i[0] < 0.5:
            temp.append(0)
        else:
            temp.append(1)
    return temp

def test():
    batch_size = 8
    # output_data_path = '.'
    train.check_preprocess()
    if os.path.exists('model'):
        if not os.listdir('model'):
            train.train()
    else:
        os.mkdir('model')
        train.train()
    model = train.masked_model()
    model.load_weights('model/masked_detection_model.h5')
    with open('model/history', "rb") as file_pi:
        model_history = pickle.load(file_pi)
    val_datagen = train.create_valDatagen()
    test_generator = train.create_dataset_generator(val_datagen, batch_size, '/test',False)
    predictions = model.predict(test_generator)
    y_pred = prediction_label(predictions)
    y_pred = np.array(y_pred)

    plot_loss_and_accuracy(model_history)
    # paths = test_generator.filenames
    # y_pred = model.predict(test_generator).argmax(axis=1)
    # classes = test_generator.class_indices
    # a_img_rand = np.random.randint(0, len(paths))
    # img = cv2.imread(os.path.join(output_data_path, 'test', paths[a_img_rand]))
    # colored_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # plt.imshow(colored_img)
    # true_label = paths[a_img_rand].split('/')[0]
    # predicted_label = list(classes)[y_pred[a_img_rand]]
    # print(f'{predicted_label} || {true_label}')
    y_true = test_generator.labels
    # print("Predictions")
    # print(y_pred)
    # print("True Values")
    # print(y_true)
    # y_pred = model.predict(test_generator) # Predict prob and get Class Indices

    # display(classes)
    print(np.bincount(y_pred))
    opt = keras.optimizers.Adam(learning_rate=0.00009)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=['accuracy', 'Recall', 'Precision']
    )
    model_loss, model_acc, recall, precision= model.evaluate(test_generator)
    F1 = 2 * (precision * recall) / (precision + recall)
    print(f"F1 Score - {F1:.4f}")
    print(classification_report(y_true, y_pred))
    evaluation(y_true, y_pred)

if __name__ == "__main__":
    test()

# Reference Code
# https://www.kaggle.com/code/xiehf355023/face-mask-detection-cnn