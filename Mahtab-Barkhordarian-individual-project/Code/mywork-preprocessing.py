from imblearn.over_sampling import RandomOverSampler
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import gdown
import zipfile

if not (os.path.exists('model')):
    os.mkdir('model')
if (not os.path.exists('images') or not os.path.exists('annotations')):

    url = "https://drive.google.com/uc?id=1dcO2Hyery3NbwPGRs11wlGnv6LIyhsUC"
    output = "dataset.zip"
    gdown.download(url, output, quiet=False)
    data_zip=zipfile.ZipFile('dataset.zip')
    data_zip.extractall()
    data_zip.close()

def parse_annotation(path):
    tree = ET.parse(path)
    root = tree.getroot()
    constants = {}
    objects = [child for child in root if child.tag == 'object']
    for element in tree.iter():
        if element.tag == 'filename':
            constants['file'] = element.text[0:-4]
        if element.tag == 'size':
            for dim in list(element):
                if dim.tag == 'width':
                    constants['width'] = int(dim.text)
                if dim.tag == 'height':
                    constants['height'] = int(dim.text)
                if dim.tag == 'depth':
                    constants['depth'] = int(dim.text)
    object_params = [parse_annotation_object(obj) for obj in objects]
    full_result = [merge(constants, ob) for ob in object_params]
    return full_result


def parse_annotation_object(annotation_object):
    params = {}
    for param in list(annotation_object):
        if param.tag == 'name':
            params['name'] = param.text
        if param.tag == 'bndbox':
            for coord in list(param):
                if coord.tag == 'xmin':
                    params['xmin'] = int(coord.text)
                if coord.tag == 'ymin':
                    params['ymin'] = int(coord.text)
                if coord.tag == 'xmax':
                    params['xmax'] = int(coord.text)
                if coord.tag == 'ymax':
                    params['ymax'] = int(coord.text)

    return params


def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

#developed by Mahtab

def balance(xdf_data):
    """
    xdf_data: dataframe for both x & y
    """
    # Augmenting Minority Target Variabe

    # The RandomOverSampler
    ros = RandomOverSampler(random_state=55)
    x = xdf_data.iloc[:, :-1]
    y = xdf_data.iloc[:, -1]

    # Augment the training data
    X_ros_train, y_ros_train = ros.fit_resample(x, y)
    new_data = pd.DataFrame(data=X_ros_train, columns=xdf_data.columns[:-1])
    new_data['name'] = y_ros_train
    return new_data
#done developing by Mahtab

def crop_img(image_path, x_min, y_min, x_max, y_max):
    x_shift = (x_max - x_min) * 0.1
    y_shift = (y_max - y_min) * 0.1
    img = Image.open(image_path)
    cropped = img.crop((x_min - x_shift, y_min - y_shift, x_max + x_shift, y_max + y_shift))
    return cropped


#
def extract_faces(image_name, image_info, input_data_path):
    faces = []
    df_one_img = image_info[image_info['file'] == image_name[:-4]][['xmin', 'ymin', 'xmax', 'ymax', 'name']]
    for row_num in range(len(df_one_img)):
        x_min, y_min, x_max, y_max, label = df_one_img.iloc[row_num]
        image_path = os.path.join(input_data_path, image_name)
        faces.append((crop_img(image_path, x_min, y_min, x_max, y_max), label, f'{image_name[:-4]}_{(x_min, y_min)}'))
    return faces


def save_image(image, image_name, output_data_path, dataset_type, label):
    output_path = os.path.join(output_data_path, dataset_type, label, f'{image_name}.png')
    image.save(output_path)


def preprocessing():
    input_data_path = str(os.getcwd()) + '/images'
    annotations_path = str(os.getcwd()) + "/annotations"
    images = [*os.listdir(str(os.getcwd()) + "/images")]
    output_data_path = '.'
    dataset = [parse_annotation(anno) for anno in glob.glob(annotations_path + "/*.xml")]

    full_dataset = sum(dataset, [])

    df = pd.DataFrame(full_dataset)
    print(df.shape)

    print(df.head())

    final_test_image = 'maksssksksss0'
    df_final_test = df.loc[df["file"] == final_test_image]
    images.remove(f'{final_test_image}.png')
    df = df.loc[df["file"] != final_test_image]

    print(df["name"].value_counts())

    # joined masked incorrectly with without mask
    df['name'] = df['name'].replace('mask_weared_incorrect', 'without_mask')

    df.insert(9, 'label', df.name)
    df = df.drop(['name'], axis=1)
    df = balance(df)
    labels = df['name'].unique()
    directory = ['train', 'test', 'val']
    output_data_path = '.'
    for label in labels:
        for d in directory:
            path = os.path.join(output_data_path, d, label)
            if not os.path.exists(path):
                os.makedirs(path)
    cropped_faces = [extract_faces(img, df, input_data_path) for img in images]
    flat_cropped_faces = sum(cropped_faces, [])
    with_mask = [(img, image_name) for img, label, image_name in flat_cropped_faces if label == "with_mask"]

    mask_weared_incorrect = [(img, image_name) for img, label, image_name in flat_cropped_faces if
                             label == "mask_weared_incorrect"]
    without_mask = [(img, image_name) for img, label, image_name in flat_cropped_faces if label == "without_mask"]

    print(len(with_mask))
    print(len(without_mask))
    print(len(mask_weared_incorrect))
    print(len(with_mask) + len(without_mask) + len(mask_weared_incorrect))

    train_with_mask, test_with_mask = train_test_split(with_mask, test_size=0.20, random_state=42)
    test_with_mask, val_with_mask = train_test_split(test_with_mask, test_size=0.7, random_state=42)
    train_without_mask, test_without_mask = train_test_split(without_mask, test_size=0.20, random_state=42)
    test_without_mask, val_without_mask = train_test_split(test_without_mask, test_size=0.7, random_state=42)
    for image, image_name in train_with_mask:
        save_image(image, image_name, output_data_path, 'train', 'with_mask')

    for image, image_name in train_without_mask:
        save_image(image, image_name, output_data_path, 'train', 'without_mask')

    for image, image_name in test_with_mask:
        save_image(image, image_name, output_data_path, 'test', 'with_mask')

    for image, image_name in test_without_mask:
        save_image(image, image_name, output_data_path, 'test', 'without_mask')

    for image, image_name in val_with_mask:
        save_image(image, image_name, output_data_path, 'val', 'with_mask')

    for image, image_name in val_without_mask:
        save_image(image, image_name, output_data_path, 'val', 'without_mask')


# Run the preprocessing pipeline
if __name__ == "__main__":
    preprocessing()
