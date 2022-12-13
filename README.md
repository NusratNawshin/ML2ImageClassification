# ML2ImageClassification

##  preprocessing.py
Downloads the data from drive and does all the preprocessing followed by storing the train, test and validation images in the directory

## Train.py
Performs the training and testing process and saves the model inside the model directory. If the preprocessed folders are not generated it would automatically run the preprocessing.py

## Test.py
Read the models and predicts. If the training model is not found it would train the model first and then executes the testing module.

## Masked_NN_withDownloadDataset.ipynb
Runs the whole pipeline in jupyter notebook file and includes all the outputs of each steps. It also includes testing and predicting with new unknown image that is saved as withoutmask.jpg into the directory.

## Instruction to run
 - python3 -m pip install -r requirements.txt
 - cd Code
 - python3 -u preprocessing.py
 - python3 -u train.py
 - python3 -u test.py
 
 To see the only test results from the models, directly run test.py file.
