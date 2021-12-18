# Multiclass Brain Tumor Classification

> Group A5

## Introduction
 A brain tumor is a mass or growth of abnormal cells in the brain. Many types of brain tumors exist, some being cancerous and others benign. Benign tumor cells rarely invade neighboring cells and have a slow progression rate. Malignant cells readily attack neighboring cells, have fuzzy borders, and rapid progression rates. The main tumor kinds are primary and metastatic. How quickly a brain tumor grows can vary, it depending on their location they are represented by different nomenclature. This project focuses on the development of a robust algorithm to distinguish between three tumors - Glioma, Meningioma, and Pituitary. We built an app in Python that can detect the brain tumor type. 

## Dataset
 The data can be accessed through: [Kaggle-Dataset](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)

## Motivations
> “What all of us have to do is to make sure we are using AI in a way that is for the benefit of humanity, not to the detriment of humanity." - Tim Cook, CEO Apple

I believe in data science for good!

## Model

![CNN_Model](https://github.com/vbgupta/Multiclass-Brain-Tumor-Classification/blob/main/project/volume/images/model.png?raw=true)

> Kernel Size = 4, activation = ReLu, image size = 128X128, color_scale = grayscale
- 5 Conv2D Layers
- 5 Max Pool Layers with pool_size = 0.2, padding = Same
- 4 Dropout Layers
- 4 Dense Layers

> Compile - loss = categorical_crossentropy, optimizer = Adam (learning_rate = 0.1, beta_1 = 0.9)

## Model Evaluation

![Accuracy](https://github.com/vbgupta/Multiclass-Brain-Tumor-Classification/blob/main/project/volume/images/CNN_V1_accuracy.png?raw=true)

![Loss](https://github.com/vbgupta/Multiclass-Brain-Tumor-Classification/blob/main/project/volume/images/CNN_V1_loss.png?raw=true)

![Last_5_Epochs](https://github.com/vbgupta/Multiclass-Brain-Tumor-Classification/blob/main/project/volume/images/last5epochs.png?raw=true)
### Achieved a validation accuracy of 84.63
## Links 

1. [Website](https://vbgupta.github.io/Multiclass-Brain-Tumor-Classification/)
2. [Code](https://github.com/vbgupta/Multiclass-Brain-Tumor-Classification)
3. [Presentation](https://docs.google.com/presentation/d/1Ly3UNyhePMgud5GrM81PgZWTFe0jJtfw/edit?usp=sharing&ouid=101144344517384173914&rtpof=true&sd=true)
4. [Final-Report](https://docs.google.com/document/d/1y7al_yqODdA9r9KnLRp56gqLGQRoemAoIWECeq7tBYE/edit?usp=sharing)

## Alternate Approaches
1. Unsupervised learning - We may not be able to directly implement this method , because we have labeled data.
2. Transfer Learning - Is there already a trained model that could be used to train on the data!?
3. Pytorch vs Tensorflow 

## How to use the Repository?

- Install the requiremnts to run this project using requirements.txt file under project/requirements/requirements.txt
- The project folder contains all the code for preprocessing the data and creating the model. This is under project/src
- The project folder contains the saved images in project/volume/images, and the data/raw folder is where the data should go from kaggle. 
- The project folder contains a models folder under project/volume/models where the model is present.
- To test this model out, run the app.py file contained at the root of the repository.
> To run the app -> open the terminal/command prompt in the repository director and type yourcomputer$ streamlit run app.py

