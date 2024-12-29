# Lung Cancer Prediction using CNN and Transfer Learning

This project will use Convolutional Neural Network and transfer learning to creat Lung Cancer deep learning model. The model classifies lung cancer images into four categories: Some types of lung cancer stages include: Lung cancer stage 1A, Lung cancer stage 1B, Lung cancer stage 2, Lung cancer stage 3, Lung cancer stage 3A, Lung cancer stage 3B Lung cancer stage 3C, and Lung cancer stage 4.


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Using the Model](#using-the-model)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Tumor of lung is among the cancer that results to many deaths in the world today. Often, cancer is only diagnosed in the later stages and then, it is very important if it is correctly classified so that the treatment can be carried out and lives can be saved. This project thereby aims at using advanced learning such as deep learning techniques to train a lung cancer classification model based on chest X-Ray Images.
## Dataset

The dataset used in this project consists of lung cancer images categorized into four classes:
1. Normal
2. Adenocarcinoma
3. Large Cell Carcinoma
4. Squamous Cell Carcinoma

The dataset should be organized into training (`train`), validation (`valid`), and testing (`test`) folders with the following subfolders for each class:

- `train/`
  - `normal/`
  - `adenocarcinoma/`
  - `large_cell_carcinoma/`
  - `squamous_cell_carcinoma/`

- `valid/`
  - `normal/`
  - `adenocarcinoma/`
  - `large_cell_carcinoma/`
  - `squamous_cell_carcinoma/`

- `test/`
  - `normal/`
  - `adenocarcinoma/`
  - `large_cell_carcinoma/`
  - `squamous_cell_carcinoma/`

Alternatively, you can also download a similar dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) which includes Chest CT scan images.

## Dependencies

The project requires the following libraries:
- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- tensorflow
- keras

You can install the required libraries using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow keras
```


## Project Structure

```
.
├── Lung_Cancer_Prediction.ipynb
├── README.md
├── dataset
│ ├── train
│ │ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ │ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ │ ├── normal
│ │ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
│ ├── test
│ │ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ │ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ │ ├── normal
│ │ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
│ └── valid
│ ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
│ ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
│ ├── normal
│ └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
└── best_model.hdf5
```

This structure outlines the files and directories included in your project:

- **Lung_Cancer_Prediction.ipynb**: Python Notebook containing the code for training and evaluating the lung cancer prediction model.
- **README.md**: Markdown file providing an overview of the project, usage instructions, and other relevant information.
- **dataset/**: Directory containing the dataset used for training and testing.
  - **train/**: Subdirectory containing training images categorized into different classes of lung cancer.
  - **test/**: Subdirectory containing testing images categorized similarly to the training set.
  - **valid/**: Subdirectory containing validation images categorized similarly to the training set.
- **best_model.hdf5**: File where the best-trained model weights are saved after training.




## Training the Model

The Python Notebook `Lung_Cancer_Prediction.ipynb` contains the code for training the model. Below are the steps involved:

1. **Mount Google Drive**: To access the dataset stored in Google Drive.
2. **Load and Preprocess Data**: Use `ImageDataGenerator` for data augmentation and normalization.
3. **Define the Model**: Use the Xception model pre-trained on ImageNet as the base model and add custom layers on top.
4. **Compile the Model**: Specify the optimizer, loss function, and metrics.
5. **Train the Model**: Fit the model on the training data and validate it on the validation data. Callbacks like learning rate reduction, early stopping, and model checkpointing are used.
6. **Save the Model**: Save the trained model for future use.

### Example Usage

```python

# Load and preprocess data
IMAGE_SIZE = (350, 350)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=8,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validate_folder,
    target_size=IMAGE_SIZE,
    batch_size=8,
    class_mode='categorical'
)

# Define the model
pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
pretrained_model.trainable = False

model = Sequential([
    pretrained_model,
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=25,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=20
)

# Save the model
model.save('/Users/deepanshudubb/Documents/projects/Lung Cancer Prediction using CNN/trained_lung_cancer_model.h5')
```


## Using the Model

To use the trained model for predictions, follow these steps:

1. **Load the Trained Model**: Load the saved `.h5` model file using TensorFlow/Keras.
2. **Preprocess the Input Image**: Load and preprocess the input image using `image.load_img()` and `image.img_to_array()`.
3. **Make Predictions**: Use the loaded model to predict the class of the input image.
4. **Display Results**: Display the input image along with the predicted class label.

### Example Code

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('/Users/deepanshudubb/Documents/projects/Lung Cancer Prediction using CNN/trained_lung_cancer_model.h5')

def load_and_preprocess_image(img_path, target_size):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Example usage with an image path
img_path = '/Users/deepanshudubb/Documents/projects/Lung Cancer Prediction using CNN/image1.png'
target_size = (350, 350)

# Load and preprocess the image
img = load_and_preprocess_image(img_path, target_size)

# Make predictions
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to the class label
class_labels = list(train_generator.class_indices.keys())  # Assuming `train_generator` is defined
predicted_label = class_labels[predicted_class]

# Print the predicted class
print(f"The image belongs to class: {predicted_label}")

# Display the image with the predicted class
plt.imshow(image.load_img(img_path, target_size=target_size))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
```



## Results

After training and evaluating the lung cancer prediction model, the following results were obtained:

- Final training accuracy: `history.history['accuracy'][-1]`
- Final validation accuracy: `history.history['val_accuracy'][-1]`
- Model accuracy: 93%


### Example Predictions

Include images and their predicted classes here, demonstrating the model's performance on new data.








