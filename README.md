
![Logo](https://github.com/AKGanesh/CNN-CIFAR100/blob/main/cnn.png)

# Classification of CIFAR-100 using CNN

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-100 dataset. 
The main goal of this project is to build and train a CNN model that can accurately classify images from the CIFAR-100 dataset. 
Tried the below methods:
- CNN custom model from scratch
- VGG-16 Pre-existing model + custom dense layers
- Transfer Learning & Finetuning on VGG-16

## Implementation Details
- Dataset: CIFAR-100, consisting of 60,000 images in 100 classes.
- CNN Architecture: The model includes convolutional layers, pooling layers, dropout, and fully connected layers for classification.
- Regularization Techniques: Dropout and batch normalization are used to prevent overfitting and stabilize training.
- Data Augmentation: Various image augmentations (e.g., rotation, zoom, flipping) are applied to increase the diversity of the training data.
- Model Evaluation: The model performance is evaluated based on accuracy, precision, recall, and other relevant metrics.
- TensorBoard Integration: TensorBoard is used to visualize training progress, including accuracy, loss, and custom metrics.
- Checkpoints: Model weights are saved periodically, and the best model is saved based on validation accuracy.
- Pre-existing Models: VGG-16

## Dataset details

The CIFAR-100 dataset contains 100 classes of images, each represented by 600 images (500 for training and 100 for testing)
The CIFAR-100 dataset can be automatically downloaded using Keras datasets. It contains 100 classes with 600 images per class. The dataset is pre-split into training and testing sets:
50,000 training images
10,000 testing images
CIFAR-100 includes 100 fine-grained classes, grouped into 20 superclasses. Each image is 32x32 pixels and belongs to one of these classes, which range from animals to vehicles, household objects, and more.

## Process

- Data and Pre-processing

  - Import the required libraries
  - Read the dataset
  - Preprocessing (Normalize the dataset)
  - Data Preparation

- #### Method-1 CNN From Scratch
  - Create a Sequential model (Input-> 3 Conv, 2 dense, Output)
  - Compile and Test
  - Set loss, Optimizer and metrics
  - Check the model summary
  - Train the model (epcohs,batchsize, validation and callbacks)
  - Callbacks are to log to tensorboard and early stopping
  - Generate the plots - Loss vs Epochs for Train and validation set
  - Evaluate the model on test set and check the results
  
- #### Method-2 Use pre-existing model(VGG-16)
  - In this case, we are using an existing model VGG-16 and following feature extraction technique
  - We ll use the base model; create and train the top layers

- #### Method-3 Pre-existing model(VGG-16)+Transfer Learning+FT
  - In this case, we are using an existing model VGG-16 and following Transfer Learning + Fine tuning
  - We ll freeze the base model and train the top layers, next we ll unfreeze a couple of layers of the base model and retrain. Also rely on techniques like GlobalAveragePooling, Dropout etc.,
  - Learning Rate has an impact on the outcome, please focus on this

## Evaluation and Results
  | Type | Test Loss | Test Accuracy |
  |------|-----------|---------------|
  |CNN From Scratch|2.4290168285|0.4009000062|
  |Feature Extraction-VGG16|1.6670670509|0.57480001449|
  |Transfer Learning + Fine Tuning|2.036684989|0.4781000018|

Philosophy:
- There is a possibility to make CNN from Scratch work better than others, but this requires expertise and finesse. Dataset size also matters here. With small Datasets, it is difficult to achieve great results incase of CNNs.
- With Pre-existing models, it is easy to converge quickly and with small Dataset you will achieve great results with minimal effort.
- With Transfer Learning and Fine tuning you ll gain more control and achieve great results. Bit of efforts are needed to unfreeze and update the weights. A little experience is needed to work with Transfer learning and fine tuning.

## Observations
- Tested the scenarios of changing the number of layers and perceptrons on the layer.
- Learning rate is playing a vital role in case of using an existing model and transfer learning.
- Dropout plays an important role in regularization.

## Libraries

**Language:** Python,

**Packages:** Tensorflow, Keras, Numpy, Matplotlib, Tensorboard

## Roadmap

- To check with other ImageNets like resNet50
- To work with checkpoints, custom callbacks in tensorboard
- To check GradCAM

## FAQ

##### What is CNN?
A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed to process structured grid data, such as images. CNNs are particularly well-suited for tasks involving visual data, like image classification, object detection, and segmentation, due to their ability to automatically learn spatial hierarchies of features.
Typical CNN Architecture
Input -> [Conv Layer -> ReLU] -> [Pooling] -> [Conv Layer -> ReLU] -> [Pooling] -> [Fully Connected] -> Output

#### Whats is Tensorboard?

In machine learning, to improve something you often need to be able to measure it. TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
https://www.tensorflow.org/tensorboard/get_started

#### What is a classification problem?

A classification problem in machine learning is a task where the goal is to predict the category (or class) that a new data point belongs to. It's essentially about sorting things into predefined groups. Different types include Binary, Multi-Class and Multi-Label.

#### What is Keras?

Keras is a deep learning API written in Python and capable of running on top of either JAX, TensorFlow, or PyTorch. As a multi-framework API, Keras can be used to develop modular components that are compatible with any framework – JAX, TensorFlow, or PyTorch.

#### What is Transfer Learning?
Transfer learning allows you to leverage the knowledge learned by a model in one task to improve performance on a different but related task. It’s particularly useful when you have limited data or computational power and is widely used in both computer vision and NLP tasks.

Feature Extraction:
In this approach, you use the pre-trained model as a fixed feature extractor. You freeze all the layers of the pre-trained model except for the final output layer, which is replaced with a new output layer specific to your task.

Fine-tuning:
Here, you not only replace the final layer of the model but also allow some (or all) of the earlier layers to be fine-tuned during training. This adjusts the pre-trained model’s features slightly to better fit the specific dataset or task.


## Acknowledgements

- https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
- https://www.tensorflow.org/tensorboard
- https://keras.io/guides/sequential_model/

## Contact

For any queries, please send an email (id on github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


