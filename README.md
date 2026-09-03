# Fog / Clear Image Classification with CNN and Ensemble Experiments

> **Semester Machine Learning Project**  
> An image-classification project built with TensorFlow/Keras that trains a custom CNN on a Kaggle fog/visibility dataset, evaluates the trained network, extracts CNN features for additional boosting and ensemble experiments, and provides a Streamlit interface for local inference.

> Live Demo link:
https://dsasmogproject-5kqfec3rejntwedvsdjw5d.streamlit.app/

> Hugging Face Model link:
https://huggingface.co/PakeezaKhalid/clear-fog-cnn-classifier

![Uploading image.png…]()

## Overview

This project investigates image-based classification of environmental visibility conditions.

The work consists of three connected parts:

1. **CNN training and evaluation** in a Google Colab notebook.
2. **Feature-based machine-learning experiments** using representations extracted from the trained CNN.
3. **Streamlit inference application** using the saved Keras model.

The main workflow implemented in the notebook is:

```text
Kaggle image dataset
        ↓
TensorFlow image dataset
        ↓
Resize + normalization
        ↓
Data augmentation
        ↓
Custom CNN
        ↓
Training + validation
        ↓
Held-out test evaluation
        ↓
Saved .keras model
        ↓
Streamlit inference app
```

The notebook also reuses the trained CNN as a feature extractor and evaluates several additional classifiers:

```text
CNN feature extractor
        ↓
 ┌──────┼────────┬─────────┐
 ▼      ▼        ▼         ▼
XGBoost LightGBM CatBoost Ensembles
                     │
                     ├─ Stacking
                     └─ Soft Voting
```

---

## Dataset

The dataset used in the notebook is downloaded directly from Kaggle with `opendatasets`:

**Fog or Smog Detection Dataset**  
Kaggle author: `ahmedislam0`

Dataset link:

```text
https://www.kaggle.com/datasets/ahmedislam0/fog-or-smog-detection-dataset
```

The training notebook reports:

```text
2,335 image files
2 classes
```

The current application maps the intended inference labels as:

```text
Clear
Fog
```

The dataset itself is not included in this repository.

---

## Framework and Environment

The main deep-learning implementation uses:

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab / Jupyter
- `opendatasets` for Kaggle dataset retrieval

Additional experiments use:

- XGBoost
- LightGBM
- CatBoost
- scikit-learn
- OpenCV
- Streamlit
- Pillow

---

# Data Preparation

The notebook defines:

```python
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20
```

Images are loaded with:

```python
tf.keras.preprocessing.image_dataset_from_directory(...)
```

using:

```text
256 × 256 RGB images
batch size = 32
shuffle = True
```

## Dataset Split

The notebook implements a nominal:

```text
80% training
10% validation
10% test
```

split using TensorFlow dataset `take()` and `skip()` operations after shuffling with seed `12`.

## Preprocessing

Each image is:

1. resized to `256 × 256`;
2. cast to `float32`;
3. normalized from pixel values `[0, 255]` to `[0, 1]`.

```python
image = tf.image.resize(image, [256, 256])
image = tf.cast(image, tf.float32) / 255.0
```

The datasets are then cached and prefetched with `tf.data.AUTOTUNE`.

---

# Data Augmentation

The CNN training pipeline includes:

```python
layers.RandomFlip("horizontal_and_vertical")
layers.RandomRotation(0.2)
layers.RandomZoom(0.2)
```

This augmentation is applied during model training to introduce variation in orientation and scale.

---

# CNN Architecture

The main model is a custom TensorFlow/Keras convolutional neural network.

```text
Input: 256 × 256 × 3
        ↓
Data Augmentation
        ↓
Conv2D — 32 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Dropout(0.3)
        ↓
Conv2D — 64 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Dropout(0.3)
        ↓
Conv2D — 128 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Conv2D — 128 filters, 3×3, ReLU
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense — 256, ReLU
        ↓
Dropout(0.5)
        ↓
Dense — 128, ReLU
        ↓
Softmax output layer
```

The notebook currently defines the final layer as:

```python
layers.Dense(3, activation="softmax")
```

while the dataset loader reports **2 classes**. This mismatch is preserved in the original training notebook and documented here rather than hidden.

It also explains why the current saved model produces three output probabilities even though the Streamlit interface has two intended labels.

---

# Training Configuration

The CNN is compiled with:

```python
optimizer = Adam(learning_rate=0.001)
loss = SparseCategoricalCrossentropy(from_logits=False)
metric = accuracy
epochs = 20
```

Training is performed with:

```python
model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds
)
```

The notebook stores the complete epoch-by-epoch training output and plots:

- training accuracy;
- validation accuracy;
- training loss;
- validation loss.

At epoch 20, the recorded output is approximately:

```text
Training accuracy:   96.61%
Validation accuracy: 96.88%
```

---

# CNN Evaluation

The CNN is evaluated in the notebook using:

```python
model.evaluate(test_ds)
```

The notebook records:

```text
Test Accuracy: 96.88%
```

This number is reported here as the **recorded notebook result from the original semester experiment**.

The notebook also contains visual test-set predictions showing:

```text
Actual class
Predicted class
Prediction confidence
```

for example test images.

> The project has not been retrained solely to regenerate cleaner metrics for this README. The purpose of this repository is to preserve and document the work actually performed in the semester project.

---

# CNN Feature Extraction

After training, the saved CNN is reused as a feature extractor.

The notebook constructs a Keras model using the trained network up to the layer before the final classification output:

```text
Input image
    ↓
trained CNN layers
    ↓
learned feature representation
    ↓
classical / ensemble classifier
```

Features are extracted separately for the training, validation, and test datasets.

These learned CNN features are then used in the additional experiments below.

---

# Additional Machine-Learning Experiments

The notebook contains experiments with five alternative classification approaches.

## CatBoost

A `CatBoostClassifier` is trained using extracted CNN features with:

```text
iterations = 1000
learning rate = 0.05
depth = 6
```

Recorded notebook result:

```text
CatBoost Test Accuracy: 50.11%
```

## LightGBM

A LightGBM classifier is trained using:

```text
n_estimators = 1000
learning rate = 0.05
max depth = 6
```

Recorded notebook result:

```text
LightGBM Test Accuracy: 99.22%
```

## XGBoost

The notebook trains an XGBoost model on the extracted CNN features using a multi-class objective.

Recorded notebook result:

```text
XGBoost Test Accuracy: 98.83%
```

## Stacking

The project also implements a stacking experiment combining predictions from:

- XGBoost;
- LightGBM;
- CatBoost.

A Logistic Regression model is used as the meta-model.

Recorded notebook result:

```text
Stacked Model Test Accuracy: 99.22%
```

## Soft-Voting Ensemble

A `VotingClassifier` combines:

- XGBoost;
- CatBoost;
- LightGBM

using soft voting.

Recorded notebook result:

```text
Ensemble Model Test Accuracy: 98.83%
```

## Recorded Experiment Summary

| Model / Experiment | Recorded Test Accuracy |
|---|---:|
| Custom CNN | **96.88%** |
| CatBoost on CNN features | **50.11%** |
| LightGBM on CNN features | **99.22%** |
| XGBoost on CNN features | **98.83%** |
| Stacking ensemble | **99.22%** |
| Soft-voting ensemble | **98.83%** |

These values are reproduced from the outputs already stored in the training notebook. They represent the original experimental runs and are not presented as a separately rerun benchmark study.

---

# Model Saving

The notebook saves the trained Keras model in several experimental locations during development, including:

```python
model.save("models/my_model.keras")
```

The Streamlit application expects the model as:

```text
./my_model.keras
```

The trained model file is relatively large, so it is kept outside the normal Git repository.

---

# Streamlit Inference Application

`app.py` provides a lightweight interface around the saved Keras model.

The inference pipeline is:

```text
User uploads JPG / PNG / JPEG
        ↓
Pillow opens image
        ↓
Convert to RGB
        ↓
Resize to 256 × 256
        ↓
Convert to NumPy float32
        ↓
Normalize / 255.0
        ↓
Add batch dimension
        ↓
my_model.keras
        ↓
Model output
```

Run locally with:

```powershell
streamlit run app.py
```

---

# Inference Safety Check

The saved semester-project model currently returns **3 probabilities**, while the application has **2 intended labels**.

The current Streamlit interface therefore checks:

```text
number of model outputs
        vs.
number of application labels
```

and refuses to silently assign a Clear/Fog label when those dimensions do not match.

Instead, it displays the raw model probabilities and explains the mismatch.

This prevents an invalid class mapping from being presented as a valid prediction.

---

# Out-of-Domain Inputs

The model was trained as a closed-set environmental image classifier.

An unrelated image — for example:

- handwritten notes;
- documents;
- indoor objects;
- other image categories absent from training —

is outside the intended input domain.

A softmax classifier can still produce a high score for an unrelated image because it must distribute probability across its known outputs.

Therefore, a high output score should not be interpreted as proof that an arbitrary uploaded image is genuinely Clear or Fog.

The current interface documents this limitation instead of treating every upload as a meaningful environmental prediction.

---

# Grad-CAM / Model Interpretation Experiment

The notebook also contains an experimental Grad-CAM implementation.

It:

- identifies the last convolutional layer;
- constructs a gradient model;
- calculates gradients for a selected class;
- produces a heatmap;
- overlays the heatmap on the input image.

This was explored as an interpretability step for understanding which image regions influence CNN predictions.

---

# Model-Layer Exploration

The notebook also includes experimental traversal of model layers, including a breadth-first traversal output.

This was used while inspecting the trained model structure and experimenting with internal representations.

---

# Repository Files

A minimal public repository can contain:

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── updated_smog_fog.ipynb
```

Local-only files should remain excluded:

```text
venv/
my_model.keras
kaggle*.json
```

The Kaggle credentials file should never be committed.

---

# Local Setup

## 1. Create a virtual environment

```powershell
python -m venv venv
```

## 2. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install application dependencies

```powershell
pip install -r requirements.txt
```

Current Streamlit application requirements:

```text
streamlit
tensorflow
pillow
numpy
```

## 4. Place the trained model

Place:

```text
my_model.keras
```

beside `app.py`.

## 5. Run

```powershell
streamlit run app.py
```

---

# Known Technical Limitations

The original experiment contains several limitations that are relevant when interpreting its results:

- the dataset loader reports 2 classes while the CNN output layer contains 3 neurons;
- the XGBoost experiment also uses `num_class = 3`;
- the current saved Keras model therefore does not align cleanly with the two-label Streamlit mapping;
- the dataset split is implemented at the TensorFlow dataset/batch level using `take()` and `skip()`;
- the notebook reports accuracy but does not contain a final confusion matrix or per-class precision, recall, and F1 report;
- the Streamlit application is intended for environmental images and does not implement dedicated out-of-distribution detection;
- the classical/ensemble results are experimental notebook results rather than independently repeated benchmark runs.

These limitations are documented because they are part of understanding the experimental pipeline and its behavior.

---

# What This Project Demonstrates

The project provides evidence of practical work across the ML pipeline:

```text
Kaggle dataset acquisition
        ↓
TensorFlow input pipeline
        ↓
image preprocessing
        ↓
data augmentation
        ↓
custom CNN design
        ↓
20-epoch training
        ↓
validation monitoring
        ↓
test evaluation
        ↓
visual predictions
        ↓
model serialization
        ↓
CNN feature extraction
        ↓
CatBoost / LightGBM / XGBoost
        ↓
stacking + voting ensembles
        ↓
Grad-CAM experimentation
        ↓
Streamlit model interface
```

It therefore captures both the main deep-learning workflow and exploratory work comparing the learned CNN representation with alternative machine-learning classifiers.
