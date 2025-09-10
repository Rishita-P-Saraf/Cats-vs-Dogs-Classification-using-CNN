# 🐶🐱 Cats vs Dogs Classification using CNN

This project builds a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images of cats and dogs.  
The dataset is taken from [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/datasets/salader/dogsvscats).

---

## 📂 Dataset Setup
- Train set: 20,000 images (cats and dogs).
- Validation set: 5,000 images (cats and dogs).

```bash
# Create Kaggle directory and copy API key
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# Download dataset
!kaggle datasets download salader/dogsvscats
```

Unzip dataset:
```python
import zipfile

zip_ref = zipfile.ZipFile("/content/dogsvscats.zip", "r")
zip_ref.extractall("/content")
zip_ref.close()
```
---

## ⚙️ Project Workflow
1. Data Loading

The dataset is loaded using keras.utils.image_dataset_from_directory, with images resized to 256x256.

```python
# Load train and test datasets
train_ds = keras.utils.image_dataset_from_directory(
    directory="/content/train",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory="/content/test",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256,256)
)
```
2. Preprocessing
- Normalization: Pixel values scaled to [0,1].
- Applied via a simple preprocessing function.
  
```python
def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

3. CNN Architecture

The model consists of:

- 3 convolutional layers with Batch Normalization and MaxPooling
- Dense layers with Dropout for regularization
- Sigmoid activation for binary classification
<img width="1412" height="369" alt="Screenshot 2025-09-10 125645" src="https://github.com/user-attachments/assets/5b680608-128e-4dbe-933a-fea393a30ff9" />

```python
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(256,256,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(64, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])
```
4. Model Compilation & Training

- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy
  
```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```
---

## 📊 Results
- Training & Validation Accuracy
<img width="577" height="431" alt="image" src="https://github.com/user-attachments/assets/7f5ff938-42cb-40c3-92a9-cdbf70d09746" />

- Training & Validation Loss
<img width="568" height="430" alt="image" src="https://github.com/user-attachments/assets/8e225ccd-58af-47ae-b5c9-53570b0da5d8" />
---

## 🐾 Predictions
- Cat Test Image
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/2d60f2d3-423b-4b82-a497-c2f879e4e077" />

```python
test_img1 = cv2.imread("/content/cat.jpg")
test_img1 = cv2.resize(test_img1, (256,256))
test_input1 = test_img1.reshape((1,256,256,3))
model.predict(test_input1)  
# Output: [[0.]] → Cat
```

- Dog Test Image
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/17de37c0-cca6-4dfc-94aa-abb3035e921c" />

```python
test_img2 = cv2.imread("/content/dog.jpeg")
test_img2 = cv2.resize(test_img2, (256,256))
test_input2 = test_img2.reshape((1,256,256,3))
model.predict(test_input2)  
# Output: [[1.]] → Dog
```
---

## 📌 Requirements
- Python 3.x
- TensorFlow / Keras
- OpenCV
- Matplotlib
- Kaggle API
---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dogs-vs-cats-classification.git
   cd dogs-vs-cats-classification
   ```
2. Download the dataset using Kaggle API.
3. Run the notebook / script to train and evaluate the model.
---

## 🎯 Future Improvements
- Data augmentation for better generalization.
- Transfer Learning using VGG16/ResNet50.
- Deploy model as a Flask/FastAPI web app.

---

## 📜 License

This project is licensed under the MIT License.
