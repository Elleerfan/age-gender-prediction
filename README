# Age and Gender Prediction using CNN

This project implements a **Convolutional Neural Network (CNN)** to predict **age** and **gender** from facial images.
The model is built using **TensorFlow/Keras** and trained on the UTKFace dataset.

---

## Dataset

The model is trained on the **UTKFace Dataset**, which contains more than **20,000 face images** with labels for:

* Age
* Gender
* Ethnicity

Each image filename contains metadata in the following format:

age_gender_race_date&time.jpg

Example:

25_0_2_20170116174525125.jpg

Where:

* 25 → Age
* 0 → Gender (Male)
* 2 → Race

For this project, only **age and gender labels** are used.

---

## Model Architecture

The CNN model consists of the following layers:

* 4 Convolutional Layers
* MaxPooling Layers
* Flatten Layer
* Fully Connected Dense Layers
* Dropout for regularization

The network uses **multi-task learning** with two output branches:

1. **Gender Classification**

   * Activation: Sigmoid
   * Loss: Binary Crossentropy
   * Metric: Accuracy

2. **Age Prediction**

   * Activation: ReLU
   * Loss: Mean Absolute Error (MAE)
   * Metric: MAE

---

## Results

### Gender Prediction

Training Accuracy: ~95%
Validation Accuracy: ~88%

### Age Prediction

Training MAE: ~3.9 years
Validation MAE: ~6.5 years

These results show that the model performs well for gender classification and provides a reasonable estimation for age prediction.

---


## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Pillow

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/age-gender-prediction.git
cd age-gender-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the main script:

```bash
python age_gender_prediction.py
```

The script will:

1. Load the dataset
2. Extract image features
3. Train the CNN model
4. Plot training results
5. Predict age and gender for a sample image

---

## Example Prediction

The trained model predicts:

* Gender (Male / Female)
* Age (estimated years)

from a given facial image.

---

## Notes

This project is created for **educational and research purposes** to demonstrate deep learning techniques for **facial analysis and multi-task learning**.

---

## Author

Ellie
