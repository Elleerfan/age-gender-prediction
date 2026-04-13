import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from PIL import Image



BASE_DIR = "/Users/ellie/Desktop/age_gender_prediction/UTK/UTKFace"



image_paths = []
age_labels = []
gender_labels = []


for filename in tqdm(os.listdir(BASE_DIR)):

    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(BASE_DIR, filename)

    temp = filename.split("_")

    age = int(temp[0])
    gender = int(temp[1])

    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)



df = pd.DataFrame()

df["image"] = image_paths
df["age"] = age_labels
df["gender"] = gender_labels

print(df.head())

gender_dict = {0: "Male", 1: "Female"}



img = Image.open(df["image"][0])
plt.axis("off")
plt.imshow(img)
plt.show()

sns.displot(df["age"])
plt.show()

sns.countplot(data=df, x="gender")
plt.show()



plt.figure(figsize=(20,20))

files = df.iloc[0:25]

for i, (index,row) in enumerate(files.iterrows()):

    plt.subplot(5,5,i+1)

    img = load_img(row["image"])
    img = np.array(img)

    plt.imshow(img)
    plt.title(f"Age:{row['age']} Gender:{gender_dict[row['gender']]}")
    plt.axis("off")

plt.show()



def extract_features(images):

    features = []

    for image in tqdm(images):

        img = load_img(image, color_mode="grayscale")
        img = img.resize((128,128), Image.Resampling.LANCZOS)

        img = np.array(img)

        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features),128,128,1)

    return features


X = extract_features(df["image"])



X = X / 255.0

y_gender = np.array(df["gender"])
y_age = np.array(df["age"])



input_shape = (128,128,1)

inputs = Input(shape=input_shape)

conv_1 = Conv2D(32,(3,3),activation="relu")(inputs)
maxp_1 = MaxPooling2D((2,2))(conv_1)

conv_2 = Conv2D(64,(3,3),activation="relu")(maxp_1)
maxp_2 = MaxPooling2D((2,2))(conv_2)

conv_3 = Conv2D(128,(3,3),activation="relu")(maxp_2)
maxp_3 = MaxPooling2D((2,2))(conv_3)

conv_4 = Conv2D(256,(3,3),activation="relu")(maxp_3)
maxp_4 = MaxPooling2D((2,2))(conv_4)

flatten = Flatten()(maxp_4)

dense_1 = Dense(256,activation="relu")(flatten)
dense_2 = Dense(256,activation="relu")(flatten)

dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)

output_1 = Dense(1,activation="sigmoid",name="gender_out")(dropout_1)
output_2 = Dense(1,activation="relu",name="age_out")(dropout_2)

model = Model(inputs=inputs,outputs=[output_1,output_2])

model.compile(
    optimizer="adam",
    loss={
        "gender_out":"binary_crossentropy",
        "age_out":"mae"
    },
    metrics={
        "gender_out":"accuracy",
        "age_out":"mae"
    }
)

model.summary()



history = model.fit(
    X,
    [y_gender,y_age],
    batch_size=32,
    epochs=30,
    validation_split=0.2
)



acc = history.history["gender_out_accuracy"]
val_acc = history.history["val_gender_out_accuracy"]

epochs = range(len(acc))

plt.plot(epochs,acc,label="Train Accuracy")
plt.plot(epochs,val_acc,label="Val Accuracy")

plt.title("Gender Accuracy")
plt.legend()
plt.savefig("results/gender_accuracy.png")
plt.show()



loss = history.history["age_out_loss"]
val_loss = history.history["val_age_out_loss"]

plt.plot(epochs,loss,label="Train Loss")
plt.plot(epochs,val_loss,label="Val Loss")

plt.title("Age Loss")
plt.legend()
plt.savefig("results/age_loss.png")
plt.show()



image_index = 100

print(
    "Original Gender:",gender_dict[y_gender[image_index]],
    "Original Age:",y_age[image_index]
)

pred = model.predict(X[image_index].reshape(1,128,128,1))

pred_gender = gender_dict[int(round(float(pred[0][0])))]
pred_age = int(round(float(pred[1][0])))

print("Predicted Gender:",pred_gender)
print("Predicted Age:",pred_age)

plt.axis("off")
plt.imshow(X[image_index].reshape(128,128), cmap="gray")

plt.title(f"True: {gender_dict[y_gender[image_index]]}, {y_age[image_index]} | Pred: {pred_gender}, {pred_age}")

plt.savefig("results/sample_prediction.png")

plt.show()