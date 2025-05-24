#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import glob as gb
import cv2
from pathlib import Path


# In[4]:


train_dir = r'C:\Users\gunan\OneDrive\Documents\Desktop\photo\train'
test_dir = r'C:\Users\gunan\OneDrive\Documents\Desktop\photo\test'


# In[5]:


from pathlib import Path
import glob as gb

train_dir = Path(train_dir) 

for folder in train_dir.iterdir():
    if folder.is_dir():
        print(folder.name)
        files = gb.glob(str(folder / '*'))  
        print(len(files))


# In[6]:


test_dir = Path(test_dir)  

for folder in test_dir.iterdir():
    if folder.is_dir():  
        print(folder.name)
        files = gb.glob(str(folder / '*'))  
        print(len(files))


# In[7]:


size=[]
for folder in os.listdir(train_dir):
    #print(folder)
    
    file=gb.glob(pathname=str(train_dir/folder/'*'))
    #print(len(file))
    for image in file:
        img=plt.imread(image)
        size.append(img.shape)
pd.Series(size).value_counts()


# In[8]:





# In[9]:


size=[]
for folder in os.listdir(test_dir):
    #print(folder)
    
    file=gb.glob(pathname=str(test_dir/folder/'*'))
    #print(len(file))
    for image in file:
        img=plt.imread(image)
        size.append(img.shape)
pd.Series(size).value_counts()


# In[10]:





# In[11]:


code={
    0: 'surprise',
    1: 'fear',
    2: 'angry',
    3: 'neutral',
    4: 'sad',
    5: 'disgust',
    6: 'happy'
}
print(code)


# In[12]:


def getcode(n):
    for y,x in code.items():
        if n==y:
            return x


# In[13]:


print(getcode(3))


# In[14]:


name_to_code = {name: idx for idx, name in code.items()}

train_dir = Path(train_dir)

x_train = []
y_train = []

for folder in os.listdir(train_dir):
    files = gb.glob(str(train_dir / folder / '*'))
    for image_path in files:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (100, 100))
        x_train.append(resized_img)
        y_train.append(name_to_code[folder])


# In[15]:


set(y_train)


# In[16]:


plt.figure(figsize=(20,20))

for n, i in enumerate(list(np.random.randint(0, len(x_train), 36))):
    plt.subplot(6,6,n+1)
    plt.imshow(x_train[i])
    plt.axis("off")
    plt.title(code[y_train[i]])


# In[17]:


def getcode(label):
    label_dict = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    return label_dict.get(label)


# In[18]:


x_test = []
y_test = []

for folder in os.listdir(test_dir):
    files = gb.glob(pathname=str(test_dir / folder / '*'))
    for image in files:
        img = cv2.imread(image)
        resized_img = cv2.resize(img, (100, 100))
        x_test.append(resized_img)
        y_test.append(getcode(folder))


# In[19]:


print(os.listdir(test_dir))


# In[20]:


set(y_test)


# In[21]:


plt.figure(figsize=(20, 20))

for n, i in enumerate(np.random.randint(0, len(x_test), 36)):
    plt.subplot(6, 6, n + 1)
    plt.imshow(x_test[i])
    plt.axis("off")
    plt.title(code[y_test[i]])


# In[22]:


x_train=np.array(x_train)
x_test=np.array(x_test)


y_train=np.array(y_train)
y_test=np.array(y_test)


# In[23]:


x_train.shape


# In[24]:


x_test.shape


# In[25]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], jit_compile=False)


# In[26]:


final=model.fit(x_train,y_train,epochs=12)


# In[27]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

y_pred_prob = model.predict(x_test)

y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# In[29]:


import cv2
import numpy as np

# Replace this with the image path you want to test
image_path = r"C:\Users\gunan\OneDrive\Documents\Desktop\h1.jpg"

# Load the image using OpenCV
img = cv2.imread(image_path)

# Resize to match model input (100x100)
resized = cv2.resize(img, (100, 100))

# Normalize pixel values to [0, 1]
normalized = resized.astype('float32') / 255.0

# Reshape for batch input: (1, 100, 100, 3)
reshaped = np.expand_dims(normalized, axis=0)

# Predict
prediction = model.predict(reshaped)

# Decode prediction
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
predicted_index = np.argmax(prediction)
predicted_emotion = emotion_labels[predicted_index]

# Output
print("Predicted Emotion:", predicted_emotion)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Emotion: {predicted_emotion}", fontsize=14)
plt.axis('off')
plt.show()


# In[30]:


import matplotlib.pyplot as plt

# Plot accuracy and loss from 'final' history object
plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(final.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(final.history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




