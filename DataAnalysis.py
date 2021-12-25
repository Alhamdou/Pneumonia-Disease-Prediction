import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS =20
CHANNELS = 1




train_data= tf.keras.preprocessing.image_dataset_from_directory(directory=(r"C:\Users\LENOVO\Desktop\Ayush Assignments\chest-xray\medical\training\chest_xray\train"), shuffle=True,image_size=(IMAGE_SIZE,IMAGE_SIZE), color_mode="grayscale",batch_size=BATCH_SIZE, seed=42)
test_data= tf.keras.preprocessing.image_dataset_from_directory(directory=(r"C:\Users\LENOVO\Desktop\Ayush Assignments\chest-xray\medical\training\chest_xray\test"), shuffle=True,image_size=(IMAGE_SIZE,IMAGE_SIZE),color_mode="grayscale", batch_size=BATCH_SIZE, seed=42)
validate_data= tf.keras.preprocessing.image_dataset_from_directory(directory=(r"C:\Users\LENOVO\Desktop\Ayush Assignments\chest-xray\medical\training\chest_xray\val"), shuffle=True,image_size=(IMAGE_SIZE,IMAGE_SIZE), color_mode="grayscale",batch_size=BATCH_SIZE, seed=42)

class_names = train_data.class_names
plt.figure(figsize=(10,10))
for images, labels in train_data.take(1):
	for i in range(12):
		ax = plt.subplot(4,3,i+1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(class_names[labels[i]])
		plt.axis("off")


categories = ["Normal", "Pneumonia"]

# Plot Image Distribution
categories = ['Normal', 'Pneumonia']
frequencies =(train_data['class'].value_counts())[::-1]
plt.bar(categories, frequencies)
plt.xlabel("Categories")
plt.ylabel("Count")
plt.title(f'Data Distribution')
plt.show()

# As shown in the histogram, the dataset is very imbalanced. The dataset is heavily biased towards the pneumonia class, with roughly 3 times as many pneumonia 
# chest images as normal chest images. This is not very surprising, given that medical data is typically imbalanced. Given this heavy imbalance of pneumonia cases, 
# we want to make sure to adjust our classifier for this imbalance.


