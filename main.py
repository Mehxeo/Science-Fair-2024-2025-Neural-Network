import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize MNIST data
x_train = x_train / 255.0
x_test = x_test / 255.0


# Load and preprocess custom dataset
def load_custom_data(folder):
    images = []
    labels = []
    image_number = 1

    while os.path.isfile(f"{folder}/digit{image_number}.jpg"):
        try:
            # Load the image in grayscale
            img = cv2.imread(f"{folder}/digit{image_number}.jpg", cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Image {folder}/digit{image_number}.jpg could not be loaded.")
                image_number += 1
                continue

            # Resize, invert, and normalize the image
            img_resized = cv2.resize(img, (28, 28))
            img_inverted = 255 - img_resized
            img_normalized = img_inverted / 255.0

            # Show the image and prompt the user for the label
            plt.imshow(img_normalized, cmap=plt.cm.binary)
            plt.title("Please input the correct label for this digit")
            plt.show()

            label = input(f"Enter the label for digit{image_number} (0-9): ")
            labels.append(int(label))
            images.append(img_normalized)

        except Exception as e:
            print(f"Error while processing image {folder}/digit{image_number}.jpg: {e}")
        finally:
            image_number += 1

    return np.array(images), np.array(labels)


# Load custom dataset
custom_images, custom_labels = load_custom_data('test3')

# Combine MNIST data with custom data
custom_images = custom_images.reshape(custom_images.shape[0], 28, 28)
x_combined = np.concatenate((x_train, custom_images), axis=0)
y_combined = np.concatenate((y_train, custom_labels), axis=0)

# Split the combined dataset into training and validation sets
x_train_combined, x_val, y_train_combined, y_val = train_test_split(
    x_combined, y_combined, test_size=0.2, random_state=42
)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the combined dataset
model.fit(x_train_combined, y_train_combined, validation_data=(x_val, y_val), epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy}")
print(f"Test Loss: {loss}")

# Save the retrained model
model.save('handwritten_retrained3.keras')
print("Retrained model saved as handwritten_retrained.keras")

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import time  # Import time module
#
# model = tf.keras.models.load_model('handwritten_retrained3.keras')
#
# # Iterate through the images in the digits folder
# image_number = 1
# while os.path.isfile(f"Experiment/digit{image_number}.jpg"):
#     try:
#         # Load the image in grayscale
#         img = cv2.imread(f"Experiment/digit{image_number}.jpg", cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             print(f"Image Experiment/digit{image_number}.jpg could not be loaded.")
#             image_number += 1
#             continue
#
#         # Resize the 960x960 image to 28x28
#         img_resized = cv2.resize(img, (28, 28))
#
#         # Invert colors (if the model assumes black digits on white background)
#         img_resized = np.invert(img_resized)
#
#         # Normalize pixel values to the range [0, 1]
#         img_resized = img_resized / 255.0
#
#         # Reshape the image to add a batch dimension
#         img_reshaped = img_resized.reshape(1, 28, 28)
#
#         # Start measuring time before making a prediction
#         start_time = time.time()
#
#         # Make a prediction
#         prediction = model.predict(img_reshaped)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#
#         # Measure time after prediction
#         end_time = time.time()
#         prediction_time = end_time - start_time
#
#         # Print the time it took for the prediction
#         print(f"Time taken for prediction: {prediction_time:.4f} seconds")
#
#         # Display the processed image
#         plt.imshow(img_resized, cmap=plt.cm.binary)
#         plt.title(f"Predicted: {np.argmax(prediction)}")
#         plt.show()
#
#     except Exception as e:
#         print(f"Error while processing image digits/digit{image_number}.jpg: {e}")
#     finally:
#         image_number += 1
