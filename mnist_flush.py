# Import required libraries.
import tensorflow as tf # Tensorflow backend.
from tensorflow.keras.callbacks import Callback # Callback object can perform actions at various steps of the model training.

from codecarbon import EmissionsTracker # CodeCarbon EmissionsTracker for embedding CodeCarbon functionality.

"""
This sample code shows how to use CodeCarbon as a Keras Callback
to save emissions after each epoch.
"""

class CodeCarbonCallBack(Callback):
    """
    CodeCarbonCallback implements a trainer callback that can customize the behaviour of the
    Keras training loop. The callback is called after each epoch.
    This callback is used for tracking the CO2 emissions of training after each epoch.
    """
    def __init__(self, codecarbon_tracker):
        self.codecarbon_tracker = codecarbon_tracker
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch during training.
        """
        self.codecarbon_tracker.flush() # CodeCarbon flush() API for registering Carbon emissions.

# Define the MNIST NumPy dataset for testing purposes.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the MNIST NumPy data and split it between train and test sets.
x_train, x_test = x_train / 255.0, x_test / 255.0 # Scale images to the [0, 1] range.

# Build the model.
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

# Define the loss function. SparseCategoricalCrossentropy is a good fit for classification tasks.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model to build the Neural Network. Computation graph is built during compilation.
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Embed CodeCarbon using EmissionsTracker.
tracker = EmissionsTracker()
tracker.start() # Start the CodeCarbon tracker.
codecarbon_cb = CodeCarbonCallBack(tracker) # Initialize the CodeCarbonCallback and pass the tracker as an argument.
# Train the model.
model.fit(x_train, y_train, epochs=4, callbacks=[codecarbon_cb]) # Hooking Callback object to Keras fit() method. The "codecarbon_cb" callback is passed as a keyword argument.
emissions: float = tracker.stop() # Stop the CodeCarbon tracker.
print(f"Emissions: {emissions} kg")
