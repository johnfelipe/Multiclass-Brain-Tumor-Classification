class Model:

    def __init__(self, train_data_augmented, test_data):
        self.train_data_augmented = train_data_augmented
        self.test_data = test_data

    def model_1(self):
        # Early Stopping set to patience of 1
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)  # patience given in paper

        # Build a CNN model
        model_1 = tf.keras.models.Sequential([

            # 1.
            tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=3,
                                   activation="relu",
                                   input_shape=(224, 224, 1)),
            tf.keras.layers.Dense(1028),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(4, activation="softmax")])

        # Compile the model

        model_1.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy'])

        model_1.summary()

        # Fit the model

        history_1 = model_1.fit(self.train_data_augmented, epochs=15,
                                steps_per_epoch=len(self.train_data_augmented),
                                validation_data=self.test_data,
                                validation_steps=len(self.test_data),
                                callbacks=[callback])

        # Save the model
        model_1.save(os.path.join(d, 'project/volume/models') + "/CNN_V2.hdf5")
        print('Model_1 Saved')
        return model_1, history_1

    def loss_curves(self, model_1, history_1):
        """
        Returns the loss curves for training and validation metrics
        :param model_1:
        :param history_1:
        :return:
        """

        loss = history_1.history['loss']
        val_loss = history_1.history['val_loss']
        epochs = range(len(history_1.history['loss']))

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(epochs, loss, label="Training Loss")
        ax.plot(epochs, val_loss, label="Validation Loss")
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.legend()
        fig.savefig(os.path.join(d, 'project/volume/images') + "/CNN_V2_loss.png")
        print('loss saved')

        return fig

    def accuracy_curves(self, model_1, history_1):
        """
        Returns the accuracy curves for training and validation metrics
        :param model_1:
        :param history_1:
        :return:
        """

        accuracy = history_1.history['accuracy']
        val_accuarcy = history_1.history['val_accuracy']
        epochs = range(len(history_1.history['loss']))

        fig2, ax = plt.subplots()
        ax.plot(epochs, accuracy, label="Training Accuracy")
        ax.plot(epochs, val_accuarcy, label="Validation Accuracy")
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.legend();
        fig2.savefig(os.path.join(d, 'project/volume/images') + "/CNN_V2_accuracy.png")
        print('accuracy saved')

        return fig2

if __name__ == "__main__":
    import tensorflow as tf
    from path import Path
    import sys
    import os
    import matplotlib.pyplot as plt
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
    print(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    d = Path(__file__).parent.parent.parent.parent
    print(d)
    Path(d).chdir()
    os.chdir(d)
    from features import modify

    data = os.path.join(d, 'project/volume/data/raw/archive')
    build = modify.ModifyBuild(data=data)
    train_data_augmented, test_data = build.create_paths()
    model1 = Model(train_data_augmented, test_data)
    cnn_model, history_1 = model1.model_1()
    loss = model1.loss_curves(cnn_model, history_1)
    accuracy = model1.accuracy_curves(cnn_model, history_1)
