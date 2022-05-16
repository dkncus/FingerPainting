import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Confusion Matrix Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


if __name__ == '__main__':

    RANDOM_SEED = 42

    dataset = 'gesture_model/model/keypoint_classifier/keypoint.csv'
    model_save_path = 'gesture_model/model/keypoint_classifier/keypoint_classifier.hdf5'

    NUM_CLASSES = 8

    # Load dataset
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    # Define model structure
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    print(model.summary())

    # Checkpointing Callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)

    # Early Stopping Callbacks
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model w/ callbacks
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )

    # Run model statistics
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

    # Create confusion matrix & Display
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    print_confusion_matrix(y_test, y_pred)

    # Save the trained model
    model.save(model_save_path, include_optimizer=False)

    # TFLite Conversion
    tflite_save_path = 'gesture_model/model/keypoint_classifier/keypoint_classifier.tflite'

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, 'wb').write(tflite_quantized_model)

    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    print(np.squeeze(tflite_results))
    print(np.argmax(np.squeeze(tflite_results)))