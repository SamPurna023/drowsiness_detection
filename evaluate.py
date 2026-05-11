import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(dataset_dir="dataset/prepared", model_path="drowsiness_model.h5"):
    test_dir = os.path.join(dataset_dir, 'test')
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Run setup_dataset.py first.")
        return
        
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Train the model first.")
        return

    print("Loading model...")
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False # Crucial for confusion matrix to map predictions to labels correctly
    )

    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    print("Generating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    class_labels = list(test_generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()
