import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    test_dir = 'data/test'
    model_path = 'models/eye_classifier.h5'
    
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please execute train.py first.")
        return
        
    if not os.path.exists(test_dir) or len(os.listdir(os.path.join(test_dir, 'open_eyes'))) == 0:
        logger.error("Test dataset splits empty. Run train.py or dataset_prep.py first.")
        return
        
    # 1. Load model
    logger.info(f"Loading trained classifier: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Setup Data Generator (Normalizing only)
    test_datagen = ImageDataGenerator(rescale=1./255)
    img_size = (64, 64)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=16,
        class_mode='binary',
        classes=['open_eyes', 'closed_eyes'],
        shuffle=False # Do not shuffle for clean evaluation mapping
    )
    
    # 3. Predict metrics
    logger.info("Executing test data evaluation predictions...")
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator, verbose=1).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # 4. Generate Classification report
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(y_true, y_pred, target_names=['Open Eyes', 'Closed Eyes']))
    print("=======================================================\n")
    
    # 5. Plot and save Confusion Matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # 6. Plot and save ROC Curve
    plot_roc_curve(y_true, y_pred_probs)
    
    # 7. Plot and save Precision-Recall Curve
    plot_precision_recall_curve(y_true, y_pred_probs)
    
    logger.info("Evaluation metrics generated and saved successfully.")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Open Eyes', 'Closed Eyes']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Render digits inside matrix cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Class Label')
    plt.xlabel('Predicted Class Label')
    plt.tight_layout()
    
    plot_path = 'training/confusion_matrix.png'
    plt.savefig(plot_path, dpi=200)
    logger.info(f"Confusion Matrix saved to: {plot_path}")
    plt.close()

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = 'training/roc_curve.png'
    plt.savefig(plot_path, dpi=200)
    logger.info(f"ROC Curve saved to: {plot_path}")
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = 'training/precision_recall.png'
    plt.savefig(plot_path, dpi=200)
    logger.info(f"Precision-Recall Curve saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
