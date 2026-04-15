from hw1.model import MLP
from hw1.utils import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix,ConfusionMatrixDisplay

CLASS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

test_img, test_label = load_mnist("data/FashionMNIST", kind="t10k")
model = MLP()
state_dict = load_model("models/model_relu.json")
model.load(state_dict)

input_val = Value(test_img)
y_pred = model.forward(input_val)
acc = accuracy(y_pred, test_label)
print(f"Test Accuracy: {acc:.4f}")

micro_f1 = f1_score(test_label, np.argmax(y_pred.data, axis=1), average='micro')
macro_f1 = f1_score(test_label, np.argmax(y_pred.data, axis=1), average='macro')
print(f"Test Micro F1 Score: {micro_f1:.4f}, Test Macro F1 Score: {macro_f1:.4f}")


pred_labels = np.argmax(y_pred.data, axis=1) 
cm = confusion_matrix(test_label, pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS)
disp.plot(
    ax = ax,
    cmap=plt.cm.Blues,
    xticks_rotation='vertical',
    values_format='d'
)
plt.title("Confusion Matrix - FashionMNIST", fontsize=14)
plt.tight_layout()
plt.show()
