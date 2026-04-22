from model import MLP
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

def train(model, train_img, train_label, valid_img,
           valid_label, batch_size, learning_rate, weight_decay, epoches):
    Train_loss = []
    Valid_loss = []
    Train_acc = []
    Valid_acc = []
    Model_snapshot = []

    n = len(train_img)
    with tqdm(range(epoches)) as epoch_bar:
        for epoch in epoch_bar:
            indices = np.random.permutation(n)
            train_img_shuffled = train_img[indices]
            train_label_shuffled = train_label[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            num_batches =  n // batch_size
            
            for i in range(num_batches):
                model.zero_grad()
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_imgs = train_img_shuffled[start_idx:end_idx]
                batch_labels = train_label_shuffled[start_idx:end_idx]
                
                input_val = Value(batch_imgs) 
                y_pred = model.forward(input_val)
                
                ce = cross_entropy(y_pred, batch_labels)
                ce.backward()
                
                ## Cosine learning rate decay
                lr_decay = (1 + np.cos(epoch / epoches * np.pi)) / 2
                current_lr = learning_rate * lr_decay

                ## Weight decay
                for name, para in model.named_parameters():
                    if "weight" in name:
                        para.grad += weight_decay * para.data

                model.update(current_lr)
                
                acc = accuracy(y_pred, batch_labels)
                epoch_loss += ce.data
                epoch_acc += acc

            y_valid = model.forward(Value(valid_img))
            valid_acc = accuracy(y_valid, valid_label)
            valid_loss = cross_entropy(y_valid, valid_label).data
            train_loss = epoch_loss / num_batches
            train_acc = epoch_acc / num_batches
            Train_loss.append(train_loss)
            Valid_loss.append(valid_loss)
            Train_acc.append(train_acc)
            Valid_acc.append(valid_acc)
            Model_snapshot.append(model.state_dict())

            # Update tqdm description with current metrics
            epoch_bar.set_postfix({
                'Loss': f"{train_loss:.4f}",
                'TrainACC': f"{train_acc:.4f}",
                'ValidACC': f"{valid_acc:.4f}"
            })
            
    return Train_loss, Valid_loss, Train_acc, Valid_acc, Model_snapshot

def draw_curve(train_loss, valid_loss, train_acc, valid_acc):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='red')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy', color='blue')
    plt.plot(valid_acc, label='Validation Accuracy', color='red')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def grid_search(model, train_img, train_label, valid_img, valid_label, param_grid):
    best_acc = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        model = MLP(hidden1=params['hiddenlayer'],hidden2=params['hiddenlayer'], nonlin=params['nonlin'])
        model.zero_grad()
        TL, VL, TA, VA, MD = train(
            model, train_img[:], train_label[:], valid_img, valid_label,
            64, params['LR'], params['weightdecay'], epoches=20
        )
        max_valid_acc = max(VA)
        if max_valid_acc > best_acc:
            best_acc = max_valid_acc
            best_params = params
        
        print()

    print(f"Best Parameters: {best_params}, Best Validation Accuracy: {best_acc:.4f}")
    return best_params

if __name__ == "__main__":
    np.random.seed(123)
    img,label = load_mnist("data/FashionMNIST", kind="train")
    total_n = len(img)
    shuffled_indices = np.random.permutation(total_n)
    train_n = int(total_n * 0.9)
    train_img = img[shuffled_indices[:train_n]]
    train_label = label[shuffled_indices[:train_n]]
    valid_img = img[shuffled_indices[train_n:]]
    valid_label = label[shuffled_indices[train_n:]]

    param_grid = {
    'LR': [1e-5, 1e-4],
    'weightdecay': [1e-5, 1e-4],
    'hiddenlayer': [128, 256],
    'nonlin': ["sigmoid","relu"]
    }

    best_params = grid_search(MLP(), train_img, train_label, valid_img, valid_label, param_grid)