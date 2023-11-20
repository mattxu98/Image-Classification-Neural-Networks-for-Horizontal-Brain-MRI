import time
import torchvision.models as models
from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn

#Define a function to set up, train the model and output the results file
def run_experiment(hyps, identifier):
    ####Log part####
    import logging

    #Clear all existing log handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #Configure logging
    #Set the name of the output file
    model_filename = f'best_model_{identifier}.pth'
    last_model_filename = f'last_model_{identifier}.pth'
    training_log_filename = f'training_{identifier}.log'
    metrics_plot_filename = f'metrics_plots_{identifier}.png'
    confusion_matrix_filename = f'confusion_matrix_{identifier}.png'

    #Configure logging to use identifier
    logging.basicConfig(filename=training_log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    #If you want to see log messages in the console, you can add a stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    #... Set hyperparameters ...
    densenet = models.densenet121()
    num_features = densenet.classifier.in_features

    if hyps['activation'] == 'relu':
        activation = nn.ReLU()
    elif hyps['activation'] == 'sigmoid':
        activation = nn.Sigmoid()
    elif hyps['activation'] == 'tanh':
        activation = nn.Tanh()

    new_classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        activation,
        nn.Dropout(hyps['dropout']),
        nn.Linear(1024, 256), 
        activation,
        nn.Dropout(hyps['dropout']),
        nn.Linear(256, 2)  #The last fully connected layer outputs 2 categories
    )

    densenet.classifier = new_classifier
    model = densenet


    #... training code ...
#   Initialize lists to store metrics
    train_losses = []
    val_losses = []
    recalls = []
    f1s = []
    accuracies = []

    if hyps['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    elif hyps['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyps['lr'])

    criterion = nn.CrossEntropyLoss() #Loss, you can also set other settings, you can try more, but I guess the difference will not be big.
    num_epochs = 300 
    early_stopping = EarlyStopping(patience=100, min_delta=0.001)
    best_val_loss = float('inf') 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ####Start training####
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")


        validate_every = 1  #For example, setting n to 5 means verifying every 5 epochs.
        if (epoch + 1) % validate_every == 0:
            #Verification phase
            model.eval()
            total_val_loss = 0.0
            all_labels = []  
            all_predictions = []  
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)

                    outputs = model(imgs)
                    _, predicted = torch.max(outputs, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            accuracy = accuracy_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions)
            val_loss = total_val_loss / len(val_loader)
            val_losses.append(val_loss) 
            recalls.append(recall)  
            f1s.append(f1)              
            accuracies.append(accuracy)       
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}\n \
                        Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save the model achieving the best validation loss seen so far
                torch.save(model.state_dict(), model_filename)

        #Check at the end of each epoch if it should stop early
        early_stopping(avg_train_loss) # avg_train_loss -> val_loss
        if early_stopping.stop:
            logging.info("Early stopping triggered.")
            break

    # save the model at the end of training, regardless of performance
    torch.save(model.state_dict(), last_model_filename)

    #... draw and save diagrams ...
    epochs = range(1, len(train_losses) + 1)

    metrics = [
        ('Train loss and test loss', train_losses, val_losses, 'Loss'),
        ('Recall', recalls, None, 'Recall'),  # None indicates no second series to plot
        ('f1', f1s, None, 'F1 Score'),
        ('Accuracy', accuracies, None, 'Accuracy')
    ]

    plt.figure(figsize=(20, 4))
    for i, (title, y1, y2, ylabel) in enumerate(metrics, 1):
        plt.subplot(1, 4, i)
        if y2 is not None:
            plt.plot(epochs, y1, label='Train Loss')
            plt.plot(epochs, y2, label='Test Loss')
            plt.legend()
        else:
            plt.plot(epochs, y1, label=title)
        plt.title(f"{model.__class__.__name__}: {title}")
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'metrics_plots_{identifier}.png')  # Save the figure as a PNG file

    # Calculate the confusion matrix
    plt.figure(figsize=(3.5, 2))

    cm = confusion_matrix(all_labels, all_predictions).T
    cm = np.flipud(cm)  # Flip rows
    cm = np.fliplr(cm)  # Flip columns
    ax = sns.heatmap(cm, annot=True, fmt='d')

    cbar = ax.collections[0].colorbar  # Get colorbar from the axis object
    cbar.ax.tick_params(labelsize=7)  # Set fontsize for colorbar tick labels

    plt.ylabel('Predicted Label', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_yticklabels(['Positive', 'Negative'])  # Flip the tick labels

    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_label_position('top')  # Set the x-label to be top
    plt.xlabel('Actual Label', fontsize=7)
    ax.xaxis.tick_top()  # Set the x-axis ticks to be top
    ax.set_xticklabels(['Positive', 'Negative'])  # Flip the tick labels
    ax.tick_params(axis='x', labelsize=7)

    plt.title(f"{model.__class__.__name__}: Confusion Matrix", fontsize=9)
    plt.savefig(f'confusion_matrix_{identifier}.png')  #Save the figure as a PNG file


#List of hyperparameter combinations
hyperparameter_combinations = [
    {'lr': 0.001, 'dropout': 0.5, 'activation': 'relu', 'optimizer': 'adam'},

    {'lr': 0.01, 'dropout': 0.5, 'activation': 'relu', 'optimizer': 'adam'},
    {'lr': 0.1, 'dropout': 0.5, 'activation': 'relu', 'optimizer': 'adam'},
    
    {'lr': 0.001, 'dropout': 0.3, 'activation': 'relu', 'optimizer': 'adam'},
    {'lr': 0.001, 'dropout': 0.1, 'activation': 'relu', 'optimizer': 'adam'},

    {'lr': 0.001, 'dropout': 0.5, 'activation': 'sigmoid', 'optimizer': 'adam'},
    {'lr': 0.001, 'dropout': 0.5, 'activation': 'tanh', 'optimizer': 'adam'},

    {'lr': 0.001, 'dropout': 0.5, 'activation': 'relu', 'optimizer': 'sgd'},
]


class AddGaussNoise(object):
    def __init__(self, mean=0.0, std=1.):
        self._mean = mean
        self._std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self._std \
        + self._mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.\
        format(self._mean, self._std)

#Early stop function
class EarlyStopping:
    '''
    用于实现早停机制。早停机制是一种在训练过程中根据模型的表现来判断是否停止训练的方法。
    
    参数：
    patience：容忍的训练步数
    min_delta：最小的损失变化。

    __call__：用于在每次计算损失后更新最佳损失和计数器。当计数器达到容忍步数时，停止训练。
    '''
    def __init__(self, patience=100, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    AddGaussNoise(0.0, 0.1) # noise ~ N(0, 0.1)
])

dataset = ImageFolder(root='../data/', transform=transform)

train_size = int(0.8 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

for i, hyps in enumerate(hyperparameter_combinations):
    # Generate unique identifier
    identifier = f"{i}"

    #Call a function to run an experiment
    run_experiment(hyps, identifier)
