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

    #... Set up hyperparameters and models ...
    class ConvNet(nn.Module):
        def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
            super(ConvNet, self).__init__()

            self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
            num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
            self.classifier = nn.Linear(num_feat, num_classes)

        def forward(self, x):
            # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out

        def _get_activation(self, net_act):
            if net_act == 'sigmoid':
                return nn.Sigmoid()
            elif net_act == 'relu':
                return nn.ReLU(inplace=True)
            elif net_act == 'leakyrelu':
                return nn.LeakyReLU(negative_slope=0.01)
            else:
                exit('unknown activation function: %s'%net_act)

        def _get_pooling(self, net_pooling):
            if net_pooling == 'maxpooling':
                return nn.MaxPool2d(kernel_size=2, stride=2)
            elif net_pooling == 'avgpooling':
                return nn.AvgPool2d(kernel_size=2, stride=2)
            elif net_pooling == 'none':
                return None
            else:
                exit('unknown net_pooling: %s'%net_pooling)

        def _get_normlayer(self, net_norm, shape_feat):
            # shape_feat = (c*h*w)
            if net_norm == 'batchnorm':
                return nn.BatchNorm2d(shape_feat[0], affine=True)
            elif net_norm == 'layernorm':
                return nn.LayerNorm(shape_feat, elementwise_affine=True)
            elif net_norm == 'instancenorm':
                return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
            elif net_norm == 'groupnorm':
                return nn.GroupNorm(4, shape_feat[0], affine=True)
            elif net_norm == 'none':
                return None
            else:
                exit('unknown net_norm: %s'%net_norm)

        def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
            layers = []
            in_channels = channel
            if im_size[0] == 28:
                im_size = (32, 32)
            shape_feat = [in_channels, im_size[0], im_size[1]]
            for d in range(net_depth):
                layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
                shape_feat[0] = net_width
                if net_norm != 'none':
                    layers += [self._get_normlayer(net_norm, shape_feat)]
                layers += [self._get_activation(net_act)]
                in_channels = net_width
                if net_pooling != 'none':
                    layers += [self._get_pooling(net_pooling)]
                    shape_feat[1] //= 2
                    shape_feat[2] //= 2


            return nn.Sequential(*layers), shape_feat

    model = ConvNet(channel=3, num_classes=2, net_width=128, net_depth=3, net_act=hyps['activation'], net_norm='batchnorm', net_pooling=hyps['pooling'], im_size=(128, 128))



    #... training code ...
    #Initialize lists to store metrics
    train_losses = []
    val_losses = []
    recalls = []
    f1s = []
    accuracies = []

    if hyps['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'])
    elif hyps['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyps['lr'])

    criterion = nn.CrossEntropyLoss() 
    num_epochs = 300
    early_stopping = EarlyStopping(patience=100, min_delta=0.001) 
    best_val_loss = float('inf') 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ####Start training####
    for epoch in range(num_epochs):
        #training phase
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
    {'lr': 0.001, 'activation': 'relu', 'optimizer': 'adam', 'pooling': 'maxpooling'},

    {'lr': 0.01, 'activation': 'relu', 'optimizer': 'adam', 'pooling': 'maxpooling'},
    {'lr': 0.1, 'activation': 'relu', 'optimizer': 'adam', 'pooling': 'maxpooling'},

    {'lr': 0.001, 'activation': 'sigmoid', 'optimizer': 'adam', 'pooling': 'maxpooling'},
    {'lr': 0.001, 'activation': 'leakyrelu', 'optimizer': 'adam', 'pooling': 'maxpooling'},

    {'lr': 0.001, 'activation': 'relu', 'optimizer': 'sgd', 'pooling': 'maxpooling'},

    {'lr': 0.001, 'activation': 'relu', 'optimizer': 'adam', 'pooling': 'avgpooling'},
    {'lr': 0.001, 'activation': 'relu', 'optimizer': 'adam', 'pooling': 'none'},
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
    #Generate unique identifier
    identifier = f"{i}"

    #Call a function to run an experiment
    run_experiment(hyps, identifier)