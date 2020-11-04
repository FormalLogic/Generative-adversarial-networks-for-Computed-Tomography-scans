"""
Code for the classifier of healthy vs hemorrhagic scans.
For the most part, taken from HA1.
"""
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, CenterCrop, Normalize
from torch import nn
from torch import optim
import torch.nn.functional as F


class ScansClassifierNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.fc1 = nn.Linear(13 * 13 * 20, 300)
        self.fc2 = nn.Linear(300, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.reshape(-1, 13 * 13 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def load_images():
    train_path = '../tiny_dataset_binclass/train'
    val_path = '../tiny_dataset_binclass/validation'
    test_path = '../pictures/split_cdcgan'
    train_tensor = ImageFolder(Path.cwd() / train_path, transform=Compose([Resize(size=68),
                                                                           CenterCrop(64),
                                                                           Grayscale(num_output_channels=1),
                                                                           ToTensor(),
                                                                           Normalize([0.5], [0.5])]))
    val_tensor = ImageFolder(Path.cwd() / val_path, transform=Compose([Resize(size=68), CenterCrop(64),
                                                                       Grayscale(num_output_channels=1),
                                                                      ToTensor(),
                                                                      Normalize([0.5], [0.5])]))
    test_tensor = ImageFolder(Path.cwd() / test_path, transform=Compose([Resize(size=68), CenterCrop(64),
                                                                       Grayscale(num_output_channels=1),
                                                                       ToTensor(),
                                                                       Normalize([0.5], [0.5])]))
    train_loader = DataLoader(train_tensor, batch_size=128, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=len(val_tensor), num_workers=4)
    test_loader = DataLoader(test_tensor, batch_size=len(test_tensor), num_workers=4)
    print(train_tensor.classes)
    print(val_tensor.classes)
    print(test_tensor.classes)
    return train_loader, val_loader, test_loader, train_tensor, val_tensor, test_tensor


def build_model():
    # From HA2
    # Compute weights for each class
    n_instances = [1019, 889]
    ns = torch.tensor(n_instances, dtype=torch.float)
    w = 1.0 / ns
    w = w / w.sum()

    model = ScansClassifierNetwork()
    model.load_state_dict(torch.load('brain_classifier.pt'))
    loss_fn = nn.NLLLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return model, loss_fn, optimizer


def train_and_evaluate_model(data_loader, tensor_size, model, loss_fn, validate=False):
    losses = []
    n_correct = 0
    if validate:
        with torch.no_grad():
            for b_x, b_y in data_loader:
                pred = model(b_x)
                loss = loss_fn(pred, b_y)
                losses.append(loss.item())

                hard_preds = pred.argmax(dim=1)
                n_correct += torch.sum(hard_preds == b_y).item()
            val_accuracy = n_correct / tensor_size
            val_avg_loss = sum(losses) / len(losses)
        return val_accuracy, val_avg_loss
    else:
        for b_x, b_y in data_loader:
            pred = model(b_x)
            loss = loss_fn(pred, b_y)
            losses.append(loss.item())

            # Count number of correct predictions
            hard_preds = pred.argmax(dim=1)
            n_correct += torch.sum(hard_preds == b_y).item()

            # Backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute accuracy and loss in the entire training set
        train_accuracy = n_correct / tensor_size
        train_avg_loss = sum(losses) / len(losses)
        return train_accuracy, train_avg_loss


if __name__ == "__main__":
    total_train_losses = []
    total_val_losses = []
    total_train_accuracies = []
    total_val_accuracies = []
    model, loss_fn, optimizer = build_model()
    train_loader, val_loader, test_loader, train_tensor, val_tensor, test_tensor = load_images()
    for epoch in range(20):
        train_accuracy, train_avg_loss = train_and_evaluate_model(train_loader, len(train_tensor),
                                                                  model, loss_fn, validate=False)
        total_train_losses.append(train_avg_loss)
        total_train_accuracies.append(train_accuracy)

        # Compute accuracy and loss in the entire validation set
        val_accuracy, val_avg_loss = train_and_evaluate_model(val_loader, len(val_tensor),
                                                              model, loss_fn, validate=True)
        total_val_losses.append(val_avg_loss)
        total_val_accuracies.append(val_accuracy)

        # Display metrics
        display_str = 'Epoch {} '
        display_str += '\tLoss: {:.3f} '
        display_str += '\tLoss (val): {:.3f}'
        display_str += '\tAccuracy: {:.2f} '
        display_str += '\tAccuracy (val): {:.2f}'
        print(display_str.format(epoch, train_avg_loss, val_avg_loss, train_accuracy, val_accuracy))

    torch.save(model.state_dict(), 'brain_classifier.pt')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax[0].plot(total_train_losses, label='Train')
    ax[0].plot(total_val_losses, label='Validation')
    ax[0].set_title('Training and validation loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper right')

    ax[1].plot(total_train_accuracies, label='Train')
    ax[1].plot(total_val_accuracies, label='Validation')
    ax[1].set_title('Training and validation accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='lower right')

    fig.tight_layout()
    plt.show()
    n_correct = 0
    losses = []
    with torch.no_grad():
        model.eval()
        for b_x, b_y in test_loader:
            pred = model(b_x)
            loss = loss_fn(pred, b_y)
            losses.append(loss.item())

            hard_preds = pred.argmax(dim=1)
            n_correct += torch.sum(hard_preds == b_y).item()
        accuracy = n_correct / len(test_tensor)
        test_loss = sum(losses) / len(losses)
        print(f'Test accuracy: {accuracy:.2%}')
        print(f'Test loss: {test_loss}')
