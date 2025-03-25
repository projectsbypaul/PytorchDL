import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm  # pip install tqdm

from dl_torch.models.TutorialModels import SimpleMLP

from dl_torch.models.TutorialModels import ConvNet

def train_on_mnist() -> None:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 784  # 28x28
    hidden_size = 500
    num_classes = 10
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/datasets',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data/datasets',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    '''
    examples = iter(test_loader)
    example_data, example_targets = next(examples)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(example_data[i][0], cmap='gray')
    plt.show()
    '''

    model = SimpleMLP(input_size, hidden_size, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass and loss calculation
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # Test the model: we don't need to compute gradients
    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)

            # max returns (output_value ,index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        acc = n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test images: {100 * acc} %')

def train_on_cifar10() -> None:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Hyper-parameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # dataset has PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/datasets', train=True,
                                                 download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='../../data/datasets', train=False,
                                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)

    print(f"Train on {len(train_loader.dataset)} samples, validate on {len(test_loader.dataset)} samples")

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):

        running_loss = 0.0

        epoch_start = time.time()

        # Display epoch header
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        # Calculate average loss and time per epoch
        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        time_per_step = epoch_time / len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

        print(f"{len(train_loader.dataset)}/{len(train_loader.dataset)} [==============================] - "
              f"{epoch_time:.0f}s {time_per_step * 1e6:.0f}us/step - loss: {avg_loss:.4f} - val_loss: {avg_val_loss:.4f}")

    print('Finished Training')
    PATH = '../../data/model_weights/cnn.pth'
    torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        n_correct = 0
        n_correct2 = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the model: {acc} %')

    '''

    loaded_model = ConvNet()
    state_dict = torch.load(PATH)

    loaded_model.load_state_dict(state_dict)  # it takes the loaded dictionary, not the path file itself
    loaded_model.to(device)
    loaded_model.eval()
    
    '''

def test_on_cifar10() -> None:

    print("Evaluating Model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)
    
    #Set up data

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size: int = 32

    # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/datasets', train=True,
                                                 download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='../../data/datasets', train=False,
                                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)
    
    #load model

    PATH = '../../data/model_weights/cnn.pth'

    model = ConvNet()
    state_dict = torch.load(PATH)

    model.load_state_dict(state_dict)  # it takes the loaded dictionary, not the path file itself
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        n_correct = 0
        n_correct2 = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the model: {acc} %')



def main() -> None:
    # train_on_mnist()
    # train_on_cifar10()
    test_on_cifar10()

if __name__ == '__main__':
    main()