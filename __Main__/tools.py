#Imports
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Subset

#General/other
from typing import Set, Tuple, List
import numpy as np
import random, time
from pathlib import Path
import os, math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Useful functions
def augment_training(data_folder: ImageFolder, split: int, noise_seed: int = 547, noise_multiplier: float = 0.6) -> ImageFolder:
    """
    Augments the given data by adding random noise into it. Returns merge of original data and augmented data (as an ImageFolder).

    Use noise_seed to manually set the random seed used to add the noise. use the noise_multiplier to define how strong the noise is

    The data is subset up to the given split
    """

    #Splitting given ImageFolder
    split_data: Subset = Subset(data_folder, list(range(0, min(split, len(data_folder)))))

    #Setting noise seed
    torch.manual_seed(noise_seed)

    # Adding noise, printing some examples
    count: int = 0
    data_to_add: List[Tuple] = list() #saves all image configurations to add to ImageFolder
    for index, data in enumerate(split_data, 0): #iterating given ImageFolder
        image_data_list: List[object] = list() #List to store elements into

        img_class: int = data[1]
        image: Tensor = data[0]
        
        #Adding noise to image
        noisy_img = image + noise_multiplier * torch.rand(*image.shape) #creating random tensor matching shape of image, adding random noise
        noisy_img = np.clip(noisy_img, -1., 1.)

        #printing noisy image
        if count != 4 and index % 100 == 0:
            count += 1
            noisy_plot = np.transpose(noisy_img, [1,2,0]) #resetting image channel back to the end
            noisy_plot = noisy_plot*0.5 + 0.5
            plt.subplot(2,2, count)
            plt.imshow(noisy_plot)

        #Adding elements to datapoint
        image_data_list.append(noisy_img)
        image_data_list.append(img_class)
        image_data: Tuple[object] = tuple(image_data_list)#image data tuple

        data_to_add.append(image_data) #adding entry to list of entries to add

    #Adding entries to data folder
    split_data = split_data + data_to_add

    print(f"Training dataset length with noisy images added: {len(split_data)}")

    return split_data

def plot_training_curve(path: str):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/accuracy.

    Args:
        path: The base path of the csv files produced during training
    """
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def get_model_name(name: str, batch_size: int, learning_rate: float, epoch: int, dropout_prob: float):
    """ Generate a name for the model consisting of all the hyperparameter values.

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_dr{1}_lr{2}_epoch{3}_bs{4}".format(name,
                                                   dropout_prob,
                                                   learning_rate,
                                                   epoch,
                                                   batch_size)
    return path

def get_correct(outputs: Tensor, labels: Tensor) -> int:
    """
    Finds number of matching inferences to expected labels.

    Assumes that outputs is given as the raw output from the neural network's forward function. 
    """
    #Getting predictions from outputs:
    preds: Tensor = outputs.max(1, keepdim=True).indices.squeeze(dim=1) #Checks max probability for each inference, keeps the indices (classes) predicted and
    # reshapes to match the labels' shape (squeezes to a one dimensional Tensor)
    
    correct: int = int(preds.eq(labels).sum().item()) #checks how many predictions match labels. Adds up results and extracts count of matches.

    return correct

def get_model_path(model: nn.Module, lr: float, batch_size: int, epoch: int, at_epoch: int = 0) -> Path:
    if at_epoch == 0:
        at_epoch = epoch

    return Path(os.getcwd())/'training_data'/f'{
        get_model_name(model.name,
                batch_size=batch_size, learning_rate=lr, epoch=epoch, dropout_prob=model.dropout_prob)
        }'/f'{
        get_model_name(model.name,
                batch_size=batch_size, learning_rate=lr, epoch=at_epoch-1, dropout_prob=model.dropout_prob)
        }'

def evaluate(net: nn.Module, loader: DataLoader, criterion: nn.Module, use_gpu: bool) -> Tuple[float, float]:
    """
    Evaluate the network on the given loader.

    Returns the total loss and accuracy of the model, on the given loader.
    """

    total_loss: float = 0.0
    total_acc: int = 0
    total_epoch: int = 0
    for i, data in enumerate(loader, 0):
        # Get the inputs
        images: Tensor = data[0]
        labels: Tensor = data[1]

        #Checking if GPU training is available/selected
        if use_gpu and torch.cuda.is_available:
            images = images.cuda()
            labels = labels.cuda()

        #Making inference, getting results
        outputs: Tensor = net(images)
        loss: Tensor = criterion(outputs, labels)
        total_acc += get_correct(outputs, labels)
        total_loss += loss.item()
        total_epoch += len(labels)

    acc: float = float(total_acc) / total_epoch
    loss: float = total_loss / (i + 1)
    return acc, loss

def seed_worker(worker_id: int) -> None:
    worker_seed: int = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_randomized_loaders(
        data_path: str, #path to search for images/classes
        batch_size: int, #batch size to use for loaders
        shuffle_seed: int = 527, #for reproducibility; defaults to 527
        splits: List[float] = [0.8, 0.1, 0.1], #split to use for training, validation, testing
        n_workers: int = 0, #number of workers to use for loaders
        augment: bool = True #Whether to augment training data
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns a randomized split of the given images_folder data into three loaders (one for training, one for validation, and one for testing).
    Data is shuffled and then split into the three loaders, with the given splits.

    data_path: Path to the images we would like to load and split.
    shuffle_seed: Seed used to shuffle the ImageFolder data; used to determine the new order after shuffling.
    splits: Percentage to assing to training, to validation, and to testing, respectively.
    batch_size: Batch size to use for the returned loaders (samples per batch).

    Returns tuple of three data loaders, in the order of: training loader, validation loader, testing loader.
    """

    #Verifying split integrity
    assert sum(splits) == 1.0, "The provided splits do not add up to one"

    #Defining transforms to apply to images
    trns: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    ) #Transforms images to tensors, then applies (img - 0.5)/0.5 to all channels to obtain data in range [-1, 1]

    #Getting data from provided path as an ImageFolder
    data_folder: ImageFolder = ImageFolder(root=data_path, transform=trns)
    data_indices: list = list(range(0, len(data_folder), 1)) #getting list of indices in loaded data

    #Setting up random seeds, and splitting data
    np.random.seed(shuffle_seed)
    np.random.shuffle(data_indices) #shuffling data indices
    training_split: int = int(len(data_indices) * splits[0]) #getting first position to split at (ends training data split)
    validation_split: int = int(len(data_indices) * sum(splits[:2])) #getting second position to split at (ends validation data split)
    
    g: torch.Generator = torch.Generator() #defining generic generator, to use with each DataLoader
    g.manual_seed(shuffle_seed) #setting generator seed, to help ensure reproducibility

    #Getting split indices, creating data loaders
    train_indices, validation_indices, testing_indices = data_indices[:training_split], data_indices[training_split:validation_split], data_indices[validation_split:]

    #Special settings for train dataset if augmenting data
    if augment: 
        augmented_training: ImageFolder = augment_training(data_folder, training_split, noise_seed=shuffle_seed)
        train_indices: list = list(range(0, len(augmented_training), 1))
        train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_indices, generator=g)
        train_loader: DataLoader = DataLoader(dataset=augmented_training, batch_size=batch_size,
                                              num_workers=n_workers, worker_init_fn=seed_worker, sampler=train_sampler)

    else:
        train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_indices, generator=g) #note the use of the custom generator we defined earlier
        train_loader: DataLoader = DataLoader(dataset=data_folder, batch_size=batch_size,
                                            num_workers=n_workers, worker_init_fn=seed_worker, sampler=train_sampler) #note the use of the worker init function
    validation_sampler: SubsetRandomSampler = SubsetRandomSampler(validation_indices, generator=g)
    validation_loader: DataLoader = DataLoader(dataset=data_folder, batch_size=batch_size,
                                        num_workers=n_workers, worker_init_fn=seed_worker, sampler=validation_sampler)
    test_sampler: SubsetRandomSampler = SubsetRandomSampler(testing_indices, generator=g)
    test_loader: DataLoader = DataLoader(dataset=data_folder, batch_size=batch_size,
                                        num_workers=n_workers, worker_init_fn=seed_worker, sampler=test_sampler)
    
    #Returning loaders
    return train_loader, validation_loader, test_loader

def save_features(loader: DataLoader, net: nn.Module, use_gpu: bool = True) -> List[Tuple[Tensor, Tensor]]:
    """
    Takes the given network structure and processes the given loader on it.

    Returns two tensors: one for the resulting features, and one for the true labels for the given features, in that order.

    This function will try to extract features directly from the model. If this fails, it will directly evaluate the model.
    """

    total_data: List[Tuple[Tensor, Tensor]] = list() #Defininig initial list to save all our results to

    if use_gpu and torch.cuda.is_available:
        net = net.cuda()

    for images, labels in loader: #Iterating batches in loader
        #Checking if GPU training is available/selected
        if use_gpu and torch.cuda.is_available:
            images: Tensor = images.cuda()
            labels: Tensor = labels.cuda()

        with torch.no_grad(): #stop keeping track of grads, no need to update model
            try:
                features: Tensor = net.features(images)
                print("Features extracted successfuly")
            except:
                features: Tensor = net(images)

        curr_data: Tuple = tuple([torch.from_numpy(features.detach().cpu().numpy()), labels]) #Creating current data tuple
        #Forces features tensor as a numpy array, then reads it back as a tensor

        total_data.append(curr_data) #Appending to total data list

    return total_data

def train_classifier(
        train_loader: List[Tuple[Tensor, Tensor]], #Custom made features, labels 'loaders'
        val_loader: List[Tuple[Tensor, Tensor]],
        net: nn.Module, #Network to use for training
        batch_size: int = 32, #Batch size for training
        learning_rate: float = 3*math.e**(-4), #Learning rate for training
        num_epochs: int = 40, #Number of epochs for training
        seed: int = 527, #Random seed
        use_gpu: bool = True, #Whether to use GPU for training or not
        weight_decay: float = 5e-4 #How much weight decay to use (lambda)
        ) -> None:
    #Performing CUDA checks
    if use_gpu and not next(net.parameters()).is_cuda:
        print("WARNING: Intended to use CUDA for training, but passed model not on CUDA.")

    if use_gpu and torch.cuda.is_available:
        print(f"Training on GPU '{torch.cuda.get_device_name()}'")
    elif use_gpu and not torch.cuda.is_available:
        print("GPU is not available, training with CPU instead")

    #Printing settings
    print(f"Batch size: {batch_size}\tLearning rate: {learning_rate}\tEpochs: {num_epochs}")
    try:
        print(f"Using dropout: {net.dropout_prob}")
    except:
        print("No dropout implemented")
    print(f"Weight decay: {weight_decay}")

    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(seed)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be the Cross Entropy Loss function, which combines softmax and Negative Log Likelihood loss
    # into one.
    # Optimizer will be SGD with Momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/accuracy
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()

    #creating savepath for epochs settings, error and loss stats
    training_data_path: Path = Path(os.getcwd())/'training_data'/get_model_name(name=net.name, batch_size=batch_size, learning_rate=learning_rate,
                                                                                epoch=num_epochs, dropout_prob=net.dropout_prob)
    #getting current working directory, then adding model data to path
    if not os.path.exists(training_data_path):
        os.makedirs(training_data_path) #creating dir to save data to

    #Iterating epochs, entering main training loop
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss: float = 0.0
        total_train_correct: int = 0
        total_epoch: int = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            features: Tensor = data[0]
            labels: Tensor = data[1]

            #Checking if GPU training is available/selected
            if use_gpu and torch.cuda.is_available:
                features = features.cuda()
                labels = labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs: Tensor = net(features)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #Calculate the statistics
            total_train_correct += get_correct(outputs, labels)
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_acc[epoch] = float(total_train_correct) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_acc[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, use_gpu=use_gpu)
        print(("Epoch {}: Train acc: {}, Train loss: {} |"+
               "Validation acc: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_acc[epoch],
                   train_loss[epoch],
                   val_acc[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path: Path = training_data_path/get_model_name(net.name, batch_size, learning_rate, epoch, net.dropout_prob) #adding file name to path
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

    return None

def plot_confusion_matrix(model: nn.Module, test_loader, class_names: List[str], fontsize=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix with smaller font size for labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")

    # Set font sizes for tick labels
    ax.tick_params(axis='both', labelsize=fontsize)

    plt.title("Confusion Matrix", fontsize=fontsize + 4)
    plt.show()

def scale_arr(arr: np.ndarray, min: float = 0.0, max: float = 1.0) -> np.ndarray:
    """
    Scales all values in given arr in the range of min and max values provided.

    Used to shape spectrograms
    """
    arr_min: float = arr.min()
    arr_max: float = arr.max()

    #Standardizing arr with min max normalization (all values will now be in the range 0 - 1)
    x_std: np.ndarray = (arr - arr_min) / (arr_max - arr_min)

    #Scaling with given max, min
    x_scaled: np.ndarray = x_std * (max - min) + min

    return x_scaled

class genre_classifier(nn.Module):
    """
    This is a generalized classifier. Please provide output_size that matches the dimensions of your encoder's output.
    """
    def __init__(self, output_size: int, dropout_prob: float = 0.3, first_fc_dim: int = 4096, second_fc_dim: int = 512):
        super(genre_classifier, self).__init__()
        self.name = "genre_classifier"
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p = self.dropout_prob) #custom dropout prob; only used for fully connected classifier
        self.output_size = output_size

        #FC layers
        self.fc1 = nn.Linear(self.output_size, first_fc_dim)
        self.fc2 = nn.Linear(first_fc_dim, second_fc_dim)
        self.softmax = nn.Linear(second_fc_dim, 10) #this layer simply transforms the last fully connected output tensor to have size ten (number of classes we have) for its last dimension

    def forward(self, x: Tensor):
        #Fully connected
        x = x.view(-1, self.output_size) #Stacking last pooling output
        x = F.relu(self.dropout(self.fc1(x))) #Applying fully connected layer, then dropout, then ReLU activation.
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.softmax(x) #Softmax is to be applied by the CrossEntroy loss fn during training, or directly apply softmax on output for testing
        return x