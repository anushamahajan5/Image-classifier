#Doing the necessary imports
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json

#Defining the oarser arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', metavar='data_directory', type=str)
    parser.add_argument('--checkpoint_path', action='store', dest='checkpoint_path', type=str, default='trained_model.pth')
    parser.add_argument('--model_arch', action='store', dest='model_arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--training_epochs', action='store', dest='training_epochs', type=int, default=1)
    parser.add_argument('--activation_function', action='store', dest='activation_function', type=str, default='tanh', choices=['relu', 'tanh'])
    parser.add_argument('--use_gpu', action='store_true', default=False)
    return parser.parse_args()

def load_and_transform_data(data_directory):
    train_directory = data_directory + '/train'      #defining the format of loading the data
    validation_directory = data_directory + '/valid'
    test_directory = data_directory + '/test'

    common_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Training transform with additional augmentations
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        common_transform
    ])
    
    # Testing transform
    testing_transform = common_transform
    # Validation transform
    validation_transform = common_transform

    #loading the datasets in the same functions
    image_datasets = [
        datasets.ImageFolder(train_directory, transform=training_transform),
        datasets.ImageFolder(validation_directory, transform=validation_transform),
        datasets.ImageFolder(test_directory, transform=testing_transform)
    ]

    data_loaders = [
        torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)
    ]

    return image_datasets, data_loaders

#defining a model for our network
def construct_model(architecture, hidden_units, use_gpu,activation_function):
    if use_gpu:
        device = torch.device("cuda")                              #decide between gpu/cpu depending on the instruction given
    else:
        device = torch.device("cpu")

    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        num_input_features = 25088
    else:
        model = models.densenet121(pretrained=True)
        num_input_features = 1024

    for param in model.parameters():
        param.requires_grad = False

    if activation_function == 'tanh':                                  #using tanh as standard function else Relu function if specified
        activation = nn.Tanh()
    else:
        activation = nn.ReLU()

    custom_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_input_features, hidden_units)),                     #First input layer of model
        ('activation1', activation),                                              #Activation function after the first input layer
        ('fc2', nn.Linear(hidden_units, hidden_units)),                           #Defining the hidden layer of function
        ('activation2', activation),                                              #Activation function after the first hidden layer
        ('fc3', nn.Linear(hidden_units, 102)),                                    #Defining the oputput layer with 102 outputs
        ('output', nn.LogSoftmax(dim=1))                                          #Softmax function following the output layer
    ]))

    model.classifier = custom_classifier
    model.to(device)
    return model, device, num_input_features


#Now we define a function to train our model
def train_custom_model(model, device, training_loader, validation_loader, criterion, optimizer, epochs,print_every):

    start = time.time()
    print('Training the Model...')
    
    #Initializing the losses
    ct = 0
    curr_loss = 0
    
    #Using the standard epochs for training
    for epoch in range(epochs):
        for inputs, labels in training_loader:
            ct += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model.forward(inputs)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()

            if ct % (print_every*2) == 0:
                avg_train_loss = curr_loss / print_every
                val_loss, accuracy = validate_custom_model(model, device, validation_loader, criterion)
                #Priniting the accuracy after each epoch
                print("Epoch {}/{}.. Average Train loss: {:.4f}.. Validation loss: {:.4f}.. Validation accuracy: {:.4f}".format(epoch + 1, epochs, avg_train_loss, val_loss, accuracy))
                curr_loss = 0

    total_time = time.time() - start
    print("Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))

#For checking on validation data  
def validate_custom_model(model, device, validation_loader, criterion):
    
    #Initializing predictions and losses
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_probs = model.forward(inputs)
            loss = criterion(log_probs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(log_probs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    model.train()

    return total_loss / len(validation_loader), accuracy


#Saving the final model using standard method
def save_model_checkpoint(checkpoint_path, model, image_datasets, epochs, optimizer, learning_rate, num_input_features, architecture, hidden_units):
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint_bundle = {
        'model_architecture': architecture,
        'learning_rate': learning_rate,
        'number_of_hidden_units': hidden_units,
        'num_input_features': num_input_features,
        'optimizer_state_dict': optimizer.state_dict(),
        'num_output_features': len(image_datasets[0].classes),
        'classifier': model.classifier,
        'number_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }

    torch.save(checkpoint_bundle, checkpoint_path)
    print("Model Saved ;-)")

    #Defining the main function and making all calls
def main():
    print("After Loading and Transforming data")
    args = parse_arguments()
    image_datasets, data_loaders = load_and_transform_data(args.data_directory)

    custom_model, device, num_input_features = construct_model(args.model_arch, args.hidden_units, args.use_gpu, args.activation_function)

    criterion = nn.NLLLoss()   #defining the loss criteria
    optimizer = optim.Adam(custom_model.classifier.parameters(), lr=args.learning_rate) #using the standard Adam optimizer for our model

    train_custom_model(custom_model, device, data_loaders[0], data_loaders[1], criterion, optimizer, args.training_epochs,5)

    save_model_checkpoint(args.checkpoint_path, custom_model, image_datasets, args.training_epochs, optimizer, args.learning_rate, num_input_features,102, args.model_arch, args.hidden_units)
if __name__ == "__main__":
    main()