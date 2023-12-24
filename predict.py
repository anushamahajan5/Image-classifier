import argparse
import json
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as Fuc
from train import load_model as load_checkpoint_trazin

#defining the parse_args()
def parse_prediction_args():
    parser = argparse.ArgumentParser(description="Image Prediction")
    parser.add_argument('input_image', metavar='input_image', type=str, default='flowers/test/23/image_03390.jpg', help='Path to the input image for prediction')
    parser.add_argument('model_checkpoint', metavar='model_checkpoint', type=str, default='train_checkpoint.pth', help='Path to the model checkpoint file')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5, help='Choose top K predictions')
    parser.add_argument('--class_mapping', action='store', dest='class_mapping', type=str, default='cat_to_name.json', help='Path to the class-to-name mapping file')
    parser.add_argument('--enable_gpu', action='store_true', default=False, help='Enable GPU for prediction')
    return parser.parse_args()

#Normalizing the image between 0-1
def preprocess_input(image_path):
    image = Image.open(image_path)
    gray_image = image.convert('L')
    inverted_image = Image.eval(gray_image, lambda x: 255 - x)
    resized_image = inverted_image.resize((224, 224))
    input_array = np.array(resized_image) / 255.0
    input_array = (input_array - np.mean(input_array)) / np.std(input_array)
    return input_array

#Defing the format of model
def load_trained_model(checkpoint_path, architecture='vgg16', hidden_units=512, use_gpu=False):
    trained_model, _, _ = load_checkpoint_train(architecture, hidden_units, use_gpu)
    trained_model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    return trained_model

#Making final predictions
def make_prediction(input_image, trained_model, top_k, use_gpu):
    if use_gpu:
        device = torch.device("cuda")                 #Selection whether to use cpu or gpu
    else:
        device = torch.device("cpu")
    trained_model.to(device)

    input_image_tensor = torch.from_numpy(input_image)     #Converting image to tensor in numpy
    input_image_tensor = input_image_tensor.unsqueeze(0)
    input_image_tensor = input_image_tensor.float().to(device)

    with torch.no_grad():
        output = trained_model.forward(input_image_tensor)      

    probabilities = Fuc.softmax(output.data, dim=1)
    top_probabilities, top_classes = probabilities.topk(top_k)

    top_probabilities = top_probabilities.cpu().numpy().squeeze()
    top_classes = top_classes.cpu().numpy().squeeze()

    return top_probabilities, top_classes, device

#defining a function to give category name
def load_category_names(category_file):
    with open(category_file) as sample:
        category_names = json.load(sample)
    return category_names

#creating a function to display the output
def display_results(labels, probabilities):
    print("Top Predictions:")
    for i, (label, probability) in enumerate(zip(labels, probabilities)):
        print(f"{i + 1}. {label} with a probability of {probability:.3%}")         #Formatting how output is printed

#Creating a main function
def main():
    args = parse_prediction_args()
    input_image_path = args.input_image
    checkpoint_path = args.model_checkpoint
    top_k = args.top_k
    category_names_file = args.class_mapping
    use_gpu = args.enable_gpu

    input_image = preprocess_input(input_image_path)
    trained_model = load_trained_model(checkpoint_path, use_gpu=use_gpu)

    top_probabilities, classes, _ = make_prediction(input_image, trained_model, top_k, use_gpu)

    category_names = load_category_names(category_names_file)
    labels = list(map(lambda index: category_names[str(index)], classes))

    print(f"Final result for file: {input_image_path}")
    display_results(labels, top_probabilities)

if __name__ == "__main__":
    main()
