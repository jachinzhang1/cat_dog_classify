import torch
from PIL import Image
from torchvision import transforms
from model import Model
import os


def test(model, image_path):
    model.eval()
    class_name = ['cat', 'dog']

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)  # add batch dimension

    output = model(image_tensor).item()
    result = 0 if output < 0.5 else 1
    print(f'Classified as a photo of a {class_name[result]}.')


def main():
    model = Model()
    model.load_state_dict(torch.load('./models/model-2.pth'))
    model.eval()
    test_set_path = 'D:/Data/Cat-Dog/test'
    print('=== Enter an image id (an integer between 1 and 12500) ===')
    while True:
        image_id = input()
        if image_id.isdigit() and 1 <= int(image_id) <= 12500:
            image_path = os.path.join(test_set_path, f'{image_id}.jpg')
            test(model, image_path)
        else:
            print('Invalid input.')
            break


if __name__ == '__main__':
    main()
