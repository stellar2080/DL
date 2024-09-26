import torch
from PIL import Image
from torchvision import transforms

from model import LeNet


def main():
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: {}", device)
    # data_transform
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # net
    net = LeNet().to(device)
    net.load_state_dict(torch.load("./LeNet.pth"))

    # test_data
    image = Image.open("./test.jpg")
    image = transform(image) # [C,H,W]
    image = torch.unsqueeze(image, dim=0) # [N,C,H,W]

    with torch.no_grad():
        output = net(image.to(device)) # [N,num_classes]
        output = torch.softmax(output, dim=1)
        predict_y = torch.max(output,dim=1)
        print("result: {}, probability: {:.3f}".format(classes[predict_y[1].item()], predict_y[0].item()))

if __name__ == "__main__":
    main()