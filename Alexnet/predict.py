import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: {}".format(device))

    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ],)

    img_path = "./test.jpeg"
    assert os.path.exists(img_path),"path {} does not exist.".format(img_path)

    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,0)

    json_path = "./classes_dict.json"
    assert os.path.exists(json_path),"path {} does not exist.".format(json_path)
    classes_dict = dict(json.load(open(json_path,mode='r')))

    net = AlexNet(num_classes=5).to(device)

    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path),"path {} does not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path))

    net.eval()

    with torch.no_grad():
        output = torch.squeeze(net(img.to(device))).cpu() # 删掉batch_size这个维度
        output = torch.softmax(output,dim=0)
        predict_y = torch.max(output,dim=0)
        print("predict result: {} probability: {:.3f}".format(
            classes_dict[str(predict_y[1].item())], predict_y[0]))

    print("Finished predicting.")

if __name__ == '__main__':
    main()

