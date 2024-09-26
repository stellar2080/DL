import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import json

from tqdm import tqdm

from model import AlexNet


def main():
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: {}".format(device))

    # data_transform
    transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        "test":transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # data_path
    folder_root = os.getcwd()
    image_path = os.path.join(folder_root,"data_set","flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # data_set
    train_set = datasets.ImageFolder(os.path.join(image_path,"train"),
                                     transform=transform["train"])
    test_set = datasets.ImageFolder(os.path.join(image_path,"test"),
                                    transform=transform["test"])
    print("using {} images to train,{} images to test.".format(len(train_set),len(test_set)))

    # classes
    flower_dict = train_set.class_to_idx
    classes_dict = dict((idx,cls) for idx,cls in enumerate(flower_dict))
    json_tmp = json.dumps(classes_dict,indent=4)
    json_path = os.path.join(folder_root,"data_set","flower_data","class_to_idx.json")
    with open("./classes_dict.json","w") as f:
        f.write(json_tmp)
    print("classes: {}".format(classes_dict))
    print("classes_dict.json file save at {}".format(json_path))

    # data_loader
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size
                                               ,shuffle=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=4
                                              ,shuffle=False,num_workers=0)

    # net
    net = AlexNet(num_classes=5,init_weights=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 10
    best_acc = 0.0
    save_path = "./AlexNet.pth"
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0 # loss of one epoch
        start_time = time.perf_counter()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step,data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device)) # [N,classes_num]
            loss = loss_fn(outputs,labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "epoch[{}/{}] training...".format(epoch+1,epochs)

        print("epoch[{}/{}] finished training. take_time: {:.2f}".format(
                epoch+1,epochs,time.perf_counter() - start_time))

        net.eval()
        acc_num = 0
        start_time = time.perf_counter()
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data in test_bar:
                images, labels = data
                outputs = net(images.to(device)) # [N,classes_num]
                predict_y = torch.max(outputs, dim=1)[1]
                acc_num += torch.eq(predict_y, labels.to(device)).sum().item()
                test_bar.desc = "epoch[{}/{}] testing...".format(epoch+1,epochs)
            acc_rate = acc_num / len(test_set)
            if acc_rate > best_acc:
                best_acc = acc_rate
                torch.save(net.state_dict(),save_path)
        print("epoch[{}/{}] finished testing. take_time: {:.2f}".format(
            epoch + 1, epochs, time.perf_counter() - start_time))
        print("epoch[{}/{}] train_loss: {:.3f} test_acc: {:.3f}".format(
            epoch + 1, epochs, running_loss/len(train_loader), acc_rate
        ))
    print("Finished training.")


if __name__ == "__main__":
    main()