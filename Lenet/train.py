import sys
import time

import torch
from torch import nn as nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm

from model import LeNet

def main():
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: {}".format(device))

    # transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])

    # data_set
    train_set = datasets.CIFAR10(
        root="./data_set", train=True, download=False, transform=transform
    )
    test_set = datasets.CIFAR10(
        root="./data_set", train=False , download=False, transform=transform
    )
    print("using {} images to train, {} images to test".format(len(train_set),len(test_set)))

    # data_loader
    train_loader = DataLoader(
        train_set, batch_size=36, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=10000, shuffle=False, num_workers=0
    )

    # classes
    classes = ('plane', 'car', 'bird', 'cat','deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # net loss_function optimizer
    net = LeNet(num_classes=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    save_path = "./LeNet.pth"
    best_acc = 0.0
    epochs = 5
    # train test
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        start_time = time.perf_counter()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar, start=0):
            images, labels = data
            optimizer.zero_grad()

            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss: {:.3f}".format(
                epoch+1,epochs,loss)

        # validate/test
        net.eval()
        acc_num = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data in test_bar:
                test_images, test_labels = data
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc_num += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_acc = acc_num / len(test_set)
        print("[epoch {:d}] take_time: {:.2f} train_loss: {:.3f} test_acc: {:.3f}".format(
            epoch+1,time.perf_counter()-start_time, running_loss / len(train_set),test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), save_path)

    print("Finished Training")

if __name__ == "__main__":
    main()