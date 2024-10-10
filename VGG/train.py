import json
import os
import sys

import torch
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import vgg

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {}".format(device))

    # transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # dataset
    folder_root = os.getcwd()
    image_path = os.path.abspath(os.path.join(folder_root, '..','data_set','flower_data'))
    assert os.path.exists(image_path),"image path {} does not exist".format(image_path)
    train_set = datasets.ImageFolder(
        root=os.path.join(image_path,'train'), transform=data_transforms['train'])
    test_set = datasets.ImageFolder(
        root=os.path.join(image_path,'test'), transform=data_transforms['test']
    )
    print("using {} images to train,{} images to test".format(len(train_set),len(test_set)))

    # classes
    flower_list = train_set.class_to_idx
    flower_dict = dict((idx,cls) for cls,idx in flower_list.items())
    flower_json = json.dumps(flower_dict,indent=4)
    json_path = os.path.abspath('class_indices.json')
    with open("class_indices.json","w") as f:
        f.write(flower_json)
    print("classes: {}".format(flower_dict))
    print("class_indices.json saved at {}".format(json_path))

    # data_loader
    batch_size =32
    train_loader = DataLoader(
        train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader = DataLoader(
        test_set,batch_size=batch_size,shuffle=False,num_workers=0)

    # train
    model_name = "vgg19"
    net = vgg(model_name,num_classes=5,init_weights=True)
    net.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)

    epochs = 30
    best_acc = 0
    save_path = './{}.pth'.format(model_name)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0 # loss of an epoch
        train_bar = tqdm(train_loader,file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_func(outputs,labels.to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss.item())

        net.eval()
        acc_num = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader,file=sys.stdout)
            for data in test_bar:
                images, labels = data
                outputs = net(images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc_num += torch.eq(predict_y,labels.to(device)).sum().item()
            acc = acc_num / len(test_set)
            print("[epoch {:d}] train_loss = {:.3f},test_acc = {:.3f}".format(
                epoch+1,running_loss/len(train_loader),acc
            ))
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(),save_path)
    print("Finished Training.")

if __name__ == '__main__':
    main()
