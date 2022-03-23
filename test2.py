import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from project1_model import project1_model
from utils import progress_bar


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = project1_model().to(device)
# model_path = './project1_model.pt'
# checkpoint = torch.load(model_path, map_location=device)
# model.load_state_dict(checkpoint['net'], strict=False)

# best_acc = checkpoint['acc']
# print(best_acc)

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

# criterion = nn.CrossEntropyLoss()

# test_loss_history = []
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = project1_model().to(device)
    model_path = './project1_model.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=False)

    best_acc = checkpoint['acc']
    print(best_acc)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    test_loss_history = []
    
    model.eval()
    corrects = 0
    for batch_idx, (inputs, labels) in enumerate(testloader, 1):
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
    print(corrects.float() / len(testloader))




# pred = predicted_output.data.max(1)[1] # get the index of the max log-probability
#       test_corrects += pred.eq(labels.data).cpu().sum()
#       test_accuracy = 100. * test_corrects / len(testDataLoader.dataset)
