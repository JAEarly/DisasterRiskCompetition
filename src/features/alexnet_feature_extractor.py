import torch
import torchbearer
from torch import nn
from torch import optim
from torchbearer import Trial
from torchvision import models, transforms, datasets


data_dir = "./data/processed/train"
save_path = "./models/alexnet_transfer"
num_classes = 5
batch_size = 8
num_epochs = 1
test_proportion = 0.2
loss_function = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()])


def get_model():
    model = models.alexnet(pretrained=True)
    model.classifier = nn.Linear(9216, num_classes)
    return model


if __name__ == "__main__":
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    train_size = int((1 - test_proportion) * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model()
    model.train()

    for param in model.parameters():
        param.requires_grad = False
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True

    #only optimse non-frozen layers
    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
    trial.with_generators(train_loader, test_generator=test_loader)
    trial.run(epochs=num_epochs)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)

    torch.save(model.state_dict(), save_path)
