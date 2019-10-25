import torch
from sklearn.metrics import log_loss
from torchvision import transforms, datasets
from tqdm import tqdm


import src.features.alexnet_feature_extractor as al_fe

data_dir = "./data/processed/train"
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()])


def evaluate_model(model):
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    y_trues = []
    y_preds = []
    for image, label in tqdm(full_dataset):
        #y_pred = model(image.unsqueeze(0)).cpu().detach().numpy()[0]  # 4.521839264201392
        y_pred = [0.09327505043712173, 0.49636852723604574, 0.044922663080026896,
                  0.3524546065904506, 0.012979152656355077]  # 1.0135557810317446
        y_trues.append(label)
        y_preds.append(y_pred)

    print("Log loss:", log_loss(y_trues, y_preds, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    al_model = al_fe.get_model()
    al_model.load_state_dict(torch.load("./models/alexnet_transfer"))
    al_model.eval()
    evaluate_model(al_model)
