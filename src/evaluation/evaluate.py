from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models import Model, BaselineModel, AlexNetModel

TEST_DATA_DIR = "./data/processed/train"
BATCH_SIZE = 8


def evaluate_model(model: Model):
    dataset = ImageFolder(TEST_DATA_DIR, transform=model.get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = []
    y_pred = []

    for batch, labels in tqdm(dataloader):
        y_true.extend(labels)
        y_pred.extend(model.predict_batch(batch))

    print("Log loss:", log_loss(y_true, y_pred, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    # Baseline - 1.01356
    # evaluate_model(BaselineModel())

    # AlexNet Linear - 4.52184
    evaluate_model(AlexNetModel(state_dict_path="./models/alexnet_transfer", eval_mode=True))
