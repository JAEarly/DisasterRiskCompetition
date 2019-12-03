from torch import nn

import features
import models
from features import BalanceMethod
from features import FeatureDatasets, ImageDatasets, SmoteExtractor
from models import PretrainedModelEnum, PretrainedNNModel, transfers
from models import XGBModel


class ModelIterator:
    def __init__(
        self, balance_method=BalanceMethod.NoSample, override_balance_methods=False
    ):
        self.balance_method = balance_method
        self.override_balance_methods = override_balance_methods

        self.feature_extractors = [
            features.AlexNet,
            features.AlexNetSMOTE,
            features.AlexNetCustom,
            features.AlexNetCustomSMOTE,
            features.ResNet,
            features.ResNetSMOTE,
            features.ResNetCustom,
            features.ResNetCustomSMOTE,
            features.VggNet,
        ]
        self.nn_models = [models.LinearNN, models.BiggerNN]
        self.pretrained_models = [
            PretrainedModelEnum.alexnet,
            PretrainedModelEnum.resnet,
            PretrainedModelEnum.vggnet,
        ]
        self.training_methods = ["grid_search", "kfold"]
        self.datasets = [FeatureDatasets, ImageDatasets]

        self.model_list = []
        self._add_feature_nn_models()
        #self._add_retrained_nn_models()
        self._add_xgb_models()

        self.idx = 0

    def _add_feature_nn_models(self):
        for training_method in self.training_methods:
            for feature_extractor in self.feature_extractors:
                for nn_model in self.nn_models:
                    self.model_list.append(
                        (training_method, nn_model, FeatureDatasets, feature_extractor)
                    )

    def _add_retrained_nn_models(self):
        for training_method in self.training_methods:
            for pretrained_model_enum in self.pretrained_models:
                self.model_list.append(
                    (training_method, pretrained_model_enum, ImageDatasets, None)
                )

    def _add_xgb_models(self):
        for training_method in self.training_methods:
            for feature_extractor in self.feature_extractors:
                self.model_list.append(
                    (training_method, XGBModel, FeatureDatasets, feature_extractor)
                )

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.model_list)

    def __next__(self):
        if self.idx == len(self.model_list):
            raise StopIteration
        model_data = self.model_list[self.idx]
        self.idx += 1
        training_method = model_data[0]
        model_class = model_data[1]
        datasets_type = model_data[2]

        try:
            # Feature dataset
            if datasets_type == FeatureDatasets:
                feature_extractor = model_data[3]()
                datasets = datasets_type(
                    feature_extractor,
                    self.balance_method,
                    self.override_balance_methods,
                )
                extractor_name = feature_extractor.name
                if issubclass(type(feature_extractor), SmoteExtractor):
                    extractor_name += "_smote"

                path = "./models/" + training_method + "_" + extractor_name + "_"

                # NN models
                if issubclass(model_class, nn.Module):
                    model = models.NNModel(
                        model_class,
                        feature_extractor.feature_size,
                        state_dict_path=path
                        + model_class.__name__.lower()
                        + "/best.pth",
                        eval_mode=True,
                    )
                    desc = training_method + "_" + feature_extractor.name + "_" + model_class.__name__.lower()

                # XGB
                else:
                    model = XGBModel(model_path=path + "xgb/best.pth")
                    desc = training_method + "_" + feature_extractor.name + "_xgb"

            # ImageDataset
            else:
                path = (
                    "./models/"
                    + training_method
                    + "_"
                    + model_class.name
                    + "_custom/best.pth"
                )
                pretrained_model = transfers.get_pretrained_model(model_class)
                transfer_function = transfers.get_transfer_method(model_class)
                model = PretrainedNNModel(
                    pretrained_model,
                    transfer_function,
                    state_dict_path=path,
                    eval_mode=True,
                )
                datasets = datasets_type()
                desc = training_method + "_" + model_class.name + "_custom"

            return model, datasets, desc
        except Exception as e:
            print(e)
            return next(self)


# Test iterator
if __name__ == "__main__":
    for _, _, _desc in ModelIterator():
        print(_desc)
