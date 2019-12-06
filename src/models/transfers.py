from torch import nn
from enum import Enum
from torchvision import models as tv_models


class PretrainedModelEnum(Enum):
    alexnet = 1
    resnet = 2
    vggnet = 3


def get_pretrained_model(model_enum: PretrainedModelEnum):
    if model_enum == PretrainedModelEnum.alexnet:
        return tv_models.alexnet
    if model_enum == PretrainedModelEnum.resnet:
        return tv_models.resnet152
    if model_enum == PretrainedModelEnum.vggnet:
        return tv_models.vgg19_bn
    raise IndexError("No pretrained model for " + model_enum.name)


def get_transfer_method(model_enum: PretrainedModelEnum):
    if model_enum == PretrainedModelEnum.alexnet:
        return final_layer_alteration_alexnet
    if model_enum == PretrainedModelEnum.resnet:
        return final_layer_alteration_resnet
    if model_enum == PretrainedModelEnum.vggnet:
        return final_layer_alteration_vggnet
    raise IndexError("No transfer method for " + model_enum.name)


def final_layer_alteration_alexnet(net, num_classes):
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def final_layer_alteration_resnet(net, num_classes):
    net.fc = nn.Linear(2048, num_classes)
    return net


def final_layer_alteration_vggnet(net, num_classes):
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net
