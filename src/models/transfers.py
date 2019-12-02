from torch import nn


def final_layer_alteration_alexnet(net, num_classes):
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def final_layer_alteration_resnet(net, num_classes):
    net.fc = nn.Linear(2048, num_classes)
    return net


def final_layer_alteration_vggnet(net, num_classes):
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net
