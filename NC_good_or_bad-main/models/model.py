from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# Resnet
def resnet18(num_classes = 10, norm_layer_type = 'bn' ,conv_layer_type = 'conv2d',linear_layer_type = 'linear', activation_layer_type = 'relu', etf_fc = False):
    return ResNet18(num_classes=num_classes, norm_layer_type = norm_layer_type, conv_layer_type = conv_layer_type, linear_layer_type = linear_layer_type,
                    activation_layer_type = activation_layer_type, etf_fc = etf_fc)

