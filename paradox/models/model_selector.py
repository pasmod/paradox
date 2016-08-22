from paradox.models.first_approach import get_model as FirstTest
from paradox.models.multi_layer_approach import get_model as MultiLayerApproach
from paradox.models.lstm_approach import get_model as CNNApproach


def get_model(name, length_input_layer, number_of_classes):
    if name == 'first_approach':
        return FirstTest(length_input_layer, number_of_classes)
    elif name == 'multi_layer_approach':
        return MultiLayerApproach(length_input_layer, number_of_classes)
    elif name == 'cnn_approach':
        return CNNApproach(length_input_layer, number_of_classes)

