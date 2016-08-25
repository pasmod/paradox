from paradox.encoders.first_approach import get_model as FirstTest
from paradox.encoders.multi_layer_approach import get_model as MultiLayerApproach
from paradox.encoders.lstm_approach import get_model as LSTMApproach


def get_model(name, length_input_layer, number_of_classes):
    if name == 'first_approach':
        return FirstTest(length_input_layer, number_of_classes)
    elif name == 'multi_layer_approach':
        return MultiLayerApproach(length_input_layer, number_of_classes)
    elif name == 'lstm_approach':
        return LSTMApproach(length_input_layer, number_of_classes)

