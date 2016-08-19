from paradox.models.first_approach import get_model as FirstTest


def get_model(name, length_input_layer, number_of_classes):
    if name == 'first_approach':
        return FirstTest(length_input_layer, number_of_classes)
