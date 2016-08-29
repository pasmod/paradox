import logging


class ResultLogger():
    def __init__(self):
        self.logger = logging.getLogger('result_logger')
        handler = logging.FileHandler('/var/www/logs/results.log')
        handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def log_result(self, localz, message=''):
        data_type = localz['data_type'].name
        localz['message'] = message
        prefix = "F1={avg_f1}, {message} {language}-{number_of_classes}, epochs={nb_epoch}".format(**localz)
        localz.pop('data_set')
        localz.pop('model')
        localz.pop('checkpoint')
        localz.pop('early_stop')
        localz.pop('avg_f1')
        localz.pop('language')
        localz.pop('number_of_classes')
        localz.pop('data_type')
        localz.pop('message')
        self.logger.info('{},data_type={},\t{}'.format(prefix, data_type, localz))
