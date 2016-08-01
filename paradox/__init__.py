from loggers.logstash_formatter import LogstashFormatterV2
import logging


def config_logger(log_level=logging.INFO):
    """Configures the logger
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Disables the logs from the requests library
    requests_log = logging.getLogger("requests")
    requests_log.addHandler(logging.NullHandler())
    requests_log.propagate = False

    # Sets formatter to the logstash formatter
    handler = logging.StreamHandler()
    formatter = LogstashFormatterV2()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

config_logger()
