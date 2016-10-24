'''
This library is provided to allow standard python
logging to output log data as JSON formatted strings
ready to be shipped out to logstash.
'''
import logging
import socket
import datetime
import traceback as tb
import json


def _default_json_default(obj):
    """
    Coerce everything to strings.
    All objects representing time get output as ISO8601.
    """
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    else:
        return str(obj)


class LogstashFormatter(logging.Formatter):
    """
    A custom formatter to prepare logs to be
    shipped out to logstash.
    """

    def __init__(self,
                 fmt=None,
                 datefmt=None,
                 json_cls=None,
                 json_default=_default_json_default):
        """
        :param fmt: Config as a JSON string, allowed fields;
               extra: provide extra fields always present in logs
               source_host: override source host name
        :param datefmt: Date format to use (required by logging.Formatter
            interface but not used)
        :param json_cls: JSON encoder to forward to json.dumps
        :param json_default: Default JSON representation for unknown types,
                             by default coerce everything to a string
        """

        if fmt is not None:
            self._fmt = json.loads(fmt)
        else:
            self._fmt = {}
        self.json_default = json_default
        self.json_cls = json_cls
        if 'extra' not in self._fmt:
            self.defaults = {}
        else:
            self.defaults = self._fmt['extra']
        if 'source_host' in self._fmt:
            self.source_host = self._fmt['source_host']
        else:
            try:
                self.source_host = socket.gethostname()
            except:
                self.source_host = ""


class LogstashFormatterV2(LogstashFormatter):
    """
    A custom formatter to prepare logs to be
    shipped out to logstash V1 format.
    """

    def _make_timestamp(self, now):
        sft = now.strftime("%Y-%m-%dT%H:%M:%S")
        millis = ".%03dZ" % (now.microsecond / 1000)
        return sft + millis

    def _drop_some(self, fields):
        for field in ['args', 'created', 'filename', 'funcName', 'levelno',
                      'lineno', 'module', 'msecs', 'pathname', 'process',
                      'processName', 'relativeCreated', 'source_host',
                      'stack_info', 'thread', 'threadName']:
            fields.pop(field, None)

    def _filter_severity(self, fields):
        severity = fields.pop('levelname').lower()
        if 'warning' == severity:
            severity = 'warn'
        elif 'critical' == severity:
            severity = 'fatal'
        fields['severity'] = severity

    def _filter_message(self, fields):
        fields['message'] = fields.pop('msg', None)

        if type(fields['message']) is dict:
            params = fields.pop('message')
            fields['message'] = params.pop('message', None)
            fields['params'] = params

    def _filter_exception(self, fields):
        if 'exc_info' in fields:
            if fields['exc_info']:
                formatted = tb.format_exception(*fields['exc_info'])
                fields['exception'] = formatted
            fields.pop('exc_info')

        if 'exc_text' in fields and not fields['exc_text']:
            fields.pop('exc_text')

    def format(self, record):
        """
        Format a log record to JSON, if the message is a dict
        assume an empty message and use the dict as additional
        fields.
        """

        fields = record.__dict__.copy()
        self._drop_some(fields)
        self._filter_severity(fields)
        self._filter_message(fields)
        self._filter_exception(fields)
        fields['@timestamp'] = self._make_timestamp(datetime.datetime.utcnow())
        fields['@version'] = 1

        logr = self.defaults.copy()
        logr.update(fields)

        return json.dumps(logr, default=self.json_default, cls=self.json_cls)
