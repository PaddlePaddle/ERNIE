# *_*coding:utf-8 *_*
"""
visual manager
可视化管理
"""
import logging
import os
import traceback
from visualdl import LogWriter


class VisualManager(object):
    """VisualManager
    """
    def __init__(self, logdir=None):
        """init
        """
        self.write_dict = {}
        if logdir:
            self.logdir = logdir
        else:
            self.logdir = "./visual_log"

    def show_metric(self, metrics_output, steps, tag):
        """评估指标展示
        :param metrics_output: 需要展示的指标，按dict方式存储
        :param steps:
        :param tag:
        :return:
        """
        logging.debug("{phase} log: steps {steps}, metrics: {metrics}".format(phase=tag,
                                                                             steps=steps,
                                                                             metrics=metrics_output))
        try:
            if metrics_output and len(metrics_output) != 0:
                for key, value in metrics_output.items():
                    # self.writer.add_scalar(tag=tag, step=steps, value=value)
                    writer = self.write_dict.get(key, None)
                    if writer and isinstance(writer, LogWriter):
                        writer.add_scalar(tag=tag, step=steps, value=value)
                    else:
                        logger_name = os.path.join(self.logdir, key)
                        with LogWriter(logdir=logger_name) as writer:
                            writer.add_scalar(tag=tag, step=steps, value=value)
                        self.write_dict[key] = writer

            else:
                logging.error("error type of metrics_output")

        except Exception:
            logging.error('traceback.format_exc():%s' % traceback.format_exc())