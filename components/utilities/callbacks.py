from collagen.core import Callback
from collagen.callbacks import Logger


class ScalarMeterLogger(Logger):
    def __init__(self, writer, log_dir: str = None, comment: str = '', name='scalar_logger'):
        super().__init__(name=name)
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = writer

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, callbacks, epoch, strategy, stage, **kwargs):
        for cb in callbacks:
            if cb.ctype == "meter" and cb.current() is not None:
                if self.__comment:
                    tag = self.__comment + "/" + cb.desc
                else:
                    tag = cb.desc
                self.__summary_writer.add_scalar(tag=tag, scalar_value=cb.current(), global_step=epoch)
