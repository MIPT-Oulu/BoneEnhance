from collagen.core import Callback
from collagen.callbacks import Logger, Meter
from collagen.core.utils import to_cpu, wrap_tuple
from datetime import datetime
from os import mkdir, remove
from os.path import exists, join, isfile
from typing import Tuple
import torch

import numpy as np
import torch.nn as nn
import dill as pickle


class ScalarMeterLogger(Logger):
    def __init__(self, writer, log_dir: str = None, comment: str = '', name='scalar_logger'):
        super().__init__(name=name)
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = writer

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, callbacks, epoch, stage, **kwargs):
        for cb in callbacks:
            if cb.ctype == "meter" and cb.current() is not None:
                if self.__comment:
                    tag = self.__comment + "/" + cb.desc
                else:
                    tag = cb.desc
                self.__summary_writer.add_scalar(tag=tag, scalar_value=cb.current(), global_step=epoch)


class RunningAverageMeter(Meter):
    def __init__(self, name: str = "loss", prefix="", desc_name=None):
        super().__init__(name=name, prefix=prefix, desc_name=desc_name)
        self.__value = 0
        self.__count = 0
        self.__avg_loss = None

    def on_epoch_begin(self, epoch, **kwargs):
        self.__value = 0
        self.__count = 0

    def on_minibatch_end(self, loss, loss_list=None, **kwargs):
        if loss_list is not None:
            loss_value = to_cpu(loss_list[self.name])
        else:
            loss_value = to_cpu(loss)
        if loss_value is not None:
            self.__value += loss_value
            self.__count += 1

    def on_epoch_end(self, *args, **kwargs):
        self.__avg_loss = self.current()

    def current(self):
        if self.__count == 0:
            return None
        return self.__value / self.__count


class ModelSaver(Callback):
    def __init__(self, metric_names: Tuple[str] or str, conditions: Tuple[str] or str, model: nn.Module,
                 save_dir: str, prefix: str = "", keep_best_only: bool = True, mode="and", stage='eval'):
        super().__init__(ctype="saver")
        self.__metric_names = wrap_tuple(metric_names)
        self.__conditions = wrap_tuple(conditions)
        self.__prefix = prefix if prefix else "model"
        self.__save_dir = save_dir
        self.__keep_best_only = keep_best_only
        self.__prev_model_path = ""
        self.__mode = mode
        self.__stage = wrap_tuple(stage)

        if self.__mode == "avg" and len(self.__conditions) > 1:
            if len(set(self.__conditions)) > 1:
                raise ValueError(
                    "All modes must be the same in {} mode, but got".format(self.__mode, self.__conditions))

        if not exists(self.__save_dir):
            print("Not found directory {} to save models. Create the directory.".format(self.__save_dir))
            mkdir(self.__save_dir)

        if len(self.__metric_names) != len(self.__conditions):
            raise ValueError("Metric names ({}) and conditions ({}) must be the same, "
                             "but got {} != {}".format(len(self.__metric_names),
                                                       len(self.__conditions),
                                                       len(self.__metric_names),
                                                       len(self.__conditions)))

        self.__best_metrics = dict()
        for i, metric_name in enumerate(self.__metric_names):
            cond = self.__conditions[i].lower()
            if cond in ["min", "max"]:
                self.__best_metrics[metric_name] = dict()
                self.__best_metrics[metric_name]["value"] = float('Inf') if cond == "min" else float('-Inf')
                self.__best_metrics[metric_name]["cond"] = cond
            else:
                raise ValueError('Values of conditions must be either min or max, but got {}'.format(cond))

        self.__model = model

    def __check_cond(self, value, metric_name):
        is_improved = False
        if self.__best_metrics[metric_name]["cond"] == "min" and self.__best_metrics[metric_name]["value"] > value:
            is_improved = True
        elif self.__best_metrics[metric_name]["cond"] == "max" and self.__best_metrics[metric_name]["value"] < value:
            is_improved = True

        return is_improved

    def __check_combined_cond(self, metrics):
        is_improved = False
        comb_metric = np.mean(np.array([metrics[metric_name] for metric_name in metrics]))
        best_metric = np.mean(
            np.array([self.__best_metrics[metric_name]["value"] for metric_name in self.__best_metrics]))
        cond = self.__best_metrics[next(iter(self.__best_metrics))]["cond"]
        if cond == "min" and comb_metric < best_metric:
            is_improved = True
        elif cond == "max" and comb_metric > best_metric:
            is_improved = True
        elif cond != "min" and cond != "max":
            raise ValueError("Condition must be `min` or `max`, but got {}".format(cond))
        return is_improved

    def on_epoch_end(self, epoch, stage, strategy, **kwargs):
        if stage in self.__stage:
            improved_metrics = dict()
            for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
                cb_name = str(cb)
                if cb.ctype == "meter" and cb_name in self.__best_metrics:
                    cb_value = cb.current()
                    if cb_value is None:
                        continue

                    if self.__mode == "and":
                        if self.__check_cond(value=cb_value, metric_name=cb_name):
                            improved_metrics[cb_name] = cb_value
                    elif self.__mode == "avg":
                        improved_metrics[cb_name] = cb_value

            if self.__mode == "avg":
                if not self.__check_combined_cond(improved_metrics):
                    improved_metrics = dict()

            if len(improved_metrics) == len(self.__best_metrics):
                list_metrics = []
                for metric_name in self.__best_metrics:
                    self.__best_metrics[metric_name]["value"] = improved_metrics[metric_name]
                    list_metrics += [metric_name.replace('/', '.'), "{0:.3f}".format(improved_metrics[metric_name])]
                metrics_desc = "_".join(list_metrics)
                date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = "_".join([self.__prefix, "{0:04d}".format(epoch), date_time, metrics_desc]) + ".pth"
                model_fullname = join(self.__save_dir, model_name)
                torch.save(self.__model.state_dict(), model_fullname)
                with open(model_fullname[:-4] + ".log", 'wb') as f:
                    pickle.dump(self.__best_metrics, f)

                if self.__keep_best_only and isfile(self.__prev_model_path):
                    remove(self.__prev_model_path)
                    remove(self.__prev_model_path[:-4] + ".log")
                self.__prev_model_path = model_fullname

    def get_metric_by_name(self, name):
        if name in self.__best_metrics:
            return self.__best_metrics[name]["value"]
        else:
            return None

    @property
    def metric_names(self):
        return self.__metric_names
