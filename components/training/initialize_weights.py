import torch.nn as nn


class InitWeight(object):
    def __init__(self, init_fn, pars=None, type='conv'):
        self.fn = init_fn
        self.pars = pars
        self.type = type

    def __call__(self, p):
        if self.pars is not None:
            if len(self.pars) == 2:
                self.fn(p, self.pars[0], self.pars[1], type=self.type)
            elif len(self.pars) == 1:
                self.fn(p, self.pars[0], type=self.type)
        else:
            self.fn(p, type=self.type)
        return


def init_weight_normal(p, mu=0.0, std=0.5, type='conv'):
    if type == 'conv':
        if isinstance(p, nn.Conv1d) or isinstance(p, nn.Conv2d) or isinstance(p, nn.Conv3d):
            nn.init.normal_(p.weight.data, mu, std)
            if p.bias is not None:
                nn.init.constant_(p.bias.data, 0.0)
    elif type == 'norm':
        if isinstance(p, nn.BatchNorm1d) or isinstance(p, nn.BatchNorm2d) or isinstance(p, nn.BatchNorm3d):
            nn.init.normal_(p.weight.data, 1.0, std)
            if p.bias is not None:
                nn.init.constant_(p.bias.data, 0.0)
    elif type == 'linear':
        if isinstance(p, nn.Linear):
            nn.init.normal_(p.weight.data, mu, std)
            if p.bias is not None:
                nn.init.constant_(p.bias.data, 0.0)
    return
