import torch.nn as nn
import torch

class Model(nn.Module):

    def __init__(self, conf):
        super(Model, self).__init__()
        self.conf = conf
        convnum = 64
        unitnum = 200
        attnum = 10

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=conf.feanum,
                out_channels=convnum,
                kernel_size=(2, 1),
                stride=1,
                padding=0
            ),
            nn.Softmax(),
        )

        self.dense = nn.Sequential(
            nn.Linear((conf.mapsize_a - 1) * conf.mapsize_b * 64, unitnum),
            nn.ReLU6(),
            torch.nn.Dropout(conf.dropout),
            nn.Linear(unitnum, conf.labelnum),
        )

        self.attnum = attnum
        tensor = torch.ones(self.attnum, conf.mapsize_a-1, conf.mapsize_b)
        self.attention = torch.nn.Parameter(tensor)

    def forward(self, data, feas, startnode):

        Att = torch.sum(self.attention, 0)
        Att = Att / self.attnum
        x = data
        x = self.conv(x)
        x = Att * x + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)

        return out, self.attention