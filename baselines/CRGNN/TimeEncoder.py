import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, dim, length):
        super(TimeEncoder, self).__init__()

        self.time_day_mlp = nn.Conv2d(dim, dim, kernel_size=(1, length))
        self.time_week_mlp = nn.Conv2d(dim, dim, kernel_size=(1, length))
        self.time_day_emb = nn.Parameter(torch.randn(48, dim))
        self.time_week_emb = nn.Parameter(torch.randn(7, dim))

    def forward(self, x, time):
        time_in_day_emb = self.time_day_emb[(time[..., -2] * 48).long()].permute(0, 3, 2, 1)
        day_in_week_emb = self.time_week_emb[(time[..., -1]).long()].permute(0, 3, 2, 1)
        time_in_day_emb = self.time_day_mlp(time_in_day_emb)
        day_in_week_emb = self.time_week_mlp(day_in_week_emb)
        x = torch.cat([x, time_in_day_emb, day_in_week_emb], 1)
        return x
