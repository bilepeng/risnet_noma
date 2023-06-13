import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

torch.manual_seed(134219)
np.random.seed(134219)


class RISNet(nn.Module):
    def __init__(self):
        super(RISNet, self).__init__()
        self.feature_dim = 8
        self.output_dim = 1
        self.local_info_dim = 16
        self.global_info_dim = 16

        self.conv1 = nn.Conv1d(self.feature_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv2 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv3 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv4 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv5 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv6 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv7 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv8 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.output_dim, 1)

    def forward(self, channel):
        def postprocess_layer(channel_, conv_output, n_entries):
            local_info = conv_output[:, :self.local_info_dim, :]
            global_info = torch.mean(conv_output[:, -self.global_info_dim:, :], dim=2, keepdim=True)
            global_info = global_info.repeat([1, 1, n_entries])
            layer_output = torch.cat((channel_, local_info, global_info), 1)
            return layer_output

        _, _, n_antennas = channel.shape

        r = F.relu(self.conv1(channel))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv2(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv3(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv4(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv5(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv6(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv7(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = self.conv8(r)

        return r
