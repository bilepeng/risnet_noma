import numpy as np
import scipy
import torch
from scipy.linalg import eigh
import torch.nn as nn
from torch import softmax
from torch import linalg as LA
import torch.nn.functional as F
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from joblib import cpu_count
solver = True
try:
    from scipy.optimize import fsolve
except:
    solver = False

torch.manual_seed(134219)
np.random.seed(134219)


def array2cp_fcn(array, factor=1):
    # Input: (batch, user x (amplitude, phase), width, height)
    # Output: (batch, user, antenna)
    cp1_amp = torch.flatten(array[:, 0, :, :], start_dim=1, end_dim=2) / factor
    cp2_amp = torch.flatten(array[:, 2, :, :], start_dim=1, end_dim=2) / factor
    cp1_pha = torch.flatten(array[:, 1, :, :], start_dim=1, end_dim=2) * np.pi
    cp2_pha = torch.flatten(array[:, 3, :, :], start_dim=1, end_dim=2) * np.pi
    cp1 = cp1_amp * torch.exp(1j * cp1_pha)  # (batch, antenna)
    cp2 = cp2_amp * torch.exp(1j * cp2_pha)  # (batch, antenna)
    _cp = torch.stack([cp1, cp2], dim=1).cfloat()  # (batch, user, antenna)
    return _cp


def cp2array_fcn(cp, factor=1, array_shape=(16, 16), device='cpu'):
    # Input: (batch, user, width, height) or (batch, user, antenna)
    # Output: (batch, user x (amplitude, phase), width, height)
    if len(cp.shape) == 3:
        shape = cp.shape
        cp = torch.reshape(cp, (shape[0], shape[1], array_shape[0], array_shape[1]))
    cp1_amp = cp[:, 0, :, :].abs() * factor
    cp2_amp = cp[:, 1, :, :].abs() * factor
    cp1_pha = cp[:, 0, :, :].angle()
    cp2_pha = cp[:, 1, :, :].angle()
    array = torch.stack([cp1_amp, cp1_pha, cp2_amp, cp2_pha], dim=1)
    return array.to(device)


def cp2array_lagia(cp, factor=1, device="cpu"):
    # User \in {1,2}
    # Input: (batch, user, antenna)
    # Output: (batch, feature, antenna))
    cp1_amp = cp[:, 0, :].abs() * factor
    cp1_pha = cp[:, 0, :].angle()
    # p = torch.FloatTensor([[phase + 2 * torch.pi if phase < 0 else phase for phase in pairs] for pairs in cp1_pha])
    if cp.size()[1] == 2:
        cp2_amp = cp[:, 1, :].abs() * factor
        cp2_pha = cp[:, 1, :].angle()
        array = torch.stack([cp1_amp, cp1_pha, cp2_amp, cp2_pha], dim=1)
    else:
        print("WARNNG: Unwanted shape")
        array = torch.stack([cp1_amp, cp1_pha], dim=1)
    return array.to(device)


def prepare_channel_direct_features(channel_direct, channel_tx_ris_pinv, params, device='cpu'):
    # new data has dType torch.complex128, old data has torch.complex64
    if channel_tx_ris_pinv.dtype == channel_direct.dtype:
        equivalent_los_channel = channel_direct @ channel_tx_ris_pinv
    else:
        print("WARNING: Unwanted dtype")
        equivalent_los_channel = channel_direct.type(torch.complex128) @ channel_tx_ris_pinv.type(torch.complex128)

    if params["fcn"]:
        equivalent_los_channel = torch.reshape(equivalent_los_channel, (-1, 2, params['ris_shape'][0],
                                                                        params['ris_shape'][1]))
        return cp2array_fcn(equivalent_los_channel, params['amplification_direct'], (16, 16), device)
    else:
        return cp2array_lagia(equivalent_los_channel, params['amplification_direct'], device)


def sum_rate(complete_channel, precoding, snr):
    channel_precoding = complete_channel @ precoding
    channel_precoding = torch.square(channel_precoding.abs())
    rate1 = torch.log2(1 + channel_precoding[:, 0, 0] / (channel_precoding[:, 0, 1] + 1 / snr))
    rate2 = torch.log2(1 + channel_precoding[:, 1, 1] / (channel_precoding[:, 1, 0] + 1 / snr))
    return rate1 + rate2
    # return -torch.mean(torch.min(rate1, rate2))


def weighted_sum_rate(complete_channel, precoding, params):
    channel_precoding = complete_channel @ precoding
    channel_precoding = torch.square(channel_precoding.abs())
    rate1 = torch.log2(1 + channel_precoding[:, 0, 0] / (channel_precoding[:, 0, 1] + 1 / params['tsnr']))
    rate2 = torch.log2(1 + channel_precoding[:, 1, 1] / (channel_precoding[:, 1, 0] + 1 / params['tsnr']))
    return params['alphas'][0] * rate1 + params['alphas'][1] * rate2


def noma_weighted_sum_rate(complete_channel, precoding, params, device="cpu"):
    h1, h2 = torch.split(complete_channel, [1, 1], dim=1)
    h1 = torch.transpose(h1, 1, 2).conj().type(torch.complex128).to(device)
    h2 = torch.transpose(h2, 1, 2).conj().type(torch.complex128).to(device)
    h1_H = torch.adjoint(h1)
    h2_H = torch.adjoint(h2)

    w1, w2 = torch.split(precoding, [1, 1], dim=2)
    w1 = w1.type(torch.complex128).to(device)
    w2 = w2.type(torch.complex128).to(device)
    # w1 = torch.transpose(w1, 1, 2)
    # w2 = torch.transpose(w2, 1, 2)
    w1_H = torch.adjoint(w1)
    w2_H = torch.adjoint(w2)

    sigma = torch.sqrt(torch.tensor([1 / params["tsnr"]])).type(torch.complex128).to(device)

    SINR_1 = torch.real(torch.bmm(h1_H, w1) * torch.bmm(w1_H, h1) / (sigma ** 2))

    SINR_21 = torch.real(torch.bmm(h1_H, w2) * torch.bmm(w2_H, h1) / (sigma ** 2 + (torch.bmm(h1_H, w1) * torch.bmm(w1_H, h1))))
    SINR_22 = torch.real((torch.bmm(h2_H, w2) * torch.bmm(w2_H, h2)) / (sigma ** 2 + (torch.bmm(h2_H, w1) * torch.bmm(w1_H, h2))))

    R1 = torch.log2(1 + SINR_1)
    R2 = torch.minimum(torch.log2(1 + SINR_21), torch.log2(1 + SINR_22))
    # R2 = torch.minimum(torch.tensor([torch.log(1 + SINR_21), torch.log(1 + SINR_22)]))

    return torch.squeeze(R1 + R2)


def precoding_power(precoding):
    precoding_norm = torch.norm(precoding, dim=1)
    power = torch.square(precoding_norm)
    sum_power = torch.sum(power, dim=1)
    # print(f"Average precoding power: {avg_power.item()}")
    return sum_power


def average_precoding_power(precoding):
    precoding_norm = torch.norm(precoding, dim=1)
    power = torch.square(precoding_norm)
    avg_power = torch.mean(torch.sum(power, dim=1))
    # print(f"Average precoding power: {avg_power.item()}")
    return avg_power


def test_model(complete_channel, precoding, params):
    if type(complete_channel) is np.ndarray:
        complete_channel = torch.from_numpy(complete_channel).cfloat()

    if type(precoding) is np.ndarray:
        precoding = torch.from_numpy(precoding).cfloat()

    complete_channel = torch.square((complete_channel @ precoding).abs())
    rate1 = torch.log2(1 + complete_channel[:, 0, 0] / (complete_channel[:, 0, 1] + 1 / params['tsnr']))
    rate2 = torch.log2(1 + complete_channel[:, 1, 1] / (complete_channel[:, 1, 0] + 1 / params['tsnr']))
    return complete_channel.cpu().detach().numpy(), rate1.cpu().detach().numpy(), rate2.cpu().detach().numpy()


def array2phase_shifts(phase_shifts):
    # Input: (batch, 1, width, height)
    # Output: (batch, antenna, antenna)
    p = torch.flatten(phase_shifts[:, 0, :, :], start_dim=1, end_dim=2)
    p = torch.diag_embed(torch.exp(1j * p))
    return p


class RTChannels(Dataset):
    def __init__(self, params, device='cpu', test=False):
        self.params = params
        self.device = device
        self.channels_ris_rx = torch.load(params['channel_ris_rx_path'], map_location=torch.device(device)).cfloat()

        if test:
            self.channels_ris_rx = torch.flip(self.channels_ris_rx, [1])
        # Reshape self.channels_direct. Change number of samples in "params" if new data were used
        self.channels_ris_rx = torch.reshape(self.channels_ris_rx, params['channel_ris_rx_original_shape'])[:, :,
                               : params['ris_shape'][0],
                               : params['ris_shape'][1]]
        self.channels_ris_rx = torch.reshape(self.channels_ris_rx, (
            params['channel_ris_rx_original_shape'][0],
            params['channel_ris_rx_original_shape'][1], params['ris_shape'][0] * params['ris_shape'][1]))
        if test:
            error = (self.channels_ris_rx.abs().mean() * params['channel_estimate_error'] *
                     torch.exp(np.pi * 2j * torch.ones(self.channels_ris_rx.shape))).detach()
            self.channels_ris_rx_array = cp2array_lagia(self.channels_ris_rx + error,
                                                        params['amplification_ris'],
                                                        device=device)
        else:
            if params["fcn"]:
                self.channels_ris_rx_array = cp2array_fcn(self.channels_ris_rx,
                                                          params['amplification_ris'],
                                                          array_shape=(16, 16),
                                                          device=device)
            else:
                self.channels_ris_rx_array = cp2array_lagia(self.channels_ris_rx,
                                                            params['amplification_ris'],
                                                            device=device)

    def __getitem__(self, item):
        return [self.channels_ris_rx_array[item, :, :], self.channels_ris_rx[item, :]]

    def __len__(self):
        return self.channels_ris_rx_array.shape[0]


class RTChannelsDirect(RTChannels):
    def __init__(self, params, channel_tx_ris_pinv, device='cpu', test=False):
        super(RTChannelsDirect, self).__init__(params=params, device=device, test=test)
        self.channels_direct = torch.load(params['channel_direct_path'], map_location=torch.device(device)).cfloat()

        # self.channels_direct old Data: 10-6 to 10-8, new Data: 10^-4 to 10^-6
        # Reshape self.channels_direct if users are not paired
        if self.channels_direct.size()[1] == 1:
            b = self.channels_direct
            self.channels_direct = torch.reshape(b, (int(b.size()[0] / 2), int(b.size()[1] * 2), int(b.size()[2] * 1)))

        if test:
            self.channels_direct = torch.flip(self.channels_direct, [1])
            error = (self.channels_direct.abs().mean() * params['channel_estimate_error'] *
                     torch.exp(np.pi * 2j * torch.ones(self.channels_direct.shape))).detach()
            channels_direct_array = prepare_channel_direct_features(self.channels_direct + error, channel_tx_ris_pinv,
                                                                    self.params, self.device)
        else:
            channels_direct_array = prepare_channel_direct_features(self.channels_direct, channel_tx_ris_pinv,
                                                                    self.params, self.device)
        self.channels_ris_rx_array = torch.cat([self.channels_ris_rx_array, channels_direct_array], 1)

    def __getitem__(self, item):
        data = super(RTChannelsDirect, self).__getitem__(item)
        channel_direct = self.channels_direct[item, :]
        data.append(channel_direct)
        return data


class RTChannelsLocation(Dataset):
    def __init__(self, params, channel_tx_ris_pinv, device='cpu'):
        self.params = params
        self.device = device
        self.channels_ris_rx = torch.load(params['channel_ris_rx_path'], map_location=torch.device(device)).cfloat()
        self.channels_direct = torch.load(params['channel_direct_path'], map_location=torch.device(device)).cfloat()
        self.locations = torch.load(params['locations_path'], map_location=torch.device(device)).float()
        self.locations = torch.unsqueeze(self.locations, dim=1)

        # Adding Noise
        for idx, channel in enumerate(self.channels_direct):
            self.channels_direct[idx] += (torch.normal(0, 1e-8, size=(1, 9)) + 1j * torch.normal(0, 1e-8, size=(1, 9))).to(device)
            self.channels_ris_rx[idx] += (torch.normal(0, 1e-8, size=(1, 1024)) + 1j * torch.normal(0, 1e-8, size=(1, 1024))).to(device)

        # Permutate channel and merch 2 as a pair
        if self.channels_ris_rx.size()[1] == 1:
            perm = torch.randperm(self.channels_ris_rx.size()[0])
            a = self.channels_ris_rx[perm]
            b = self.channels_direct[perm]
            c = self.locations[perm]
            self.channels_ris_rx = torch.reshape(a, (int(a.size()[0] / 2), int(a.size()[1] * 2), int(a.size()[2] * 1)))
            self.channels_direct = torch.reshape(b, (int(b.size()[0] / 2), int(b.size()[1] * 2), int(b.size()[2] * 1)))
            self.locations = torch.reshape(c, (int(c.size()[0] / 2), int(c.size()[1] * 2), int(c.size()[2] * 1)))

        # Filter
        mask = torch.ones(self.channels_ris_rx.size()[0], dtype=torch.bool)
        for idx, channel in enumerate(self.channels_ris_rx):
            if torch.norm(channel[0]) < torch.norm(channel[1]):
                self.channels_ris_rx[idx][[0, 1]] = self.channels_ris_rx[idx][[1, 0]]
                self.channels_direct[idx][[0, 1]] = self.channels_direct[idx][[1, 0]]
                self.locations[idx][[0, 1]] = self.locations[idx][[1, 0]]
            if torch.norm(channel[0]) < torch.norm(channel[1]) * 2:
                mask[idx] = False
            # if not (self.locations[idx][0][1] > 500 and self.locations[idx][1][1] < 450):
            #     mask[idx] = False
            # self.channels_direct[idx][1] = self.channels_direct[idx][1] * 0


        self.channels_ris_rx = self.channels_ris_rx[mask]
        self.channels_direct = self.channels_direct[mask]
        self.locations = self.locations[mask]

        # for idx, channel in enumerate(self.channels_ris_rx):
        #     self.channels_ris_rx[idx][0] = self.channels_ris_rx[idx][0] * 2

        # mask = torch.ones(self.channels_ris_rx.size(dim=0), dtype=torch.bool)
        # for i in range(10000):
        #     hd_1 = torch.unsqueeze(self.channels_ris_rx[i][0], 0).H
        #     hd_2 = torch.unsqueeze(self.channels_ris_rx[i][1], 0).H
        #     if cos_squared(hd_1, hd_2) < 0.3:
        #         mask[i] = False
        #
        # self.channels_ris_rx = self.channels_ris_rx[mask]
        # self.channels_direct = self.channels_direct[mask]
        # self.locations = self.locations[mask]

        self.channels_ris_rx = torch.reshape(self.channels_ris_rx, (self.channels_ris_rx.size(dim=0), 2,
                                                                    params['ris_shape'][0], params['ris_shape'][1]))
        self.channels_ris_rx = torch.reshape(self.channels_ris_rx, (self.channels_ris_rx.size(dim=0), 2,
                                                                    params['ris_shape'][0] * params['ris_shape'][1]))

        if params["fcn"]:
            channels_ris_rx_array = cp2array_fcn(self.channels_ris_rx, params['amplification_ris'],
                                                      array_shape=(16, 16), device=device)
        else:
            channels_ris_rx_array = cp2array_lagia(self.channels_ris_rx, params['amplification_ris'],
                                                        device=device)

        channels_direct_array = prepare_channel_direct_features(self.channels_direct, channel_tx_ris_pinv,
                                                                self.params, self.device)

        self.channels_array = torch.cat([channels_ris_rx_array, channels_direct_array], 1)

        # Normalization of channel features
        # self.mean = torch.mean(self.channels_array, dim=0)
        # self.std = torch.std(self.channels_array, dim=0)
        # self.channels_array = (self.channels_array - self.mean) / self.std

    def __getitem__(self, item):
        data = [self.channels_array[item, :, :], self.channels_ris_rx[item, :], self.channels_direct[item, :], self.locations[item, :]]
        return item, data

    def __len__(self):
        return self.channels_array.shape[0]


def mmse_precoding(channel, params, device='cpu'):
    if type(channel) is np.ndarray:
        channel = torch.from_numpy(channel).to(device)
    eye = torch.eye(channel.shape[1]).repeat((channel.shape[0], 1, 1)).to(device)
    p = channel.transpose(1, 2).conj() @ torch.linalg.inv(channel @ channel.transpose(1, 2).conj() +
                                                          1 / params['tsnr'] * eye)
    trace = torch.sum(torch.diagonal((p @ p.transpose(1, 2).conj()), dim1=1, dim2=2).real, dim=1, keepdim=True)
    p = p / torch.unsqueeze(torch.sqrt(trace), dim=2)
    return p


def compute_complete_channel_discrete(channel_tx_ris, fcn_output, channel_ris_rx, channel_direct, params):
    phase = torch.tensor([0, np.pi / 2, np.pi, np.pi * 1.5])
    phase_3d = phase[None, None, :].to("cuda")
    if params["fcn"]:
        fcn_output = torch.unsqueeze(torch.flatten(fcn_output[:, 0, :, :], start_dim=1, end_dim=2), dim=1)
    phi = torch.exp(1j * (phase_3d @ fcn_output))
    if channel_direct is None:
        complete_channel = (channel_ris_rx * phi) @ channel_tx_ris
    else:
        complete_channel = (channel_ris_rx * phi) @ channel_tx_ris + channel_direct
    return complete_channel


def compute_complete_channel_continuous(channel_tx_ris, fcn_output, channel_ris_rx, channel_direct, params):
    if params["fcn"]:
        phi = torch.unsqueeze(torch.exp(1j * torch.flatten(fcn_output[:, 0, :, :], start_dim=1, end_dim=2)), dim=1)
    else:
        phi = torch.exp(1j * fcn_output * 2 * torch.pi)
    if channel_direct is None:
        complete_channel = (channel_ris_rx * phi) @ channel_tx_ris
    else:
        complete_channel = (channel_ris_rx * phi) @ channel_tx_ris + channel_direct
    return complete_channel


def prepare_channel_tx_ris(params, device):
    channel_tx_ris = torch.load(params['channel_tx_ris_path'], map_location=torch.device(device))
    if channel_tx_ris.size()[0] != params["ris_shape"][0] * params["ris_shape"][1]:
        channel_tx_ris = channel_tx_ris.transpose(dim0=0, dim1=1)
        channel_tx_ris = channel_tx_ris[:params["ris_shape"][0] * params["ris_shape"][1], :]
    channel_tx_ris_pinv = torch.linalg.pinv(channel_tx_ris)
    return channel_tx_ris, channel_tx_ris_pinv


def delay2phases(delay, frequencies):
    phases = delay * frequencies * 2 * np.pi
    return phases


def precoding_quasi_degradated(h1, h2, params, device='cpu'):
    sigma = torch.sqrt(torch.tensor([1 / params["tsnr"]])).to(device)
    r1 = params['min_SNR_ratio'][0]
    r2 = params['min_SNR_ratio'][1]

    e_1 = (h1 / torch.norm(h1))
    e_2 = (h2 / torch.norm(h2))
    alpha_1 = (r1 * sigma ** 2) / torch.square(torch.norm(h1)) * (1 / torch.square(1 + r2 * (1 - cos_squared(h1, h2))))
    alpha_2 = (r2 * sigma ** 2) / torch.square(torch.norm(h2)) + ((r1 * sigma ** 2) / torch.square(torch.norm(h1))) * (r1 * cos_squared(h1, h2)) \
              / torch.square(1 + r1 * (1 - cos_squared(h1, h2)))

    alpha_1 = torch.sqrt(alpha_1)
    alpha_2 = torch.sqrt(alpha_2)

    # test1 = e_2.H
    # test2 = e_2.H @ e_1
    # test3 = (e_2.H @ e_1) * e_2

    # w_1_factor = (((1 + r2) * e_1) - r2 * e_2.H @ e_1 * e_2)
    w_1 = alpha_1 * (((1 + r2) * e_1) - r2 * e_2.H @ e_1 * e_2)
    w_2 = alpha_2 * e_2
    return torch.cat((w_1, w_2), 1)


def qd_precoding(complete_channel, params, device='cpu'):
    sigma = torch.sqrt(torch.tensor([1 / params["tsnr"]])).to(device)
    if type(complete_channel) is np.ndarray:
        complete_channel = torch.from_numpy(complete_channel).to(device)

    # Using Tensor operations
    # Function_test assumes two BS antennas:
    # e = complete_channel / torch.norm(complete_channel, dim=2).unsqueeze(2).repeat(1, 1, 2)

    channel = complete_channel.transpose(1, 2).conj()
    channel0, channel1 = torch.chunk(channel, 2, dim=2)

    channel_norm0 = torch.norm(channel0, dim=1)
    channel_norm1 = torch.norm(channel1, dim=1)

    e0 = (channel0 / torch.unsqueeze(channel_norm0, 1)).type(torch.complex64)
    e1 = (channel1 / torch.unsqueeze(channel_norm1, 1)).type(torch.complex64)

    cos2_numerator = (torch.matmul(channel0.transpose(1, 2).conj(), channel1) *
                      torch.matmul(channel1.transpose(1, 2).conj(), channel0)).real
    cos2_denumerator = torch.square(torch.norm(channel0, dim=1)) * torch.square(torch.norm(channel1, dim=1))

    cos2 = (torch.squeeze(cos2_numerator) / torch.squeeze(cos2_denumerator))
    #print(f"Cos^2 : {cos2}")

    alpha0 = (params['min_SNR_ratio'][0] * sigma ** 2) / torch.square(torch.squeeze(torch.norm(channel0, dim=1)))
    alpha0 = torch.sqrt(alpha0 * (1 / torch.square(1 + params['min_SNR_ratio'][1] * (1 - cos2))))

    alpha1 = (params['min_SNR_ratio'][0] * sigma ** 2) / torch.square(torch.squeeze(torch.norm(channel0, dim=1)))
    alpha1 = alpha1 * ((params['min_SNR_ratio'][1] * cos2) / torch.square(1 + params['min_SNR_ratio'][1] * (1 - cos2)))
    alpha1 = torch.sqrt(alpha1 + (params['min_SNR_ratio'][1] * sigma ** 2) /
                                  torch.square(torch.squeeze(torch.norm(channel1, dim=1))))

    alpha0_batch = torch.unsqueeze(torch.unsqueeze(alpha0, -1), -1).type(torch.complex64)
    # Function_test has batch size 0, therefore 1 additional dimension is necessary:
    # alpha0_batch = torch.unsqueeze(alpha0_batch, -1)

    #test1 = e1.conj()
    #test2 = (e1.conj() @ e0.transpose(2, 1))
    #test3 = (e1.conj() @ e0.transpose(2, 1)) * e1

    w0_factor = (((1 + params['min_SNR_ratio'][1]) * e0) - (params['min_SNR_ratio'][1] *
                                                           (e1.transpose(2, 1).conj() @ e0) * e1)).type(torch.complex64)
    if len(alpha0_batch.size()) < 3:
        alpha0_batch = torch.unsqueeze(alpha0_batch, 0)
    w0 = torch.bmm(w0_factor, alpha0_batch)

    alpha1_batch = torch.unsqueeze(torch.unsqueeze(alpha1, -1), -1).type(torch.complex64)
    # Function_test has batch size 0, therefore 1 additional dimension is necessary:
    # alpha1_batch = torch.unsqueeze(alpha1_batch, -1)
    if len(alpha1_batch.size()) < 3:
        alpha1_batch = torch.unsqueeze(alpha1_batch, 0)
    w1 = torch.bmm(e1, alpha1_batch)

    V = torch.cat((w0, w1), 2).to(device)

    # V_old = torch.empty(torch.movedim(complete_channel, 2, 1).size(), dtype=torch.cfloat).to(device)
    # for count, channel in enumerate(complete_channel):
    #     h1 = torch.unsqueeze(channel[0], 1)
    #     h2 = torch.unsqueeze(channel[1], 1)
    #     new_prcoding = precoding_quasi_degradated(params['min_SNR_ratio'][0], params['min_SNR_ratio'][1], h1, h2)
    #     #new_prcoding = torch.unsqueeze(new_prcoding, 0)
    #     V_old[count] = new_prcoding #torch.cat((V, new_prcoding), dim=0)
    # print(f"Equality: {torch.equal(V, V_old)}")
    return V


def precoding_zero_forcing(h1, h2,  params, device='cpu'):
    sigma = torch.sqrt(torch.tensor([1 / params["tsnr"]])).to(device)
    r1 = torch.tensor(params['min_SNR_ratio'][0])
    r2 = torch.tensor(params['min_SNR_ratio'][1])

    cos2 = cos_squared(h1, h2)
    sin2 = 1 - cos2

    a1 = torch.square(torch.norm(h2)) * h1 - (h2.H @ h1) * h2
    a2 = - (h1.H @ h2) * h1 + torch.square(torch.norm(h1)) * h2

    w_1 = torch.sqrt(r1) * a1 * sigma / (torch.square(torch.norm(h1)) * torch.square(torch.norm(h2)) * sin2)
    w_2 = torch.sqrt(r2) * a2 * sigma / (torch.square(torch.norm(h1)) * torch.square(torch.norm(h2)) * sin2)

    V = torch.cat((w_1, w_2), 1)
    return V


def zf_precoding(complete_channel, params, device='cpu'):
    sigma = torch.sqrt(torch.tensor([1 / params["tsnr"]])).to(device)
    if type(complete_channel) is np.ndarray:
        complete_channel = torch.from_numpy(complete_channel).to(device)

    r1 = torch.tensor(params['min_SNR_ratio'][0])
    r2 = torch.tensor(params['min_SNR_ratio'][1])

    channel = complete_channel.transpose(1, 2).conj()
    channel0, channel1 = torch.chunk(channel, 2, dim=2)

    channel_norm0 = torch.norm(channel0, dim=1)
    channel_norm1 = torch.norm(channel1, dim=1)

    cos2_numerator = (torch.matmul(channel0.transpose(1, 2).conj(), channel1) *
                      torch.matmul(channel1.transpose(1, 2).conj(), channel0)).real
    cos2_denumerator = torch.square(torch.norm(channel0, dim=1)) * torch.square(torch.norm(channel1, dim=1))

    cos2 = (torch.squeeze(cos2_numerator) / torch.squeeze(cos2_denumerator))

    mask_small = torch.greater(torch.tensor(0.00000001), cos2)
    cos2[mask_small] = torch.tensor(0.00000001)  # .type(torch.DoubleTensor)
    mask_big = torch.greater(cos2, torch.tensor(0.99999999))
    cos2[mask_big] = torch.tensor(0.99999999)  # .type(torch.DoubleTensor)
    sin2 = 1 - cos2

    a1 = torch.unsqueeze(torch.square(channel_norm1), dim=2) * channel0 - \
         torch.bmm(channel1.transpose(1, 2).conj(), channel0) * channel1
    a2 = - torch.bmm(channel0.transpose(1, 2).conj(), channel1) * channel0 + \
         torch.unsqueeze(torch.square(channel_norm0), dim=2) * channel1

    w0_num = torch.sqrt(r1) * a1 * 1e9 * sigma
    w1_num = torch.sqrt(r2) * a2 * 1e9 * sigma
    w_dnum = torch.unsqueeze(torch.unsqueeze(torch.squeeze(torch.square(channel_norm0)) * torch.squeeze(torch.square(channel_norm1)) * sin2, dim=1), dim=2) * 1e9
    w0 = w0_num / w_dnum
    w1 = w1_num / w_dnum
    V = torch.cat((w0, w1), 2).to(device)
    # test = complete_channel @ V
    return V

def is_quasi_degradated(complete_channel, params, numeric=False):
    r1 = params['min_SNR_ratio'][0]
    r2 = params['min_SNR_ratio'][1]

    batch_size = complete_channel.size()[0]
    numeric_val = torch.zeros(batch_size).type(torch.DoubleTensor)
    percentage = 0

    h1, h2 = torch.split(complete_channel, [1, 1], dim=1)
    h1 = torch.transpose(h1, 1, 2).conj()
    h2 = torch.transpose(h2, 1, 2).conj()
    h1_H = torch.adjoint(h1)
    h2_H = torch.adjoint(h2)

    numerator = torch.squeeze((torch.bmm(h1_H, h2) * torch.bmm(h2_H, h1)).real)
    denumerator = torch.squeeze(torch.square(LA.norm(h1, dim=1)) * torch.square(LA.norm(h2, dim=1)))

    cos2 = numerator / denumerator
    mask_small = torch.greater(torch.tensor(0.00001), cos2)
    cos2[mask_small] = torch.tensor(0.00001).type(torch.DoubleTensor)
    mask_big = torch.greater(cos2, torch.tensor(0.99999))
    cos2[mask_big] = torch.tensor(0.99999).type(torch.DoubleTensor)

    bound = torch.squeeze(torch.square(LA.norm(h1, dim=1))) / torch.squeeze(torch.square(LA.norm(h2, dim=1)))

    Q = ((1 + r1) / cos2) - (r1 * cos2) / torch.square(1 + r2 * (1 - cos2))

    m = nn.ReLU()  # nn.ReLU()  #ELU()
    numeric_val = m(Q - bound)  # m(Q - bound) : torch.special.expit(Q - bound) : Q - bound
    qd = torch.greater(bound, Q)

    if not numeric:
        if batch_size == 1:
            return qd
        else:
            return qd  # sum(qd) / len(qd)
    else:
        return numeric_val



def can_be_quasi_degradaded(Upsilon1: torch.tensor, Upsilon2: torch.tensor):
    # \lambda_{max} (Upsilon1 - Upsilon2) > 0
    e = torch.linalg.eigvals(Upsilon1 - 2 * Upsilon2)
    e2 = torch.linalg.eigvals(Upsilon2 - Upsilon1)
    max = torch.max(torch.real(e))
    min = torch.min(torch.real(e2))
    # if torch.max(torch.imag(e)) > torch.min(torch.real(e)) * 1000:
    #     print("WARNING: Large imaginary part")
    if max <= 0:  # 0.68 * 1e-4:
        return False
    else:
        return True


def cos_squared(input1: torch.Tensor, input2):
    if input1.dim() == 2:
        # cos^2 = h_1^H h_2 h_2^H h_1 / (||h_1||² ||h_2||²)
        if torch.is_complex(input1) or torch.is_complex(input2):
            numerator = ((input1.H @ input2) * (input2.H @ input1)).real
            if (input1.H @ input2 * input2.H @ input1).real < (input1.H @ input2 * input2.H @ input1).imag * 1000:
                print("Warning: Big Floating-point error")
        else:
            numerator = (input1.H @ input2 * input2.H @ input1)
        denumerator = torch.square(input1.norm()) * torch.square(input2.norm())
        return numerator / denumerator
    if input1.dim() == 3:
        h1, h2 = torch.split(input1, [1, 1], dim=1)
        h1 = torch.transpose(h1, 1, 2).conj()
        h2 = torch.transpose(h2, 1, 2).conj()
        h1_H = torch.adjoint(h1)
        h2_H = torch.adjoint(h2)

        numerator = torch.squeeze((torch.bmm(h1_H, h2) * torch.bmm(h2_H, h1)).real)
        denumerator = torch.squeeze(torch.square(LA.norm(h1, dim=1)) * torch.square(LA.norm(h2, dim=1)))

        cos2 = numerator / denumerator
        mask_small = torch.greater(torch.tensor(0.00001), cos2)
        cos2[mask_small] = torch.tensor(0.00001)  # .type(torch.DoubleTensor)
        mask_big = torch.greater(cos2, torch.tensor(0.99999))
        cos2[mask_big] = torch.tensor(0.99999)
        return cos2


def percentage_quasi_degradated(complete_channel, params):
    anz_quasi_degradated = 0
    for position in complete_channel:
        if is_quasi_degradated(params["min_SNR_ratio"][0], params["min_SNR_ratio"][1], torch.reshape(position[0], (9, 1)), torch.reshape(position[1], (9, 1))):
            anz_quasi_degradated += 1
    percentage_quasi_degradated = anz_quasi_degradated * 100 / params["batch_size"]
    return percentage_quasi_degradated


def gen_Phi(h_rk: torch.Tensor, G: torch.Tensor):
    # Phi = diag(h_{rk}) @ G
    h_rk = torch.squeeze(h_rk)
    phi = torch.diag(h_rk) @ G
    return phi


def gen_Upsilon(Phi: torch.Tensor, h_d: torch.Tensor):
    # upsilon_k = [[phi_k @ phi_k^H, phi_k @ h_{dk}], [h_{dk}^H @ phi_k^H, h_{dk}^H @ h_{dk}]]
    PhiH = Phi.transpose(0, 1).conj()
    h_dH = h_d.transpose(0, 1).conj()
    upper = torch.hstack(((Phi @ PhiH), (Phi @ h_d)))
    lower = torch.hstack(((h_dH @ PhiH), (h_dH @ h_d)))
    upsilon = torch.vstack((upper, lower))
    return upsilon


def gen_R(Phi1: torch.Tensor, Phi2: torch.Tensor, h_d1: torch.Tensor, h_d2: torch.Tensor):
    # R = [[phi_1 @ phi_2^H, phi_1 @ h_{d2}], [h_{d1}^H @ phi_2^H, h_{d1}^H @ h_{d2}]]
    Phi2H = Phi2.transpose(0, 1).conj()
    h_d1H = h_d1.transpose(0, 1).conj()
    upper = torch.hstack(((Phi1 @ Phi2H), (Phi1 @ h_d2)))
    lower = torch.hstack(((h_d1H @ Phi2H), (h_d1H @ h_d2)))
    R = torch.vstack((upper, lower))
    return R


def gen_Chi(Phi: torch.Tensor, h_d: torch.Tensor):
    # Chi = [phi_k^H, h_{dk}]
    PhiH = Phi.transpose(0, 1).conj()
    Chi = torch.hstack((PhiH, h_d))
    return Chi


def determine_kappa(entropies, previous_kappa, params):
    lr_kappa = params["lr_kappa"]
    if len(entropies) < 10:
        return previous_kappa
    elif entropies[-1] >= np.mean(entropies[-20:-1]):
        return previous_kappa + lr_kappa
    else:
        return F.relu(previous_kappa - lr_kappa / 2)