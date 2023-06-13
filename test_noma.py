import numpy as np
from util import RTChannelsLocation, test_model, prepare_channel_tx_ris, \
    compute_complete_channel_continuous, compute_wmmse_v, qd_precoding, noma_weighted_sum_rate, \
    precoding_power, is_quasi_degradated, zf_precoding, sum_rate
import torch
from core import RISNet
from torch.utils.data import DataLoader
from params import params
import matplotlib.pyplot as plt
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False

# REPRODUCIBILITY
torch.manual_seed(134219)
np.random.seed(134219)
g = torch.Generator()
g.manual_seed(134219)

if __name__ == '__main__':
    # params['los'] = True
    # if params['ris_shape'][0] > 16:
    #     model = FCNBig(params['los'])
    # else:
    #     model = FCN(params['los'])

    model = RISNet()

    device = 'cpu'
    model.load_state_dict(torch.load('results_NOMA/21-09-2022_12-09-44/ris_100000000000.0_(32, 32)_[0.5, 0.5]_50000'))
    model.eval()
    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)

    data_set = RTChannelsLocation(params, channel_tx_ris_pinv, device)
    len_data_set = data_set.__len__()
    # all_loader = DataLoader(dataset=data_set, batch_size=len_data_set)
    # _, all_arrays = next(iter(all_loader))
    # all_arrays = all_arrays[0]
    # max_val = torch.max(all_arrays)
    # min_val = torch.min(all_arrays)
    # norm_arrays1 = (all_arrays - min_val) / (max_val - min_val)
    #
    # mean = torch.mean(all_arrays, dim=0)
    # std = torch.std(all_arrays, dim=0)
    # norm_arrays2 = (all_arrays - mean) / std

    train_data, test_data = torch.utils.data.random_split(data_set, [int(np.floor(len_data_set * 0.9)),
                                                                     int(np.ceil(len_data_set * 0.1))])
    train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'], shuffle=True, generator=g)
    test_loader = DataLoader(dataset=test_data, batch_size=1024, shuffle=True, generator=g)

    sum_rates_test = list()
    quasi_degradation_test = list()
    power_test = list()

    for channel_indices, channels_ris_rx_array in test_loader:
        channels_ris_rx_array, channels_ris_rx, channels_direct, locations = channels_ris_rx_array

        # loc_x = locations[:, :, 0].cpu().detach().numpy()
        # loc_y = locations[:, :, 1].cpu().detach().numpy()
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # im = ax.scatter(loc_y, loc_x)
        # ax.invert_xaxis()
        # ax.set_xlabel('y (m)')
        # ax.set_ylabel('x (m)')
        # plt.show()

        phase_shifts = model(channels_ris_rx_array)

        complete_channel = compute_complete_channel_continuous(channel_tx_ris, phase_shifts, channels_ris_rx,
                                                               channels_direct, params)

        quasi_degradation = is_quasi_degradated(complete_channel, params)

        non_qd_direct = channels_direct[~quasi_degradation]
        non_qd_reflected = channels_ris_rx[~quasi_degradation]

        precoding_1 = qd_precoding(complete_channel, params, device)
        precoding_2 = zf_precoding(complete_channel, params, device)


        noma_all_rate = noma_weighted_sum_rate(complete_channel, precoding_1, params, device)
        zf_all_rate = sum_rate(complete_channel, precoding_2, params["tsnr"])

        NOMA_rate = noma_all_rate[quasi_degradation]
        ZF_rate = zf_all_rate[~quasi_degradation]

        all_rates = torch.cat([NOMA_rate, ZF_rate], dim=0)
        average_rate = torch.mean(all_rates)

        NOMA_all_power = precoding_power(precoding_1)
        ZF_all_power = precoding_power(precoding_2)

        NOMA_power = NOMA_all_power[quasi_degradation]
        ZF_power = ZF_all_power[~quasi_degradation]

        average_power_NOMA = torch.mean(NOMA_power)
        average_power_ZF = torch.mean(ZF_power)
        print(f"Batch average NOMA power = {average_power_NOMA}")
        print(f"Batch average ZF power = {average_power_ZF}")

        all_powers = torch.cat([NOMA_power, ZF_power], dim=0)
        average_power = torch.mean(all_powers)
        print(f"Batch average power = {average_power}")
        sum_rates_test.append(average_rate.item())
        power_test.append(average_power.item())
        quasi_degradation_test.append(sum(quasi_degradation) / len(quasi_degradation))

    print(f"Test results: Sum rate = {np.mean(sum_rates_test)}, "
          f"quasi degradation = {np.mean(quasi_degradation_test)} %, power = {np.mean(power_test)}")

