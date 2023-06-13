from util import RTChannelsLocation, compute_complete_channel_discrete, \
    prepare_channel_tx_ris, compute_complete_channel_continuous, average_precoding_power, \
    is_quasi_degradated, sum_rate, qd_precoding, \
    determine_kappa
from core import RISNet
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from params import params
import argparse
import datetime
from pathlib import Path

tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsnr")
    parser.add_argument("--ris_shape")
    parser.add_argument("--weights")
    parser.add_argument("--lr")
    args = parser.parse_args()
    if args.tsnr is not None:
        params["tsnr"] = float(args.tsnr)
    if args.lr is not None:
        params["lr"] = float(args.lr)
    if args.weights is not None:
        weights = args.weights.split(',')
        params["alphas"] = np.array([float(w) for w in weights])
    if args.ris_shape is not None:
        ris_shape = args.ris_shape.split(',')
        params["ris_shape"] = tuple([int(s) for s in ris_shape])

    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    Path(params["results_path_NOMA"] + dt_string).mkdir(parents=True, exist_ok=True)
    params["results_path_NOMA"] = params["results_path_NOMA"] + dt_string + "/"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: " + str(device))

    model = RISNet().to(device)

    params["frequencies"] = params["frequencies"][None, :, None]

    optimizer = optim.Adam(model.parameters(), params['lr'])
    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)
    data_set = RTChannelsLocation(params, channel_tx_ris_pinv, device)
    result_name = "ris_" + str(params['tsnr']) + "_" + str(params['ris_shape']) + '_' + str(params['alphas']) + "_"
    train_loader = DataLoader(dataset=data_set, batch_size=params['batch_size'], shuffle=True)
    losses = list()
    qd_percentage = list()
    powers = list()
    rates = list()

    if tb:
        writer = SummaryWriter(logdir=params["results_path_NOMA"])
    model.train()

    kappa = torch.tensor(0)
    previous_entropies = torch.ones(100).to(device) * 1.8

    for epoch in range(params['epoch'] + 1):
        losses_current_epoch = list()
        entropy_current_epoch = list()
        quasi_degradation_current_epoch = list()
        power_current_epoch = list()
        sum_rates_current_epoch = list()

        for channel_indices, channels_ris_rx_array in train_loader:
            if params['los']:
                channels_ris_rx_array, channels_ris_rx, channels_direct, locations = channels_ris_rx_array
            else:
                channels_ris_rx_array, channels_ris_rx = channels_ris_rx_array
                channels_direct = None

            optimizer.zero_grad()
            if params["phase_shift"] == "discrete":
                fcn_raw_output, entropy = model(channels_ris_rx_array)
                entropy_current_epoch.append(entropy.item())
                complete_channel = compute_complete_channel_discrete(channel_tx_ris, fcn_raw_output,
                                                                     channels_ris_rx, channels_direct, params)
            elif params["phase_shift"] == "continuous":
                fcn_raw_output = model(channels_ris_rx_array)
                complete_channel = compute_complete_channel_continuous(channel_tx_ris, fcn_raw_output,
                                                                       channels_ris_rx, channels_direct, params)
            else:
                raise ValueError("Unkown phase shift.")

            precoding = qd_precoding(complete_channel, params, device)

            power = average_precoding_power(precoding)

            # rate = torch.mean(noma_weighted_sum_rate(complete_channel, precoding, params))
            rate = torch.mean(sum_rate(complete_channel, precoding, params["tsnr"]))

            constraint = torch.mean(torch.log(1 + is_quasi_degradated(complete_channel, params, True)))

            if params["phase_shift"] == "continuous":
                loss = power  # constraint + 1e-2 * power
            elif params["phase_shift"] == "discrete":
                loss = constraint + 1e-2 * power + kappa * entropy
                previous_entropies = torch.cat([previous_entropies, torch.unsqueeze(entropy, dim=0)])
                kappa = determine_kappa(previous_entropies, kappa, params)

            loss.backward()
            optimizer.step()

            losses_current_epoch.append(loss.item())
            power_current_epoch.append(power.item())
            sum_rates_current_epoch.append(rate.item())
            qd = is_quasi_degradated(complete_channel, params)
            quasi_degradation_current_epoch.append((sum(qd) / len(qd)).item())

        losses.append(np.mean(losses_current_epoch))
        powers.append(np.mean(power_current_epoch))
        rates.append(np.mean(sum_rates_current_epoch))
        qd_percentage.append(np.mean(quasi_degradation_current_epoch))

        print(f'Epoch {epoch}, loss = {losses[-1]:.5f}, power: {powers[-1]:.3e}, qd_percentage: {qd_percentage[-1]:.5f},'
              f' rate: {rates[-1]:.5f}')  # , entropy: {entropy:.5f}, kappa = {kappa:.5f}

        if tb:
            writer.add_scalar("Training/Loss", losses[-1], epoch)
            writer.add_scalar("Training/Power", powers[-1], epoch)
            writer.add_scalar("Training/qd_percentage", qd_percentage[-1], epoch)
            if params["phase_shift"] == "discrete":
                writer.add_scalar("Training/entropy", np.mean(entropy_current_epoch), epoch)

        if epoch % params['saving_frequency'] == 0:
            torch.save(model.state_dict(), params['results_path_NOMA'] + result_name + '{epoch}'.format(epoch=epoch))
            np.save(params['results_path_NOMA'] + result_name + 'losses.npy', np.array(losses))
            if tb:
                writer.flush()
