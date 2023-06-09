import numpy as np


# RT channel model
params = {'lr': 1e-5,
          'fcn': False,
          'lr_wmmse': 2e-6,
          'lr_kappa': 1e-4,
          'epsilon': 1e3,
          'epoch': 10000,
          'iter_wmmse': 400,
          'epoch_per_iter_wmmse': 10,
          'alphas': [0.5, 0.5],
          'min_SNR_ratio': [1, 1],
          'feature_dim': 20,
          'saving_frequency': 1000,
          'wmmse_saving_frequency': 50,
          'batch_size': int(1024 / 2),
          'results_path_OMA': 'results_OMA/',
          'results_path_NOMA': 'results_NOMA/',
          'results_path_NOMA_global': 'results_NOMA_global/',
          'results_path_location': 'results_location/',
          # 'factor_normalization': 1e6,
          'tsnr': 1e11,
          "frequencies": np.linspace(1e6, 1e6 + 100, 10),
          'quantile2keep': 0.6,
          "Permutation_Invariant": False,
          "phase_shift": "continuous",
          'amplification_ris': 1e4,
          'amplification_direct': 1e4,
          'ris_shape': (32, 32),
          'channel_tx_ris_original_shape': (32, 32, 9),  # width, height of RIS and Tx antennas. Do not change this!
          'channel_ris_rx_original_shape': (5000, 2, 32, 32), # (10000, 2, 32, 32), (5000, 2, 32, 32),  # samples, width, height of RIS and users
          'n_tx_antennas': 9,
          'los': True,
          'precoding': 'wmmse',
          'channel_direct_path': 'data/channels_direct.pt',
          'channel_tx_ris_path': 'data/channel_tx_ris.pt',
          'channel_ris_rx_path': 'data/channels_ris_rx.pt',
          'locations_path': 'data/locations.pt',
          'channel_ris_rx_array_path': 'data/channel_ris_rx_array.pt',
          'trained_mmse_model': None,
          'channel_estimate_error': 0,
          'debugging': True,
          'testing': True,
          }
