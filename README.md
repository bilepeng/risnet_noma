# risnet_noma

This repository is the source code and data for the paper

F. Siegismund-Poschmann, B. Peng and E. A. Jorswieck, "Non-Orthogonal Multiple Access Assisted by Reconfigurable Intelligent Surface Using Unsupervised Machine Learning", 31st European signal processing conference.

Run `train_noma.py` with the following arguments to train the model:

- `tsnr`: transmit SNR.
- `pmax`: maximum transmit power.
- `ris_shape`: RIS shape, default: 32 x 32.
- `lr`: learning rate, default: 1e-5.

Run `test.py` to test the saved model on the validation data set.