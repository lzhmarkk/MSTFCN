# Multi-mode Spatial-Temporal Data Modeling with Fully Connected Networks

This is PyTorch implementation of MSTFCN in the following paper:

**Multi-mode Spatial-Temporal Data Modeling with Fully Connected Networks**. KSEM, 2024.

## Data

All datasets have been allocated in `./data/h5data/` folder.

## Model Training

1. Change **model settings** in `./model/{MODEL}/config.json`.
2. Change **running settings** in `./config.json`.

   In `data` field, `nyc-mix`, `chicago-mix` and `beijing-mix` are available.

   Your can change `expid` field to name the experiments.

3. Run `train_multi_step_mix.py`!

4. Results will be saved in `./saves/{DATASET}/{MODEL}/{expid}` folder.

## Model Inference

If you want to only inference trained models, run `test_multi_step_mix.py`.

## Add More Models...

If you want to add more models:

1. Create a new directory in `./model/`, place you `MYMODEL.py` and create a `config.json` file to record your hyper-parameters.

2. Register your model in `./model/__init__.py` and `./util.py`.

   + If your model requires auxiliary information such as predefined graph, write it in `get_auxiliary()` function.
   + Init your model in `get_model()` function.

3. A little changes on the `forward()` function of your model:

   + The forward function of your model must be `forward(input, **kwargs)`.
   + The input has shape `(B, T, N, C + 2)`. 
   The value of C is 4, with the first 2 entries refer to features of modality 1, and the last 2 entries refer to features of modality 2.
   + The remaining 2 refers to time semantic information `(B, T, N, 2)`.
   + The output must have shape `(B, T, N, C)`, without time semantic.

4. Run!
