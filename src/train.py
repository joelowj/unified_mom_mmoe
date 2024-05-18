#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

"""
    This is a sample training & inference script.
"""


import random
import numpy as np
import pandas as pd
import torch

from functools import reduce

from src.loss_fn import neg_sharpe_ratio_loss
from src.model import UnifiedMomMMoE
from src.utils import *


"""
    Do whatever you need to make sure you can reproduce the results.
"""

# Set the seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# Ensure that PyTorch operations are deterministic
# Note: This may impact performance and not all operations are guaranteed to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Use the same device (CPU or GPU) setting as before
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    Here you prepare your features and targets that will be required for training & inference.
"""


features = pd.DataFrame([])
targets = pd.DataFrame([])

feature_ids = []
target_ids = []

"""
    This is the main training script. 
    The parameters below are tunable.
"""

LOOKBACK = 120
INPUT_DIM = 126
LSTM_EXPERT_HIDDEN_DIM = 126
LSTM_EXPERT_NUM_LAYERS = 2
NUM_EXPERTS = 12
TASK_SPECIFIC_NUM_LAYERS = 4
NUM_TASKS = 3
TASK_SPECIFIC_FINAL_NUM_LAYERS = 4


for year in range(2000, 2024):

    features_is = features[(features['date'] >= f'1990-01-01') & (features['date'] < f'{year - 1}-06-01')].copy()
    feature_is_ids = [col for col in features_is.columns if 'feature' in col]
    ticker_ids = features_is['ticker'].unique().tolist()

    targets_is = targets[(targets['date'] >= f'1990-01-01') & (targets['date'] < f'{year - 1}-06-01')].copy()
    targets_is = targets_is[targets_is['ticker'].isin(ticker_ids)]

    features_oos = features[(features['date'] >= f'{year - 1}-06-01') & (features['date'] < f'{year + 1}-01-01')].copy()
    features_oos = features_oos.loc[features_oos['ticker'].isin(ticker_ids), ['date', 'ticker'] + feature_is_ids]
    targets_oos = targets[(targets['date'] >= f'{year - 1}-06-01') & (targets['date'] < f'{year + 1}-01-01')].copy()
    targets_oos = targets_oos[targets_oos['ticker'].isin(ticker_ids)]

    lookback = LOOKBACK
    model = UnifiedMomMMoE(
        num_features=len(feature_is_ids),
        num_assets=len(targets_is['ticker'].unique()),
        input_dim=INPUT_DIM,
        lstm_expert_hidden_dim=LSTM_EXPERT_HIDDEN_DIM,
        lstm_expert_num_layers=LSTM_EXPERT_NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        task_specific_num_layers=TASK_SPECIFIC_NUM_LAYERS,
        num_tasks=NUM_TASKS,
        task_specific_final_num_layers=TASK_SPECIFIC_FINAL_NUM_LAYERS,
    )  # .to(device)  # if you have GPU then uncomment this, but make sure you also place the tensor to GPU as well.

    num_epochs = 10
    best_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_epochs = 2  # Stop if no improvement after 5 epochs

    task_specific_loss_fn_1 = torch.nn.MSELoss()
    task_specific_loss_fn_2 = torch.nn.MSELoss()
    task_specific_loss_fn_3 = torch.nn.MSELoss()

    # Assuming `model` is your multi-task model
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-5}, ])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0  # To track the loss over the epoch
        num_batches = 0
        trainX_batches = generate_batches(pivot_and_stack(features_is, index_id='date', column_ids='ticker', values_ids=feature_ids), batch_size=lookback)
        trainy_batches = generate_batches(pivot_and_stack(targets_is, index_id='date', column_ids='ticker', values_ids=target_ids), batch_size=lookback)
        for trainX, trainy in zip(trainX_batches, trainy_batches):
            num_batches += 1
            optimizer.zero_grad()
            trainX = torch.Tensor(trainX)  # .to(device)
            trainX[torch.isnan(trainX)] = 0
            trainX[torch.isinf(trainX)] = 0

            trainy = torch.Tensor(trainy)  # .to(device)
            trainy[torch.isnan(trainy)] = 0
            trainy[torch.isinf(trainy)] = 0

            if trainX.shape[0] < lookback:
                break
            assert trainX.shape[1] == trainy.shape[1]
            num_assets = trainX.shape[1]
            task_specific_networks_outpus, task_specific_network_final_output = model(trainX)
            # calculate the individual task specific loss.
            task_specific_loss_1 = task_specific_loss_fn_1(task_specific_networks_outpus[:, 0, :], trainy[:, :, 0])
            task_specific_loss_2 = task_specific_loss_fn_2(task_specific_networks_outpus[:, 1, :], trainy[:, :, 1])
            task_specific_loss_3 = task_specific_loss_fn_3(task_specific_networks_outpus[:, 2, :], trainy[:, :, 2])
            task_specific_portfolio_returns = (task_specific_networks_outpus.permute(0, 2, 1) * trainy[:, :, 3].unsqueeze(-1).repeat(1, 1, 3)).sum(dim=1) / num_assets
            total_portfolio_returns = (task_specific_portfolio_returns * task_specific_network_final_output).sum(dim=1)
            task_specific_loss_final = neg_sharpe_ratio_loss(total_portfolio_returns, None)
            task_specific_network_loss = sum([task_specific_loss_1, task_specific_loss_2, task_specific_loss_3, task_specific_loss_final]) / 4.
            loss = task_specific_network_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_epochs:
            print(f"Stopping early at epoch {epoch + 1}. No improvement in loss after {early_stopping_epochs} epochs.")
            break

    testX_batches = generate_batches(pivot_and_stack(features_oos, index_id='date', column_ids='ticker', values_ids=feature_ids), batch_size=lookback)
    testy_batches = generate_batches(pivot_and_stack(targets_oos, index_id='date', column_ids='ticker', values_ids=target_ids), batch_size=lookback)

    task_specific_network_returns_ = []
    total_portfolio_returns_ = []
    task_specific_network_weights_ = []

    with torch.no_grad():
        for testX, testy in zip(testX_batches, testy_batches):
            testX = torch.Tensor(testX)
            testX[torch.isnan(testX)] = 0
            testX[torch.isinf(testX)] = 0

            testy = torch.Tensor(testy)
            testy[torch.isnan(testy)] = 0
            testy[torch.isinf(testy)] = 0

            task_specific_networks_outpus, task_specific_network_final_output = model(testX)
            task_specific_portfolio_returns = (task_specific_networks_outpus.permute(0, 2, 1) * testy[:, :, 3].unsqueeze(-1).repeat(1, 1, 3)).sum(dim=1) / num_assets
            total_portfolio_returns = (task_specific_portfolio_returns * task_specific_network_final_output).sum(dim=1)
            task_specific_network_returns_.append(task_specific_portfolio_returns.numpy())
            total_portfolio_returns_.append(total_portfolio_returns.numpy())
            task_specific_network_weights_.append(task_specific_network_final_output.numpy())

    task_specific_network_weights_ = pd.DataFrame(np.concatenate(task_specific_network_weights_, axis=0), columns=['fast_mom_wt', 'mid_mom_wt', 'slow_mom_wt'])
    task_specific_network_weights_.index = targets_oos['date'].drop_duplicates()

    task_specific_network_returns_ = pd.DataFrame(np.concatenate(task_specific_network_returns_, axis=0), columns=['fast_mom_ret', 'mid_mom_ret', 'slow_mom_ret'])
    task_specific_network_returns_.index = targets_oos['date'].drop_duplicates()

    total_portfolio_returns_ = pd.DataFrame(np.concatenate(total_portfolio_returns_, axis=0), columns=['total_port_ret'])
    total_portfolio_returns_.index = targets_oos['date'].drop_duplicates()

    results = [task_specific_network_returns_, total_portfolio_returns_, task_specific_network_weights_]

    results = reduce(lambda l, r: pd.merge(l, r, left_index=True, right_index=True, how='outer'), results)
    results.to_csv(f"oos_{year}.csv")
