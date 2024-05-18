#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import torch


def neg_sharpe_ratio_loss(
        y_pred,
        y_target,
        threshold: float = 1.,
        use_modified_sr: bool = True,
):
    epsilon = 1e-8
    threshold = torch.tensor(threshold)
    portfolio_ret = y_pred
    portfolio_ret_mean_ann = portfolio_ret.mean() * 252
    portfolio_ret_std_ann = portfolio_ret.std() * torch.sqrt(torch.tensor(252.0)) + epsilon
    portfolio_sharpe_ratio = portfolio_ret_mean_ann / portfolio_ret_std_ann
    if use_modified_sr:
        upper_linear_part = torch.minimum(portfolio_sharpe_ratio, threshold)
        upper_excess_part = torch.maximum(portfolio_sharpe_ratio - threshold, torch.tensor(0.0))
        lower_linear_part = torch.maximum(upper_linear_part, -threshold)
        lower_excess_part = torch.minimum(portfolio_sharpe_ratio - threshold, torch.tensor(0.0))
        modified_sharpe_ratio = lower_linear_part + torch.log1p(upper_excess_part) - torch.log1p(-lower_excess_part)
        return -modified_sharpe_ratio
    return -portfolio_sharpe_ratio
