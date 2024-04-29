# UnifiedMomMMoE: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts

## Abstract 
This paper introduces UnifiedMomMMoE, a deep learning framework that enhances portfolio management by leveraging a multi-task learning approach and a multi-gate mixture of experts. The essence of UnifiedMomMMoE lies in its ability to create unified momentum portfolios that incorporate the dynamics of time series momentum across a spectrum of speeds, a capability that traditional momentum strategies often miss. By doing so, UnifiedMomMMoE crafts robust portfolios that demonstrate superior risk-adjusted performance. Our extensive backtesting, which spans a range of asset classes such as equity indexes and commodities, reveals that UnifiedMomMMoE consistently outperforms benchmark models, even after accounting for transaction costs. This performance highlights UnifiedMomMMoE's ability to capture the full spectrum of momentum opportunities within financial markets, presenting a compelling solution for practitioners aiming to leverage the entire spectrum of momentum opportunities within financial markets. The results underscore UnifiedMomMMoE's effectiveness in navigating the complexities of momentum strategies, making it a valuable strategy for enhancing risk-adjusted returns.
## Note
This repository contains code snippets accompanying the paper 'UnifiedMomMMoE: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts'. We believe this is more than sufficient to reproduce the results and serve as a guidance for readers who wish to extend the research.

- `model.py`

In addition, the bulk of the codes were not open-sourced as they are proprietary research framework code based. These include the backtesting framework, futures contracts price adjustments, etc. Having said that, this would not hamper your ability to conduct research, as the three scripts released above are more than sufficient to aid you.

Due to the data licensing agreement, we cannot share the data. There are a few ways to do this (i) you may use a Bloomberg Terminal to extract the futures contract's price and use a backward ratio adjusted method to stitch the prices together to obtain a continuous price series (ii) purchase the futures contracts pricing data from data vendors (i.e., CQG, Pinnacle Data, etc.). We are not affiliated with the data vendors and do not receive anything from mentioning them here. They are viable options for obtaining futures pricing data to conduct research.

For questions or suggestion, please write an email to joel_ong at mymail dot sutd dot edu dot sg

If you read our paper and use the code to extend your research, we would appreciate it if you could cite our paper!
