#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmExpert(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LstmExpert, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class GatingMechanism(nn.Module):

    def __init__(self, input_dim, num_experts):
        super(GatingMechanism, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Assuming x is of shape [batch_size, time, features], flatten time and features for Linear
        batch_size = x.size(0)
        x_flattened = x.view(batch_size, -1)  # This might be incorrect based on your input shape
        gate_weights = self.gate(x_flattened)
        gate_weights = F.softmax(gate_weights, dim=-1)
        return gate_weights


class TaskSpecificNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, softmax_on_final_output=False):
        super(TaskSpecificNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim))
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(output_dim, output_dim))
        if num_layers > 1:
            self.layers.append(nn.Linear(output_dim, output_dim))
        self.activation_tanh = nn.Tanh()
        self.softmax_on_final_output = softmax_on_final_output

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_tanh(layer(x))
        x = self.layers[-1](x)
        if self.softmax_on_final_output:
            x = F.softmax(x, dim=-1)
        return x


class UnifiedMomMMoE(nn.Module):

    def __init__(
            self,
            num_features,
            num_assets,
            input_dim,
            lstm_expert_hidden_dim,
            lstm_expert_num_layers,
            num_experts,
            task_specific_num_layers,
            num_tasks,
            task_specific_final_num_layers,
    ):
        super(UnifiedMomMMoE, self).__init__()
        self.num_features = num_features
        self.num_assets = num_assets

        self.experts = nn.ModuleList([
            LstmExpert(input_dim, lstm_expert_hidden_dim, lstm_expert_num_layers)
            for _ in range(num_experts)
        ])

        self.task_specific_gates = nn.ModuleList([
            GatingMechanism(self.num_features * self.num_assets, num_experts)
            for _ in range(num_experts)
        ])
        self.task_specific_networks = nn.ModuleList([
            TaskSpecificNetwork(
                lstm_expert_hidden_dim,
                self.num_assets,
                task_specific_num_layers
            )
            for _ in range(num_tasks)
        ])

        self.task_specific_gate_final = GatingMechanism(self.num_features * self.num_assets, num_tasks)
        self.task_specific_network_final = TaskSpecificNetwork(
            self.num_assets,
            num_tasks,
            task_specific_final_num_layers,
            True
        )

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        task_specific_networks_outputs = []
        for gate, task_specific_network in zip(self.task_specific_gates, self.task_specific_networks):
            gate_weights = gate(x).unsqueeze(-1)  # Shape (batch_size, num_experts, 1)
            expert_outputs_weighted = (expert_outputs * gate_weights).sum(dim=1)
            task_specific_network_outputs = task_specific_network(expert_outputs_weighted)
            task_specific_networks_outputs.append(task_specific_network_outputs)
        final_gate_weights = self.task_specific_gate_final(x).unsqueeze(-1)
        task_specific_networks_outputs = torch.stack(task_specific_networks_outputs, dim=1)
        task_specific_networks_outputs_weighted = (
                task_specific_networks_outputs *
                final_gate_weights
        ).sum(dim=1)
        task_specific_network_final_output = self.task_specific_network_final(task_specific_networks_outputs_weighted)
        return task_specific_networks_outputs, task_specific_network_final_output
