#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variational Dropout implementation for linear layers and Bayesian neural networks.

This module provides a `LinearGroupNJ` layer that implements Group Variational Dropout, as well as utilities for working with Bayesian layers.

References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
"""

import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn.modules import utils


def reparametrize(mu, logvar, sampling=True):
    """
    Apply the reparametrization trick for sampling from a Gaussian distribution.

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        logvar (torch.Tensor): Log variance of the Gaussian distribution.
        sampling (bool, optional): If True, samples using reparametrization. If False, returns the mean. Defaults to True.

    Returns:
        torch.Tensor: Sampled tensor or the mean if sampling is False.
    """
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*std.size(), device=mu.device)
        return mu + eps * std
    else:
        return mu


# -------------------------------------------------------
# LINEAR LAYER
# -------------------------------------------------------


class LinearGroupNJ(Module):
    """
    Fully Connected Group Normal-Jeffrey's (Group Variational Dropout) layer.

    Implements variational dropout for sparsifying neural networks.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        cuda (bool, optional): Whether to use CUDA. Defaults to False.
        init_weight (torch.Tensor, optional): Initial weight tensor. Defaults to None.
        init_bias (torch.Tensor, optional): Initial bias tensor. Defaults to None.
        clip_var (float, optional): Maximum variance for clipping. Defaults to None.

    Attributes:
        z_mu (torch.nn.Parameter): Mean for the variational dropout rates.
        z_logvar (torch.nn.Parameter): Log variance for the variational dropout rates.
        weight_mu (torch.nn.Parameter): Mean for the weight distribution.
        weight_logvar (torch.nn.Parameter): Log variance for the weight distribution.
        bias_mu (torch.nn.Parameter): Mean for the bias distribution.
        bias_logvar (torch.nn.Parameter): Log variance for the bias distribution.
    """

    def __init__(
        self,
        in_features,
        out_features,
        cuda=False,
        init_weight=None,
        init_bias=None,
        clip_var=None,
    ):

        super(LinearGroupNJ, self).__init__()
        self.cuda = cuda
        self.in_features = in_features
        self.out_features = out_features
        self.clip_var = clip_var
        self.deterministic = False  # flag is used for compressed inference
        # trainable params according to Eq.(6)
        # dropout params
        self.z_mu = Parameter(torch.Tensor(in_features))
        self.z_logvar = Parameter(torch.Tensor(in_features))  # = z_mu^2 * alpha
        # weight params
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_logvar = Parameter(torch.Tensor(out_features))

        # init params either random or with pretrained net
        self.reset_parameters(init_weight, init_bias)

        # activations for kl
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # numerical stability param
        self.epsilon = 1e-8

    def reset_parameters(self, init_weight, init_bias):
        """
        Initialize the layer parameters.

        Args:
            init_weight (torch.Tensor, optional): Initial weight tensor. Defaults to None.
            init_bias (torch.Tensor, optional): Initial bias tensor. Defaults to None.
        """
        # init means
        stdv = 1.0 / math.sqrt(self.weight_mu.size(1))

        self.z_mu.data.normal_(1, 1e-2)

        if init_weight is not None:
            self.weight_mu.data = torch.Tensor(init_weight)
        else:
            self.weight_mu.data.normal_(0, stdv)

        if init_bias is not None:
            self.bias_mu.data = torch.Tensor(init_bias)
        else:
            self.bias_mu.data.fill_(0)

        # init logvars
        self.z_logvar.data.normal_(-9, 1e-2)
        self.weight_logvar.data.normal_(-9, 1e-2)
        self.bias_logvar.data.normal_(-9, 1e-2)

    def clip_variances(self):
        if self.clip_var:
            self.weight_logvar.data.clamp_(max=math.log(self.clip_var))
            self.bias_logvar.data.clamp_(max=math.log(self.clip_var))

    def get_log_dropout_rates(self):
        log_alpha = self.z_logvar - torch.log(self.z_mu.pow(2) + self.epsilon)
        return log_alpha

    def compute_posterior_params(self):
        weight_var, z_var = self.weight_logvar.exp(), self.z_logvar.exp()
        self.post_weight_var = (
            self.z_mu.pow(2) * weight_var
            + z_var * self.weight_mu.pow(2)
            + z_var * weight_var
        )
        self.post_weight_mu = self.weight_mu * self.z_mu
        return self.post_weight_mu, self.post_weight_var

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying variational dropout and the linear transformation.
        """
        if self.deterministic:
            assert (
                self.training == False
            ), "Flag deterministic is True. This should not be used in training."
            return F.linear(x, self.post_weight_mu, self.bias_mu)

        batch_size = x.size()[0]
        # compute z
        # note that we reparametrise according to [2] Eq. (11) (not [1])
        z = reparametrize(
            self.z_mu.repeat(batch_size, 1),
            self.z_logvar.repeat(batch_size, 1),
            sampling=self.training,
        )

        while len(x.size()) > len(z.size()):
            z = z.unsqueeze(1)

        # apply local reparametrisation trick see [1] Eq. (6)
        # to the parametrisation given in [3] Eq. (6)

        xz = x * z
        mu_activations = F.linear(xz, self.weight_mu, self.bias_mu)
        var_activations = F.linear(
            xz.pow(2), self.weight_logvar.exp(), self.bias_logvar.exp()
        )

        return reparametrize(
            mu_activations, var_activations.log(), sampling=self.training
        )

    def kl_divergence(self):
        """
        Compute the KL divergence for the layer.

        Returns:
            torch.Tensor: KL divergence value.
        """
        # KL(q(z)||p(z))
        # we use the kl divergence approximation given by [2] Eq.(14)
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.get_log_dropout_rates()
        KLD = -torch.sum(
            k1 * self.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * self.softplus(-log_alpha)
            - k1
        )

        # KL(q(w|z)||p(w|z))
        # we use the kl divergence given by [3] Eq.(8)
        KLD_element = (
            -0.5 * self.weight_logvar
            + 0.5 * (self.weight_logvar.exp() + self.weight_mu.pow(2))
            - 0.5
        )
        KLD += torch.sum(KLD_element)

        # KL bias
        KLD_element = (
            -0.5 * self.bias_logvar
            + 0.5 * (self.bias_logvar.exp() + self.bias_mu.pow(2))
            - 0.5
        )
        KLD += torch.sum(KLD_element)

        return KLD

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


BAYESIAN_LAYERS = LinearGroupNJ


def get_kl_modules(model: nn.Module):
    """
    Generator for iterating over Bayesian layers in a model.

    Args:
        model (nn.Module): The model containing Bayesian layers.

    Yields:
        nn.Module: A Bayesian layer in the model.
    """
    for module in model.modules():
        if isinstance(module, BAYESIAN_LAYERS):
            yield module
