import torch
from torch import nn
import numpy as np
from .base import PINN
from .siren import SirenPINN, SirenDouble, SirenMagPhase, SirenConditioned
from .rff import PINN_RFF, RBFPinn, PINN_directional_RFF, Polar_RFF
from .activation import *
from .phisk_module import DirectionInterpolationModule, ContinuousDirectionHyperNetworkCorrector, PhiskModule
from .phisk_loader import load_directional_model
from .lora import init_with_Lora, init_with_Lora_rff
from .lora_interp_loader import load_interpolated_model
_all__ = ["rff", "siren", "mag_phase", "rff_polar", "re_im"]


def init_model_from_conf(conf):
    '''
    Init a model from a configuration conf.
    The model is defined by the argument 'model'.
    siren -> SIREN network with real/imaginary part representation of the ouput pressure as in Implicit Neural Representations with Periodic Activation Functions Sitzmann 2020.
    mag_phase -> SIREN with an magnitude/phase representation of the ouput pressure.
    re_im -> Two independant SIREN networks predicting Re and Im part of the ouput pressure.
    rff -> Random Fourier Features based PINN with real/imaginary representation.
    rff_polar -> Random Fourier Features based PINN with magnitude/phase representation.
    Note : You can modify RFF activation function by adding the argument (activation = your_activation)
    when calling a model. Baseline activation is QuadraticTanh as suggested in the article.
    '''
    model_type = conf['model']
    if model_type == 'siren':
        return SirenPINN(conf)
    elif model_type == 'mag_phase':
        return SirenMagPhase(conf)
    elif model_type == 're_im':
        return SirenDouble(conf)
    elif model_type == 'rff_polar':
        return Polar_RFF(conf) 
    elif model_type == 'rff':
        return PINN_RFF(conf)
    return None
    

def init_conditioned_model(conf):
    '''
    Similar as init_model_from_conf with 2 input : coordinate and direction.
    Only real/imaginary part representation of output pressure are considered but can add more.
    '''
    model_type = conf['model']
    if model_type == 'siren':
        return SirenConditioned(conf)
    elif model_type == 'rff':
        return  PINN_directional_RFF(conf)
    else:
        return None