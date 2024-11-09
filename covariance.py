import os 
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import time
import matplotlib.pyplot as plt

"""
    Get the data from items.json and use torch.cov and torch.corrcoef to calculate the correlation matrix for them
"""

class coeff_matrix:
    items = None
    def __init__():
        if os.path.exists("items.json"):
            with open("items.json", "r") as i:
            items = json.load(items)
