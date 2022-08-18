# ThermalGCN

ThermalGCN is a fast graph convolutional networks(GCN)-based method for thermal simulation of chiplet-based systems.

- Use global information (total power) as input,
- Apply the skip connection in graph convolution network,
- Integrate PNA network into the model,
- Use edge based attention network to represent the connection effect.

## Installation

ThermalGCN requires Pytorch and DGL to be installed as backend. 

## Instructions
- Random chiplet layout generation:
  
  python Generate.py

- Obtain dataset:

  python run.py
  python data_preprocess.py

- Training GCN:

  python GCNPNAGAT.py

## Publications

L. Chen, W. Jin and S. X.-D. Tan, "Fast Thermal Analysis for Chiplet Design based on Graph Convolution Networks," 2022 27th Asia and South Pacific Design Automation Conference (ASP-DAC), 2022, pp. 485-492..

## The Team

ThermalGCN was originally developed by [Liang Chen](https://vsclab.ece.ucr.edu/people/liang-chen) and [Wentian Jin](https://vsclab.ece.ucr.edu/people/wentian-jin) at [VSCLAB](https://vsclab.ece.ucr.edu/VSCLAB) under the supervision of Prof. [Sheldon Tan](https://profiles.ucr.edu/app/home/profile/sheldont).

ThermalGCN is currently maintained by [Liang Chen](https://vsclab.ece.ucr.edu/people/liang-chen).
