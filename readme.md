#  [NeurIPS 2023]-<tt>TopoSRL</tt>: Topology Preserving Self-Supervised Simplicial Representation Learning
Code for <tt>TopoSRL</tt> model proposed in the NeurIPS 2023 submission.

## Dependencies

- Python 3.9
- PyTorch 2.0
- dgl 1.0.2.cu113
- gudhi 3.8.0


## Datasets

Node classification: 'contact-high-school', 'contact-primary-school' and 'senate-bills'.

Simplicial closure: 'contact-high-school', 'contact-primary-school' and 'email-Enron'.

## Usage
To run the codes, use the following commands:
```python
# Node classification example
python node_classification.py --dataname contact-high-school --epochs 20 --lr 1e-3 --wd 0 --dim 4 --alpha 0.5 --snn MPSN --delta 300 --augmentation open --rho 0.1

# Simplicial closure example
python simplicial_closure.py --dataname email-Enron --epochs 20 --lr 1e-3 --wd 0 --dim 4 --alpha 0.5 --snn MPSN --delta 300 --augmentation open --rho 0.1

# Graph classification example
python graph_classification.py --dataname proteins --epochs 5 --lr 1e-3 --wd 0 --dim 4 --alpha 0.5 --snn MPSN --delta 10 --rho 0.1

# Trajectory prediction example
python graph_classification.py --epochs 5 --lr 1e-3 --wd 0 --dim 4 --alpha 0.5 --delta 10 --rho 0.1
```