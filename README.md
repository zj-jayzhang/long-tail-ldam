# long-tail-ldam

This is **not** the official implementation of LDAM-DRW in the paper [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/pdf/1906.07413.pdf) in PyTorch.

## How to run the code?
- To train the ERM baseline on long-tailed imbalance with ratio of 100
```shell
python3 lt_train.py --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```

- To train the LDAM Loss along with DRW training on long-tailed imbalance with ratio of 100
```shell
python3 lt_train.py --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
```

## Others
Modified from rerpository [kaidic/LDAM-DRW](https://github.com/kaidic/LDAM-DRW)
