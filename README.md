# Getting Started

Setting up the environment:

```Bash
$ ./run venv
$ . venv/bin/activate
```

Setting up the cache directory:

```Bash
$ mkdir $HOME/datacache
$ export WDS_CACHE=$HOME/datacache
$ export WDS_CACHE_SIZE=400e9
```

Running text training with default parameters:

```Bash
$ ./ocropus4 texttrain
ngpus 1 batch_size/multiplier 12/6 actual 12
model created and is JIT-able
# logging locally
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
mcheckpoint.dirpath None
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name     | Type      | Params
---------------------------------------
0 | ctc_loss | CTCLoss   | 0     
1 | model    | TextModel | 4.8 M 
---------------------------------------
4.8 M     Trainable params
0         Non-trainable params
4.8 M     Total params
19.119    Total estimated model params size (MB)
Sanity Checking: 0it [00:00, ?it/s]/home/tmb/proj/ocropus4/venv/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:378: UserWarning: One of given dataloaders is None and it will be skipped.
  rank_zero_warn("One of given dataloaders is None and it will be skipped.")
Epoch 0: : 0it [00:00, ?it/s]# downloading http://storage.googleapis.com/nvdata-ocropus-words/generated-000226.tar to /home/tmb/gs/nvdata-ocropus-words/generated-000226.tar
# downloading http://storage.googleapis.com/nvdata-ocropus-words/generated-000162.tar to /home/tmb/gs/nvdata-ocropus-words/generated-000162.tar
# downloading http://storage.googleapis.com/nvdata-ocropus-words/uw3-word-000004.tar to /home/tmb/gs/nvdata-ocropus-words/uw3-word-000004.tar
...
```

Running segmentation training with default parameters:

```Bash
$ ./ocropus4 segtrain
# checking training batch size torch.Size([8, 3, 553, 553]) torch.Size([8, 553, 553])
model created and is JIT-able
# logging locally
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# optimizer SGD (
Parameter Group 0
    dampening: 0
    foreach: None
    lr: 0.01
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)
# scheduler <torch.optim.lr_scheduler.LambdaLR object at 0x7fb662a03a30>

  | Name  | Type     | Params
-----------------------------------
0 | model | SegModel | 9.0 M 
-----------------------------------
9.0 M     Trainable params
0         Non-trainable params
9.0 M     Total params
35.833    Total estimated model params size (MB)
...
```
