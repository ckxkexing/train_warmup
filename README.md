文件结构

```
model_train_warmup
├── readme.md
├── cifar-10-batches-py
├── googleNet
│   ├── model.py
│   ├── train.py
│   ├── googlenet-without-bn.pth
│   ├── googlenet.pth
├── alexNet
│   ├── model.py
│   ├── train.py
```



### Normal net

|dropout|epoch|optim|result|
|  ----  | ----  |----|----|
|0.1|2|sgd|53%|
|0.1|4|sgd|57%|
|0.5|2|sgd|48%|
|0.5|2|adam|47%|
|0.1|2|adam|52%|
|0.1|4|adam|54%|
|0.|2|adam|56%|

### GoogLeNet
| GoogLeNet                                                    |      | 精度                      |
| ------------------------------------------------------------ | ---- | ------------------------- |
| **Training on cpu**                                          |      |                           |
| bs = 1，epoch=2，lr=0.001                                    |      | 准确度徘徊在20～30%       |
| bs = 4，epoch=2，lr=0.001                                    |      | 精度达到67%,中途最高是69% |
| bs = 8，epoch=2，lr=0.001                                    |      | 精度达到68%               |
| **Training on GPU 1080 8G**                                  |      |                           |
| bs = 8，epoch=2，lr=0.001                                    |      | 69%                       |
| bs = 16，epoch=2，lr=0.001                                   |      | 66%                       |
| bs = 32，epoch=10，lr=0.001                                  |      | 71%                       |
| bs = 64，epoch=10，lr=0.001                                  |      | 68%                       |
| bs = 64，epoch=10，lr=0.001， 在inception中添加BN层          |      | 69%                       |
| bs = 64，epoch=10，lr=0.001                                  |      | 73%                       |
| bs=128。50=74%，中途能达到75%，lr=0.001                      |      | 74%                       |
| 给所有的conv2d后面添加了bn，将解码器变为（1024，1000），（1000，128），（128，10） |      |                           |
| bs=128。epoch=50，lr=0.001                                   |      | 74%                       |
| bs=128。epoch=50，lr=0.02                                    |      | 78%                       |
| 添加了l1 norm（sum of abs of parm * 1e-7）                   |      | 78%                       |

### AlexNet

| AlexNet                     |      | 精度 |
| --------------------------- | ---- | ---- |
| **GPU 1080 8G**             |      |      |
| bs128，epoch 50， lr = 0.01 |      | 75%  |
|                             |      |      |
|                             |      |      |

### Finding && 结论
1、本次试验中，我体会到学习率的设置是重要的，前期实验，我的学习率都是1e-3的级别。后来，将学习率放大10倍后，精度可以提升4个点左右。
2、bs=1在googleNet做图像分类任务上行不通，可以发现模型根本就没有收敛。

