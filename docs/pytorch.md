# Pytorch

[Pytorch](https://pytorch.org/) is likely the most popular neural network framework given its ease of use compared to Tensorflow, and having more options then the high level framework Keras.

## GPU

Pytorch is able to use [GPU](https://pytorch.org/docs/stable/notes/cuda.html) to accelerate its processing speed. 

### Activate CUDA

Sometimes, just adding `cuda` will not work, but we have to specify the device id, i.e. `cuda:0`

```python
import torch
print('PyTorch Version:', torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print('Number of GPUs:', torch.cuda.device_count())
    print('Device properties:', torch.cuda.get_device_properties(0))
    print('Device ID:', torch.cuda.current_device())
    print('Device Name:', torch.cuda.get_device_name(0))
```

```bash
Number of GPUs: 1
Device properties: _CudaDeviceProperties(name='Quadro P1000', major=6, minor=1, total_memory=4040MB, multi_processor_count=4)
Device ID: 0
Device Name: Quadro P1000
```

### Model

```python
# check if using gpu
next(model.parameters()).is_cuda
# use gpu
model.cuda()
# or
model.to(device)
```

### Data

```python
a = torch.FloatTensor([1.0,2.0])
# check if using gpu
a.device
# use gpu
a.cuda()
# or
a.to(device)
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()
```