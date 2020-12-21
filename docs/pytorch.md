# Pytorch

[Pytorch](https://pytorch.org/) is likely the most popular neural network framework given its ease of use compared to Tensorflow, and having more options then the high level framework Keras. It is developed by Facebook's AI Research lab.

## Tensors

Tensors are used in place of `numpy` in Pytorch. This allows faster processing using GPUs.

If as use `torch.as_tensor` or `torch.tensor`, it will infer the datatype from the original array and assign it as such.

```python
# convert array/list to pytorch tensor, retains a link to the array
x = torch.as_tensor(arr)
# convert array to tensor, no linkage, just a copy
x = torch.tensor(arr)
# check datatype
x.dtype
...torch.int32
```

If we want to convert the tensor to specific datatypes, we can refer to the [table](https://pytorch.org/docs/stable/tensors.html) below.


| Data type | dtype | CPU tensor | GPU tensor |  |
|-|-|-|-|-|
| 32-bit floating point | torch.float32 or torch.float | torch.FloatTensor | torch.cuda.FloatTensor |  |
| 64-bit floating point | torch.float64 or torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |  |
| 16-bit floating point 1 | torch.float16 or torch.half | torch.HalfTensor | torch.cuda.HalfTensor |  |
| 16-bit floating point 2 | torch.bfloat16 | torch.BFloat16Tensor | torch.cuda.BFloat16Tensor |  |
| 32-bit complex | torch.complex32 |  |  |  |
| 64-bit complex | torch.complex64 |  |  |  |
| 128-bit complex | torch.complex128 or torch.cdouble |  |  |  |
| 8-bit integer (unsigned) | torch.uint8 | torch.ByteTensor | torch.cuda.ByteTensor |  |
| 8-bit integer (signed) | torch.int8 | torch.CharTensor | torch.cuda.CharTensor |  |
| 16-bit integer (signed) | torch.int16 or torch.short | torch.ShortTensor | torch.cuda.ShortTensor |  |
| 32-bit integer (signed) | torch.int32 or torch.int | torch.IntTensor | torch.cuda.IntTensor |  |
| 64-bit integer (signed) | torch.int64 or torch.long | torch.LongTensor | torch.cuda.LongTensor |  |
| Boolean | torch.bool | torch.BoolTensor | torch.cuda.BoolTensor |  |

## GPU

Pytorch is able to use [GPU](https://pytorch.org/docs/stable/notes/cuda.html) to accelerate its processing speed. 

We can set cuda by writing an if-else clause. Sometimes, just adding `cuda` will not work, but we have to specify the device id, i.e. `cuda:0`

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

We can set the model to run in GPU, ideally by placing the device variable `model.to(device)`.

```python
# check if using gpu
next(model.parameters()).is_cuda
# use gpu
model.cuda()
# or
model.to(device)
```

We can do the same for the tensors, to specify them to use the GPU.

```python
a = torch.FloatTensor([1.0,2.0])
# check if using gpu
a.device
# use gpu
a.cuda()
# or
a.to(device)
```

Here's a more specify example on a train-test split dataset.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)
```