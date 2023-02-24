import numpy as np
import deepchem as dc
from deepchem.data.datasets import NumpyDataset
dataset = NumpyDataset(X=np.random.rand(5, 3), y=np.random.rand(5,), ids=np.arange(5))
print(dataset)

x,y = 