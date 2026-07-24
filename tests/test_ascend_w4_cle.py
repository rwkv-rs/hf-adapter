import torch
import torch.nn as nn
import torch.nn.functional as F
from rwkv7_hf.ascend_w4_cle import apply_sqrelu_channel_equalization,calibrate_sqrelu_value_w4


def test_sqrelu_channel_equalization_is_mathematically_exact():
    torch.manual_seed(4)
    key=nn.Linear(8,16,bias=True,dtype=torch.float64);value=nn.Linear(16,8,bias=False,dtype=torch.float64)
    x=torch.randn(3,5,8,dtype=torch.float64)
    ref=value(torch.relu(key(x)).square())
    scale=torch.exp(torch.linspace(-1.0,1.0,16,dtype=torch.float64))
    apply_sqrelu_channel_equalization(key,value,scale)
    out=value(torch.relu(key(x)).square())
    torch.testing.assert_close(out,ref,rtol=1e-12,atol=1e-12)


def test_cle_grid_never_worse_than_identity_candidate():
    torch.manual_seed(5)
    key=nn.Linear(8,16,bias=False,dtype=torch.float32);value=nn.Linear(16,8,bias=False,dtype=torch.float32)
    x=torch.randn(64,8)
    baseline=calibrate_sqrelu_value_w4(key,value,x,group_size=4,alphas=())
    chosen=calibrate_sqrelu_value_w4(key,value,x,group_size=4,alphas=(0.0,0.25,0.5,0.75,1.0))
    assert chosen.mse<=baseline.mse+1e-12
    assert chosen.scale.shape==(16,) and bool((chosen.scale>0).all())
