"""不用 PyTorch,把 Real-ESRGAN SRVGGNetCompact 的 .pth 權重轉成 ONNX。
用法:python convert_sr_model.py 權重.pth 輸出.onnx
權重下載:https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
(轉好的模型已放在 python_engine/models/,一般不需要再轉。)
.pth 是 zip 包著 pickle;只允許重建張量所需的少數類別,不執行其他任何程式。"""
import pickle, sys, zipfile, collections
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
root = z.namelist()[0].split('/')[0]
DT = {'FloatStorage': np.float32, 'HalfStorage': np.float16}

class Storage:
    def __init__(self, key, dtype): self.key, self.dtype = key, dtype

def rebuild(storage, offset, size, stride, *args):
    raw = np.frombuffer(z.read(f'{root}/data/{storage.key}'), storage.dtype)
    n = int(np.prod(size)) if size else 1
    arr = np.lib.stride_tricks.as_strided(raw[offset:], shape=size, strides=[s * raw.itemsize for s in stride])
    return np.array(arr, dtype=np.float32)

class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) == ('torch._utils', '_rebuild_tensor_v2'): return rebuild
        if (mod, name) == ('collections', 'OrderedDict'): return collections.OrderedDict
        if mod == 'torch' and name in DT: return name
        raise pickle.UnpicklingError(f'不允許的類別 {mod}.{name}')
    def persistent_load(self, pid):
        _, kind, key, _, _ = pid
        return Storage(key, DT[kind])

sd = U(z.open(f'{root}/data.pkl')).load()
sd = sd.get('params', sd)
keys = sorted({int(k.split('.')[1]) for k in sd}, key=int)
nodes, inits = [], []
cur = 'input'
idx = 0
for i in keys:
    w = sd[f'body.{i}.weight']
    if w.ndim == 4:  # conv
        b = sd[f'body.{i}.bias']
        inits += [numpy_helper.from_array(w, f'w{i}'), numpy_helper.from_array(b, f'b{i}')]
        nodes.append(helper.make_node('Conv', [cur, f'w{i}', f'b{i}'], [f'y{i}'], pads=[1, 1, 1, 1], kernel_shape=[3, 3]))
    else:  # PReLU(num_parameters=C) → slope 形狀 (C,1,1)
        inits.append(numpy_helper.from_array(w.reshape(-1, 1, 1), f'w{i}'))
        nodes.append(helper.make_node('PRelu', [cur, f'w{i}'], [f'y{i}']))
    cur = f'y{i}'
scale = int(round(np.sqrt(sd[f'body.{keys[-1]}.weight'].shape[0] // 3)))
nodes.append(helper.make_node('DepthToSpace', [cur], ['ps'], blocksize=scale, mode='CRD'))
# 殘差:輸入最近鄰放大後相加
inits.append(numpy_helper.from_array(np.array([1, 1, scale, scale], np.float32), 'scales'))
nodes.append(helper.make_node('Resize', ['input', '', 'scales'], ['up'], mode='nearest'))
nodes.append(helper.make_node('Add', ['ps', 'up'], ['output']))
g = helper.make_graph(nodes, 'srvgg', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, None, None])],
                      [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, None, None])], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
m.ir_version = 8  # 讓較舊的 onnxruntime 與 OpenCV 都能讀
onnx.checker.check_model(m)
onnx.save(m, dst)
print('層數', len(keys), '放大', scale, '參數', sum(v.size for v in sd.values()))
