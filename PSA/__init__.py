from .psa import eres2net2PSA
from modelscope import snapshot_download
import pathlib
import os

cache_dir = os.path.join(os.path.expanduser('~'), ".cache/psa/")
model_pt = "eres2net_large_model.ckpt"
model_dir = snapshot_download('iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',cache_dir=cache_dir)
model_dir = pathlib.Path(model_dir)
model = os.path.join(model_dir,model_pt)
def get_PSA():
    return eres2net2PSA(model)