import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch
from nnAudio.Spectrogram import CQT1992v2
# from nnAudio.Spectrogram import CQT

from functools import partial
import joblib
from tqdm import tqdm
from nnAudio.Spectrogram import CQT1992v2
from _00_utils import *


cfg_transform = {
    "sr":2048,
    "fmin":20,
    # "hop_length":64,
    "hop_length":32,
    "freq_bins":64
}
# FUNC_CQT = CQT1992v2()

sr = cfg_transform["sr"]
fmin = cfg_transform["fmin"]
hop_length = cfg_transform["hop_length"]
freq_bins = cfg_transform["freq_bins"]
FUNC_CQT = CQT1992v2(sr=sr, fmin=fmin,hop_length=hop_length, n_bins=freq_bins)

# ここまで修正済み
# todo
# type4作る
# このプログラム内で使われているconvert2cqt, convert2melの引数修正

#####################################################
# makeSpectrumに渡す変換関数を定義する
# func(waves, input_shape, output_shape)でなくてはならない
# 出力はimageで、image.shape:(h, w, channel) channelは1or3


# waves: (3, 4096)
# image: (27, 128, 1)
@check_shape
def func_mel_1channel(waves, input_shape, output_shape):
    melspecs = []

    for wave in waves:
        melspec = convert2mel(wave)
        melspecs.append(melspec)

    # melspecs : (channel, h, w)
    image = np.array(melspecs)

    # image : (channel*h, w)
    image = image.reshape(-1, image.shape[-1], 1)

    del melspecs
    return image


# waves: (3, 4096)
# melspecs: (3, 27, 128)
# image: (3*27, 128)
@check_shape
def func_mel_3channel(waves, input_shape, output_shape):
    melspecs = []

    for wave in waves:
        melspec = convert2mel(wave)
        melspecs.append(melspec)

    # melspecs : (channel, h, w)
    image = np.array(melspecs)

    # (h, w, channel)の順番にかえる
    image = image.transpose((1, 2, 0))

    del melspecs
    return image


# waves: (3, 4096)
# image: (27, 128, 3)
@check_shape
def func_mel_3channel_weightedstack(waves, input_shape, output_shape):

    # image: (27, 128)
    image = np.vstack([convert2mel(wave) for wave in waves])

    factors = [1, (1+1.5)/2, 1.5]
    assert len(factors)==3, f"len(factors) should be 3, but {len(factors)}"
    image = np.stack([image*factor for factor in factors], axis=2)

    return image


@check_shape
def func_cqt_1channel(waves, input_shape, output_shape):

    # image : (channel, h, w)
    image = np.vstack([convert2cqt(wave) for wave in waves])

    # image : (h, w, channel)
    image = image.transpose((1, 2, 0))

    # image : (h, w*channel, 1)
    image = image.reshape(image.shape[0], -1, 1)

    # image : (h, w*channel, 3)
    # image = np.stack([image for _ in range(3)], axis=2)

    return image

@check_shape
def func_mel_cqt_3channel_simplestack(waves, input_shape, output_shape):
    layer_a = []
    layer_c = []

    for wave in waves:
        # mel_shape:(64, 65)
        mel = convert2mel(wave, cfg_transform)
        mel = min_max(mel)
        layer_a.append(mel)

        # cqt_shape:(64, 65)
        cqt = convert2cqt(wave, FUNC_CQT)
        cqt = min_max(cqt)
        layer_c.append(cqt)

    # layer_a.shape : (3*64, 65)
    layer_a = np.concatenate(layer_a, axis=0)
    layer_c = np.concatenate(layer_c, axis=0)
    layer_b = (layer_a + layer_c) / 2

    # image.shape : (3*64, 65, 3)
    image = np.stack([layer_a, layer_b, layer_c], axis=-1)

    del layer_a, layer_b, layer_c
    return image

@check_shape
def func_cqt_3channel_sitestack(waves, input_shape, output_shape):
    # image : (channel, h, w)
    image = np.stack([min_max(convert2cqt(wave, FUNC_CQT)) for wave in waves])

    # image : (h, w, channel)
    image = image.transpose((1, 2, 0))

    return image

@check_shape
def func_cqt_3channel_bandpass_boxcox(waves, input_shape, output_shape):
    # image : (channel, h, w)

    def myprep(wave):
        wave_filtered = filtering(wave, sr)
        wave_mm = min_max(wave_filtered)
        wave_bc = myboxcox_1channel(wave_mm)

        return wave_bc

    waves = np.apply_along_axis(myprep, 1, waves)
    # waves = myprep(waves)

    image = np.stack([min_max(convert2cqt(wave, FUNC_CQT)) for wave in waves])

    # image : (h, w, channel)
    image = image.transpose((1, 2, 0))

    return image

#####################################################
# グローバル変数
type2func = {
    "0" : func_mel_1channel,
    "1" : func_mel_3channel,
    "2" : func_mel_3channel_weightedstack,
    "3" : func_cqt_1channel,
    "4" : func_mel_cqt_3channel_simplestack,
    "5" : func_cqt_3channel_sitestack,
    "6" : func_cqt_3channel_bandpass_boxcox
    }

ORIGNAL_SHAPE = (3, 4096)

#####################################################

#####################################################
# p_file -> spectrum
def make_spectrum(p_file, cfg, OUT_DIR, flg_show=False):
    waves = np.load(p_file).astype(np.float32) # (3, 4096)

    input_shape = cfg["input_shape"]
    output_shape = cfg["output_shape"]
    n_type = cfg["type"]

    conversion_func = type2func[n_type]

    assert waves.shape == input_shape, f"shapes are incompatible. expected input_shape :{input_shape}, but waves.shape:{waves.shape}"

    image = conversion_func(waves=waves, input_shape=input_shape, output_shape=output_shape)

    assert image.shape == output_shape, f"shapes are incompatible. expected output_shape :{output_shape}, but image.shape:{image.shape}"

    if flg_show:
        title = os.path.basename(p_file)
        plt.title(title)
        plt.imshow(image)
        plt.show()

    # saveするパート
    os.makedirs(OUT_DIR, exist_ok=True)
    file_name = os.path.basename(p_file).split('.npy')[0]
    np.save(f"{OUT_DIR}/{file_name}", image)

    del image, waves
    # return image

#####################################################


#####################################################
# 並列化

def do_parallel(cfg, is_train):
    n_type = cfg["type"]


    if is_train:
        list_path = get_list_of_train_path()
        func_parallel = partial(make_spectrum, cfg=cfg, OUT_DIR =f"./train_type{n_type}", flg_show=False)
    else:
        list_path = get_list_of_test_path()
        func_parallel = partial(make_spectrum, cfg=cfg, OUT_DIR =f"./test_type{n_type}", flg_show=False)

    _ = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(func_parallel)(file_path) for file_path in tqdm(list_path)
)
#####################################################


if __name__ == '__main__':

    # type == "0" : 1channel
    # type == "1" : 3channel (LIGO,xx,xx) (9, 128, 3)
    # type == "2" : 3channel  (27, 128, 3)

    cfg = {
        "type":"3",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(69, 195, 1)
    }

    cfg = {
        "type":"0",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(27, 128, 1)
    }

    cfg = {
        "type":"1",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(27, 128, 1)
    }


    cfg = {
        "type":"2",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(27, 128, 3)
    }

    cfg = {
        "type":"4",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(192, 65, 3)
    }

    cfg = {
        "type":"5",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(64, 129, 3)
    }

    cfg = {
        "type":"6",
        "input_shape":ORIGNAL_SHAPE,
        "output_shape":(64, 129, 3)
    }



    # is_train = False
    # do_parallel(cfg=cfg, is_train=is_train)
    for is_train in [True, False]:
        do_parallel(cfg=cfg, is_train=is_train)

