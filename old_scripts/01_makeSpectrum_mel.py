import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from functools import partial
import joblib
from tqdm import tqdm



def get_train_file_path(image_id):
    return "../input/g2net-gravitational-wave-detection/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "../input/g2net-gravitational-wave-detection/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)


# p_file -> spectrum
def make_spectrum(p_file, type_channel, flg_show=False, flg_save=False, OUT_DIR="./train"):
    waves = np.load(p_file).astype(np.float32) # (3, 4096)
    melspecs = []
    for j in range(3):
        melspec = librosa.feature.melspectrogram(waves[j] / max(waves[j]),
                                                 sr=4096, n_mels=128, fmin=20, fmax=2048)

        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = melspec.transpose((1, 0))
        melspecs.append(melspec)

    if type_channel==0:
        # melspecs : (channel, h, w)
        image = np.array(melspecs)

        # image : (channel*h, w)
        image = image.reshape(-1, image.shape[-1])


    elif type_channel==1:
        # melspecs : (channel, h, w)
        image = np.array(melspecs)
        # (h, w, channel)の順番にかえる
        image = image.transpose((1, 2, 0))


    elif type_channel==2:
        # melspecs : (channel, h, w)
        image = np.array(melspecs)

        factor_c = 1.5

        layer_a = image.reshape(-1, image.shape[-1])
        layer_c = image.reshape(-1, image.shape[-1]) * factor_c
        layer_b = (layer_a + layer_c) / 2

        image = np.stack([layer_a, layer_b, layer_c])

        # (h, w, channel)の順番にかえる
        image = image.transpose((1, 2, 0))

    else:
        # 1チャネル
        raise Exception("Please specify type_channel")

    if flg_show:
        title = os.path.basename(p_file)
        plt.title(title)
        plt.imshow(image)
        plt.show()

    if flg_save:
        os.makedirs(OUT_DIR, exist_ok=True)
        file_name = os.path.basename(p_file).split('.npy')[0]
        np.save(f"{OUT_DIR}/{file_name}", image)

    return image

# img_input:ndarrray (27, 128)
def type0To1(img_input):
    NUM_CHANNEL = 3
    image = img_input.reshape(NUM_CHANNEL, -1, img_input.shape[-1])

    # (h, w, channel)の順番にかえる
    img_output = image.transpose((1, 2, 0))
    return img_output

# img_input:ndarrray (27, 128)
def type0To2(img_input):
    # print(img_input.shape)
    factor_c = 1.5
    channel_a = img_input
    channel_c = img_input * factor_c
    channel_b = (channel_a + channel_c) / 2

    image = np.stack([channel_a, channel_b, channel_c])

    # (h, w, channel)の順番にかえる
    img_output = image.transpose((1, 2, 0))
    return img_output
    # return img_output

def type0ToX_all(INPUT_DIR, OUTPUT_DIR, n_type):
    flg_train_input = "train" in INPUT_DIR.lower()
    flg_train_output = "train" in OUTPUT_DIR.lower()

    assert flg_train_input == flg_train_output, f"INPUT_DIR is train: {flg_train_input}, but OUTPUT_DIR is train:{flg_train_output}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if flg_train_input:
        df_csv = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')

    else:
        df_csv = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')

    if n_type == 1:
        func_convert = type0To1

    elif n_type == 2:
        func_convert = type0To2

    else:
        raise Exception("Invalid n_type!!")


    df_csv['path_input'] = INPUT_DIR + "/" + df_csv['id'] + ".npy"
    path_input = df_csv['path_input'].values
    df_csv['path_output'] = OUTPUT_DIR + "/" + df_csv['id'] + ".npy"
    path_output = df_csv['path_output'].values


    for idx in tqdm(range(len(df_csv))):
        p_input = path_input[idx]
        p_output = path_output[idx]

        img_input = np.load(p_input)
        # print(img_input.shape)
        img_output = func_convert(img_input)

        np.save(p_output, img_output)




type_channel = 0
# type_channel == 0 : 1channel
# type_channel == 1 : 3channel (LIGO,xx,xx) (9, 128, 3)
# type_channel == 2 : 3channel  (27, 128, 3)
save_images_train = partial(make_spectrum, type_channel=type_channel, flg_show=False, flg_save=True, OUT_DIR ="./train_1channel")
save_images_test = partial(make_spectrum, type_channel=type_channel, flg_show=False, flg_save=True, OUT_DIR ="./test_1channel")


# list_path = ["./alsk.npy", "./laksj.npy","./aSijd.npy"]みたいな感じ
def save_all_images_train(list_path):

    _ = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(save_images_train)(file_path) for file_path in tqdm(list_path)
)
def save_all_images_test(list_path):
    _ = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(save_images_test)(file_path) for file_path in tqdm(list_path)
)


if __name__ == '__main__':

    # train = pd.read_csv('../input/g2net-gravitational-wave-detection/training_labels.csv')
    # test = pd.read_csv('../input/g2net-gravitational-wave-detection/sample_submission.csv')

    # train['file_path'] = train['id'].apply(get_train_file_path)
    # test['file_path'] = test['id'].apply(get_test_file_path)


    # list_path_train = train['file_path'].values
    # save_all_images_train(list_path_train)
    # list_path_test = test['file_path'].values
    # save_all_images_test(list_path_test)


    # data_type = "train"
    # data_type = "test"
    # n_type = 1
    # input_dir = f"./{data_type}_1channel"
    # output_dir = f"./{data_type}_3channel_type{n_type}"
    # type0ToX_all(input_dir, output_dir, n_type)
    # raise



    convert_types = [2]
    data_types = ["train", "test"]
    for n_type in convert_types:
        for data_type in data_types:
            input_dir = f"./{data_type}_1channel"
            output_dir = f"./{data_type}_3channel_type{n_type}"
            type0ToX_all(input_dir, output_dir, n_type)

