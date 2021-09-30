import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import librosa
import librosa.display

P_TRAIN_LABEL = "../input/g2net-gravitational-wave-detection/training_labels.csv"
DF_TRAIN_LABEL = pd.read_csv(P_TRAIN_LABEL)
p_parent_train = "./train"
DF_TRAIN_LABEL["path"] = p_parent_train + "/" + DF_TRAIN_LABEL["id"] + ".npy"
# print(DF_TRAIN_LABEL.head())

def showimg(p, label, show=True):
    img = np.load(p)
    name = p.split("/")[-1].split(".")[-2]
    title = f"id:{name}    label:{label}"
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.title(title)
    if show:
        plt.show()
    plt.close()
    return fig, name


def save_imgs(num_save):

    num_all = len(DF_TRAIN_LABEL)
    indices = random.sample(range(num_all), num_save)

    for idx in indices:
        p = DF_TRAIN_LABEL["path"][idx]
        label = DF_TRAIN_LABEL["target"][idx]
        fig, title = showimg(p ,label, show=False)
        p_save = f"./visualize/{title}.jpg"
        fig.savefig(p_save)

    num_true = (DF_TRAIN_LABEL["target"].iloc[indices]==1).sum()
    num_false = (DF_TRAIN_LABEL["target"].iloc[indices]==0).sum()
    print(f"True: {num_true}/{num_save}")
    print(f"False: {num_false}/{num_save}")

def myCQT(p):
    sgnls = np.load(p)
    print(sgnls.shape)
    fig = plt.figure()
    sr = 2048
    for i in range(3):
        x = [_x for _x in range(len(sgnls[i,:]))]
        # print(x)
        # print(y)
        y = sgnls[i,:]

        cqt = librosa.cqt(y, sr=sr, fmin=10, hop_length=64)
        cqt_amplitude = np.abs(cqt)

    librosa.display.specshow(librosa.amplitude_to_db(cqt_amplitude, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('constant-Q power spectrum')
    plt.tight_layout()
    plt.show()
        # plt.plot(x,y)
    # plt.show()




if __name__ == '__main__':
    # idx = 23
    # p = DF_TRAIN_LABEL["path"][idx]
    # label = DF_TRAIN_LABEL["target"][idx]
    # fig, title = showimg(p ,label)
    # p_save = f"./visualize/{title}.jpg"
    # fig.savefig(p_save)

    p = "../input/g2net-gravitational-wave-detection/train/0/0/0/0000bb9f3e.npy"
    myCQT(p)


    # save_imgs(100)