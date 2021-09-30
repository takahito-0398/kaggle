import pandas as pd
from glob import glob
import pandas as pd
import numpy as np
import re
from _00_utils import add_dt


P_TEST = "../input/g2net-gravitational-wave-detection/sample_submission.csv"
def test_a():

    # p_sub_1 = "./submission/submission_ResNeXt101_type5_2epochs.csv"
    p_sub_1 = "./submission/submission_ResNeXt101_3channel_type2_5epochs.csv"
    p_sub_2 = "./submission/submission_ResNet50V2_type5_3epochs.csv"

    df_sub_1 = pd.read_csv(p_sub_1)
    df_sub_2 = pd.read_csv(p_sub_2)

    print(df_sub_1.head())
    print(df_sub_2.head())

    df_submit = df_sub_1.copy()
    df_submit["target"] = (df_sub_1["target"] + df_sub_2["target"]) / 2
    p_submit = "./submission/ensemble_test.csv"

    print(df_submit.head())
    df_submit.to_csv(p_submit, index=False)

def blend(p_submits, p_save, target_col="target"):

    df_submit = pd.read_csv(P_TEST)

    blend_pred = np.empty((len(p_submits), len(df_submit)))

    for i, p in enumerate(p_submits):
        blend_pred[i] = pd.read_csv(p)[target_col].values

    df_submit["target"] = blend_pred.mean(axis=0)

    # os.makedirs(p_save, exist_ok=True)
    p_save = add_dt(p_save)
    df_submit.to_csv(p_save, index=False)
    print(f"submission file has been saved to {p_save}")





if __name__ == '__main__':

    cfg = {
        "p_parent_submit" : "./submission",

        # p_parent_submit以下のファイルのうち、次の正規表現にマッチするものをブレンドの対象にする
        "regex" : "ResNet50V2_type5_(2|3|4)th.*",

        # p_parent_submit以下に次のファイル名のcsvを作成する
        "save_filename" : "ResNet50V2_blend",
    }

    p_parent_submit = cfg["p_parent_submit"]
    regex           = cfg["regex"]
    save_filename   = cfg["save_filename"]


    ps = glob(f"{p_parent_submit}/*.csv")
    p_submits = [p for p in ps if re.search(regex, p)]
    p_save = f"{p_parent_submit}/{save_filename}.csv"
    print(p_submits)

    blend(p_submits, p_save)
