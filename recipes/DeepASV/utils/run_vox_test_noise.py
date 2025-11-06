import os, sys

sys.path.append("..")
sys.path.append("../../..")
sys.path.append("../../../deeplab/pretrained/audio2vector/module/transformers/src")
sys.path.append("../../../../../audio_toolkit/transforms")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
from deeplab.utils.fileio import read_hyperyaml, load_audio
from pathlib import Path
import torch
from sklearn.metrics import roc_curve, auc
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np
from torch_audiomentations import OneOf
from mix import ExternalMix

from calflops import calculate_flops


def calc_model_flops(model, *args, **kwargs):
    """
    Measure FLOPs, MACs, and Params of a PyTorch model using calflops.
    Works even if args is a tuple (auto-converted to list).
    """
    args_list = list(args)  # calflops expects a mutable sequence

    flops, macs, params = calculate_flops(
        model=model,
        args=args_list,
        kwargs=kwargs,
        output_as_string=True,
        output_precision=4,
    )
    return flops, macs, params


def get_eer(y, y_pred, name, save_roc_curve=True):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    if save_roc_curve:
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure()
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot(
            [0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random guess"
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig(f"{name}_ROC.png")  # You can also use .jpg, .pdf, .svg, etc.

    return eer, eer_threshold


def plot_heatmap(score_matrix, name):
    labels = sorted(
        set([i for i, j in score_matrix.keys()] + [j for i, j in score_matrix.keys()])
    )

    # Build DataFrame
    matrix = pd.DataFrame(index=labels, columns=labels)
    for (i, j), val in score_matrix.items():
        matrix.loc[i, j] = val
    matrix = matrix.astype(float)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, cmap="Blues", cbar=True)
    plt.title("Similarity Score Heatmap")
    plt.xlabel("Predicted / Column Class")
    plt.ylabel("True / Row Class")
    plt.savefig(f"{name}_heatmap.png")  # You can also use .jpg, .pdf, .svg, etc.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speaker verification testing with optional noise"
    )
    parser.add_argument(
        "--enable_noise", action="store_true", help="Enable noise augmentation"
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=20.0,
        help="Signal-to-noise ratio in dB (default: 20.0)",
    )
    parser.add_argument(
        "--test_pairs",
        type=str,
        default="/home/dino/Documents/SpeakerVerification/data/voxceleb1/test/test_pairs.txt",
        help="The test pairs file path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/dino/Documents/SpeakerVerification/data/voxceleb1/test",
        help="Directory containing test data",
    )
    return parser.parse_args()


args = parse_args()
snr = args.snr
apply_augmentation = OneOf(
    transforms=[
        ExternalMix(
            external_noise_set="/home/dino/Documents/SpeakerVerification/data/DEMAND/STRAFFIC_48k",
            min_snr_in_db=snr,
            max_snr_in_db=snr + 0.01,
            p=1.0,
        )
    ],
    output_type="dict",
)


ckpt_path = "/home/dino/Documents/SpeakerVerification/git/test_sv_repos/w2v-BERT-2.0_SV/model_lmft_0.14.pth"


ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

hparams = read_hyperyaml(
    path="/home/dino/Documents/SpeakerVerification/git/test_sv_repos/w2v-BERT-2.0_SV/recipes/DeepASV/conf/w2v-bert/s3.yaml"
)
modules = hparams["modules"]

for key, module in modules.items():
    if key == "classifier":
        continue
    curr_state_dict = module.state_dict()
    ckpt_state_dict = ckpt_data["modules"][key]
    mismatched = False
    for k in curr_state_dict.keys():
        if (
            k in ckpt_state_dict
            and curr_state_dict[k].shape == ckpt_state_dict[k].shape
        ):
            curr_state_dict[k] = ckpt_state_dict[k]
        else:
            mismatched = True
    module.load_state_dict(curr_state_dict)
    module = module.eval().cuda()

    if mismatched:
        print("      {}: <Partial weights matched>".format(key))
    else:
        print("      {}: <All weights matched>".format(key))


data_dir = args.data_dir
sr = hparams["sample_rate"]
max_len = int(hparams["max_valid_dur"] * hparams["sample_rate"])

os.listdir(data_dir)
data = {}

cross_refernce = {}
subjects = list(data.keys())
score_fn = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

y = []
y_pred = []

with open(args.test_pairs, "r") as f:
    lines = f.readlines()
test_pairs = [line.strip().split() for line in lines]
limit = len(test_pairs)

for i, (match, f1, f2) in tqdm(enumerate(test_pairs)):
    subject1 = f1.split("/")[0]
    subject2 = f2.split("/")[0]

    if subject1 not in data:
        data[subject1] = {}
    if f1 not in data[subject1]:

        with torch.no_grad():
            signal = torch.from_numpy(
                load_audio(f"{data_dir}/{f1}", sr)[0][:max_len]
            ).float()
            if args.enable_noise:
                signal = apply_augmentation(
                    signal.view(1, 1, -1), sample_rate=sr
                ).squeeze()
            data[subject1][f"{data_dir}/{f1}"] = (
                modules["spk_model"](signal).float().detach()
            )
    if subject2 not in data:
        data[subject2] = {}
    if f2 not in data[subject2]:

        with torch.no_grad():
            signal = torch.from_numpy(
                load_audio(f"{data_dir}/{f2}", sr)[0][:max_len]
            ).float()
            if args.enable_noise:
                signal = apply_augmentation(
                    signal.view(1, 1, -1), sample_rate=sr
                ).squeeze()
            data[subject2][f"{data_dir}/{f2}"] = modules["spk_model"](signal).float()

    embeddings1 = data[subject1][f"{data_dir}/{f1}"]
    embeddings2 = data[subject2][f"{data_dir}/{f2}"]
    score = score_fn(embeddings1, embeddings2)

    y.append(int(match))
    y_pred.append(score.item())

    cross_refernce[(subject1, subject2)] = score.item()

    if i == limit:
        break

if args.enable_noise:
    name = f"w2v_bert_sv_pretrained_snr{snr}"
else:
    name = f"w2v_bert_sv_pretrained_clean"
plot_heatmap(cross_refernce, name)
eer, threshold = get_eer(y, y_pred, name)
print(f"EER: {eer*100:.2f}%")
