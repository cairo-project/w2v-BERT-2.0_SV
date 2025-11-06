import os, sys

sys.path.append("..")
sys.path.append("../../..")
sys.path.append("../../../deeplab/pretrained/audio2vector/module/transformers/src")
sys.path.append("../../../../../audio_toolkit/transforms")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from deeplab.utils.fileio import read_hyperyaml
from pathlib import Path
import torch


class W2VBERT_SPK_Module(torch.nn.Module):
    def __init__(
        self,
        path="recipes/DeepASV/conf/w2v-bert/s3.yaml",
    ):
        super(W2VBERT_SPK_Module, self).__init__()
        self.hparams = read_hyperyaml(path)
        self.modules = self.hparams["modules"]

    def load_model(self, ckpt_path="../pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model_lmft_0.14.pth"):
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key, module in self.modules.items():
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
            self.model = module["spk_model"]

            if mismatched:
                print("      {}: <Partial weights matched>".format(key))
            else:
                print("      {}: <All weights matched>".format(key))
        
    def forward(self, x):
        return self.model(x).float().detach()