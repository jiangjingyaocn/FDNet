# This code is primarily based on the VST project (https://github.com/nnizhang/VST/tree/main?tab=readme-ov-file).
# Only minor modifications have been made by the FDNet team.
import torch.nn as nn

# from GroupAttention import GroupAttention
from .t2t_vit import T2t_vit_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()


        # VST Encoder
        self.rgb_backbone = T2t_vit_14(pretrained=True, args=args)
        # self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        # self.group_att = GroupAttention(in_dim=64)

        # VST Decoder
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)


    def forward(self, img_input):
        B, _, _, _ = img_input.size()
        # print("img_input.size()",img_input.size())
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(img_input)
        # print("rgb_fea_1_16.shape",rgb_fea_1_16.shape)
        # print("rgb_fea_1_8.shape",rgb_fea_1_8.shape)
        # print("rgb_fea_1_4.shape",rgb_fea_1_4.shape)
        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]
        # group_feature
        # rgb_fea_1_16 = self.group_att(rgb_fea_1_16) + rgb_fea_1_16
        # VST Decoder
        outputs = self.decoder(rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4)

        return outputs
