from models.frnet import *


class ObjectCreator:
    def __init__(self, args, cls) -> None:
        self.args = args
        self.cls_net = cls
    def __call__(self):
        return self.cls_net(**self.args)


models = {
    "FRNet-base": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock
    )),
    "FRNet": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock
    )),
    # More models can be added here......
}
