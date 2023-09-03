import torch
import torch.nn as nn

# original code from:
# https://www.youtube.com/watch?v=Grir6TZbc1M&list=RDCMUCkzW5JSFwvKRjXABI-UTAkQ&start_radio=1&rv=Grir6TZbc1M&t=27

# downloade yolov3-pytorch config files:
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

# Tuple: (out_channel, kernel_size, stride)
# List: ["B" for "block", num_repetition_of_block]
config = [
    (32, 3, 1), # (out_channel=32,ks=3,stride=1)
    (64, 3, 2),
    ["B", 1], # ["block", repeate this block once]
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4], # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S", # check image yolov3_arch.png: the 1st scale prediction layer 79-82
    (256, 1, 1),
    "U", # upsampling layer
    (256, 1, 1),
    (512, 3, 1),
    "S", # check image yolov3_arch.png: the 2nd scale prediction layer 91-94
    (128, 1, 1),
    "U", # upsampling layer
    (128, 1, 1),
    (256, 3, 1),
    "S", # check image yolov3_arch.png: the 3rd scale prediction layer 106
]


class CNNBlock(nn.Module):
    """
    nb_act: whether to use batch normalization
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_act = True,
                 **kwargs):
        super().__init__()
        self.use_bn_act = bn_act
        self.conv = nn.Conv2d(in_channels,out_channels,bias=not self.use_bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 num_repeats=1,
                 use_residual=True
                 ):
        super().__init__()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        self.layers = nn.ModuleList()
        for _ in range(self.num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(in_channels=in_channels,
                             out_channels=in_channels//2,
                             kernel_size=1), # reduce filter number
                    CNNBlock(in_channels=in_channels//2,
                             out_channels=in_channels,
                             kernel_size=3, padding=1) # restore filter number
                )
            ]

    def forward(self,x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    """
    Prediction(output) of bounding boxes with different scale
    3*(num_classes+5) in 2nd CNNBlock:
        3: each cell has 3 anchor boxes
        num_classes: how many classes we should classify
        5: 
            prob: there is an object in this cell with probablity prob
            x: x-coordinate of center of bounding box
            y: y-coordinate of center of bounding box
            w: width of bounding box
            h: height of bounding box
    """
    def __init__(self,
                 in_channels,
                 num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            CNNBlock(in_channels=in_channels,
                     out_channels=2*in_channels,
                     kernel_size=3,
                     padding=1),
            CNNBlock(in_channels=2*in_channels,
                     out_channels=3*(self.num_classes+5),
                     bn_act=False,
                     kernel_size=1),
        )
    
    def forward(self, x):
        # e.g. 1st scale prediction layer with gride 13x13
        # [num_sample=N, num_anchor=3, num_grid_x=13, num_grid_y=13, output_dim=5+num_classes]
        return (
            self.pred(x).reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3]).permute(0,1,3,4,2)
        )


class YOLOv3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_conv_layers()

    def forward(self,x):
        # the output of each scale prediction layer
        # i.e. prediction of bounding box with different scale
        output = []
        # the layer which would be added across many layers
        # i.e check architecture image layer 36, layer 61
        route_connections = []

        for layer in self.layers:
            # if it is to output prediction
            if isinstance(layer, ScalePrediction):
                output.append(layer(x))
                continue # go back to forward network
            # if not ouput prediction, then send input and go on
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats==8:
                route_connections.append(x)
            
            elif isinstance(layer, nn.Upsample):
                # concatenate: check architecture image layer 36, layer 61
                x = torch.cat([x, route_connections[-1]], dim=1) # concatenate on dim=1, i.e. dimension of num_anchor
                route_connections.pop()
        return output

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels # 1st in_channels
        for module in config:

            # create CNN block
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size==3 else 0,
                    )
                )
                in_channels = out_channels
            
            # create residual block
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        num_repeats=num_repeats,
                    )
                )
            
            # ScalePrediction or Upsampling layer
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels=in_channels,
                                      num_repeats=1,
                                      use_residual=False),
                        CNNBlock(in_channels=in_channels,
                                 out_channels=in_channels//2,
                                 kernel_size=1),
                        ScalePrediction(in_channels=in_channels//2,
                                        num_classes=self.num_classes)
                    ]
                    in_channels = in_channels//2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 # concatenate 36, 61 and layer after upsampling

        return layers
    

if __name__ == "__main__":
    
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(in_channels=3,num_classes=num_classes)
    # input shape: 2 examples, 3 channels
    x = torch.randn((2,3,IMAGE_SIZE,IMAGE_SIZE))
    out = model(x)
    print("out shape:")
    print(f'out[0]: {out[0].shape}')
    print(f'out[1]: {out[1].shape}')
    print(f'out[2]: {out[2].shape}')
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes+5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes+5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes+5)
    print("Success!")
