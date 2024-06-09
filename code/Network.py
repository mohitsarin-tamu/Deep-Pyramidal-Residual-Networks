### YOUR CODE HERE
import torch
import torch.nn as nn
import math

"""This script defines the network.
"""

class PyramidNet(nn.Module):
        
    def __init__(self,model_configs, bottleneck=True):
            super(PyramidNet, self).__init__()   
            	
            # intializing the configs
            self.depth = model_configs['depth']
            self.alpha = model_configs['alpha']
            self.num_classes = model_configs['num_classes']

            # considering default values of inplanes as in the original paper
            self.inplanes = 16
            
            if bottleneck == True:
                n = int((self.depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((self.depth - 2) / 6)
                block = BasicBlock

            # using additive Pyramidnet structure
            self.addrate = self.alpha / (3*n*1.0)

            # initialize different layers
            self.input_dim = self.inplanes
            self.conv_layer1 = nn.Conv2d(3, self.input_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.batch_norm1 = nn.BatchNorm2d(self.input_dim)

            # intially feature map dimension same as input dimension
            self.featuremap_dim = self.input_dim 
            self.layer1 = self.build_network(block, n)
            self.layer2 = self.build_network(block, n, stride=2)
            self.layer3 = self.build_network(block, n, stride=2)

            self.output_dim = self.input_dim
            self.batch_norm_last= nn.BatchNorm2d(self.output_dim)
            self.relu_layer = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fully_connected = nn.Linear(self.output_dim, self.num_classes)

    # build the network
    def build_network(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            #print(temp_featuremap_dim)
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        
            x = self.conv_layer1(x)
            x = self.batch_norm1(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.batch_norm_last(x)
            x = self.relu_layer(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fully_connected(x)
    
            return x

    
# performing 3x3 convolution with padding    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# basic block for Pyramid Net
class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # initialize different layers
        self.batch_norm1 = nn.BatchNorm2d(inplanes)
        self.conv_layer1 = conv3x3(inplanes, planes, stride)        
        self.batch_norm2 = nn.BatchNorm2d(planes)
        self.conv_layer2 = conv3x3(planes, planes)
        self.batch_norm3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.batch_norm1(x)
        out = self.conv_layer1(out)        
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv_layer2(out)
        out = self.batch_norm3(out)
       
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

# bottle neck block for Pyramid Net
class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(inplanes)
        self.conv_layer1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(planes)
        self.conv_layer2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d((planes*1))
        self.conv_layer3 = nn.Conv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.batch_norm1(x)
        out = self.conv_layer1(out)
        
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv_layer2(out)
 
        out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.conv_layer3(out)

        out = self.batch_norm4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


### END CODE HERE