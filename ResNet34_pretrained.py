class ResNet34(nn.Module):

  def __init__(self):
    super(ResNet34, self).__init__() #метод супер позволяет использовать функции класса родителя, в скобках - функции которые хотим перенести

    self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2)

    self.conv2_1 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    self.conv2_2 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )
    self.conv2_3 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    self.conv3_1 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )

    self.upsample1 = nn.Sequential(

        nn.Conv2d(64, 128, kernel_size = (1, 1), stride = (2, 2), bias = False),
        nn.BatchNorm2d(128)

    )

    self.conv3_2 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )
    self.conv3_3 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )
    self.conv3_4 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )

    self.conv4_1 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.upsample2 = nn.Sequential(

        nn.Conv2d(128, 256, kernel_size = (1, 1), stride = (2, 2), bias = False),
        nn.BatchNorm2d(256)

    )

    self.conv4_2 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv4_3 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv4_4 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv4_5 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv4_6 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv5_1 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )

    self.upsample3 = nn.Sequential(

        nn.Conv2d(256, 512, kernel_size = (1, 1), stride = (2, 2), bias = False),
        nn.BatchNorm2d(512)

    )

    self.conv5_2 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )

    self.conv5_3 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )


    self.avgpool = nn.AvgPool2d(kernel_size = 3, stride=None, padding=1, ceil_mode=False, count_include_pad=True, divisor_override=None)

# we extract fully connected layers that do prediction

    '''self.FC1000 = nn.Sequential(
        nn.Linear(512, 512)
    )'''

    # Load pretrained layers
    self.load_pretrained_layers()


  def forward(self, x):

      x = self.conv1(x)
      x = self.pool1(x)

      x = self.bn1(x)
      indices1 = x

      x = self.conv2_1(x)
      indices2 = x

      x = self.conv2_2(x + indices1)
      indices3 = x

      x = self.conv2_3(x + indices2)
      indices4 = x

      x = self.conv3_1(x + indices3)
      indices5 = x

      indices4 = self.upsample1(indices4)

      x = self.conv3_2(x + indices4)
      indices6 = x

      x = self.conv3_3(x + indices5)
      indices7 = x

      x = self.conv3_4(x + indices6)
      indices8 = x

      x = self.conv4_1(x, indices7)
      indices9 = x

      indices8 = self.upsample2(indices8)

      x = self.conv4_2(x + indices8)
      indices10 = x
      x = self.conv4_3(x + indices9)
      indices11 = x
      x = self.conv4_4(x + indices10)
      indices12 = x
      x = self.conv4_5(x + indices11)
      indices13 = x
      x = self.conv4_6(x + indices12)
      indices14 = x

      indices14 = self.upsample3(indices14)

      x = self.conv5_1(x + indices13)
      indices15 = x
      x = self.conv5_2(x + indices14)
      indices16 = x
      x = self.conv5_3(x + indices15)

      x = self.avgpool(x + indices16)
      conv7_feats = x

      print(x.shape)
      print(x)


      return conv7_feats


    def decimate(tensor, m):
        """
        Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
        This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
        :param tensor: tensor to be decimated
        :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
        :return: decimated tensor
        """
        assert tensor.dim() == len(m)
        for d in range(tensor.dim()):
            if m[d] is not None:
                tensor = tensor.index_select(dim=d,
                                            index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

        return tensor


  def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained resnet base
        pretrained_state_dict = torchvision.models.resnet34(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        print(pretrained_param_names)

        for param_tensor in pretrained_state_dict:
            print(param_tensor, "\t", pretrained_state_dict[param_tensor].size())

        for param in state_dict:
            print(param, "\t", state_dict[param].size())

        #print(pretrained_param_names)

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-2]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
# 'fc.weight', 'fc.bias'

        '''conv_fc5_weight = pretrained_state_dict['fc.weight']#.view(1024, 512, 3, 3)  # (4096, 512, 7, 7)
        (print(conv_fc5_weight.shape))
        conv_fc5_bias = pretrained_state_dict['fc.bias']#.view(1024)  # (1000)
        state_dict['fc.weight'] = decimate(conv_fc5_weight, m=[500, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['fc.bias'] = decimate(conv_fc5_bias, m=[500])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)
'''
        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")
        return pretrained_param_names
