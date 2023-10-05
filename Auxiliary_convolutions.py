class AuxiliaryConvolutions(nn.Module):

  #super(AuxiliaryConvolutions, self).__init__()

  def __init__(self, n_classes):

    super(AuxiliaryConvolutions, self).__init__()

    self.n_classes = n_classes
    n_boxes = {'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}

    # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
    #self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
    self.loc_conv7 = nn.Conv2d(512, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
    self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
    self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
    self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
    self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

    # Class prediction convolutions (predict classes in localization boxes)
    #self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
    self.cl_conv7 = nn.Conv2d(512, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
    self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
    self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
    self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
    self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

    self.init_conv2d()

    self.conv5_1 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(1000, 1024, kernel_size = 1, stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(1024)
    )

    self.conv6 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 512, kernel_size = 3, stride = 2),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )

    self.conv7 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(512, 128, kernel_size = 1, stride = 1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv8 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

    self.conv9 = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )

  def init_conv2d(self):
      """
      Initialize convolution parameters.
      """
      for c in self.children():
          #if isinstance(c, nn.Conv2d):
              nn.init.xavier_uniform_(c.weight)
              nn.init.constant_(c.bias, 0.)

  def forward(self, conv7_feats):

      x  = self.conv5_1(conv7_feats)  # 1024 -> 1024

      print(x.shape)
      x = self.conv6(x) # 1024 -> 512
      conv8_2_feats = x
      x = self.conv7(x) # 512 -> 256
      conv9_2_feats = x
      x = self.conv8(x) # 256 -> 256
      conv10_2_feats = x
      x = self.conv8(x) # 256 -> 256
      conv11_2_feats = x

      #aux_out = [conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats]

      batch_size = conv7_feats.size(0)

      #conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = aux_out

      # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
      l_conv7 = self.loc_conv7(conv7_feats)  # (N, 16, 38, 38)
      l_conv7 = l_conv7.permute(0, 2, 3,
                                    1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
      # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
      l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

      '''l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
      l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
      l_conv7 = l_conv7.view(batch_size, -1, 4)'''  # (N, 2166, 4), there are a total 2116 boxes on this feature map

      l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
      l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
      l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

      l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
      l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
      l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

      l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
      l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
      l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

      l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
      l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
      l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

      # Predict classes in localization boxes
      '''c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
      c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                    1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
      c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                  self.n_classes)'''  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

      c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
      c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
      c_conv7 = c_conv7.view(batch_size, -1,
                              self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

      c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
      c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
      c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

      c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
      c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
      c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

      c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
      c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
      c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

      c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
      c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
      c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

      # A total of 8732 boxes
      # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
      locs = torch.cat([l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
      classes_scores = torch.cat([c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                  dim=1)  # (N, 8732, n_classes)

      return locs, classes_scores
