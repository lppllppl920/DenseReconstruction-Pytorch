import torch
from torch import nn
import numpy as np
from torch.optim import Optimizer


# Removed dropout and changed the transition up layers in the original implementation
# to mitigate the chessboard patterns of the network output
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))

    def forward(self, x):
        return super(DenseLayer, self).forward(x)


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super(TransitionDown, self).forward(x)


class TransitionUp(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop_(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


def center_crop_(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class FCDenseNetStd(torch.nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48):
        super(FCDenseNetStd, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        self.mean_conv_1 = nn.Conv2d(in_channels=cur_channels_count,
                                     out_channels=128, kernel_size=1, stride=1,
                                     padding=0, bias=True)
        self.mean_conv_2 = nn.Conv2d(in_channels=128,
                                     out_channels=1, kernel_size=1, stride=1,
                                     padding=0, bias=True)
        self.std_conv_1 = nn.Conv2d(in_channels=cur_channels_count,
                                    out_channels=128, kernel_size=1, stride=1,
                                    padding=0, bias=True)
        self.std_conv_2 = nn.Conv2d(in_channels=128,
                                    out_channels=1, kernel_size=1, stride=1,
                                    padding=0, bias=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        # mean_out = self.finalConv(out)

        mean_out = torch.abs(self.mean_conv_2(self.relu_1(self.mean_conv_1(out))))
        std_out = torch.abs(self.std_conv_2(self.relu_2(self.std_conv_1(out))))
        return mean_out, std_out


class DepthMeanStdScalingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8, gpu_id=0, lower_limit=0.1, upper_limit=3.0):
        super(DepthMeanStdScalingLayer, self).__init__()

        self.gpu_id = gpu_id
        self.epsilon = torch.tensor(epsilon).float().cuda(self.gpu_id)
        self.zero = torch.tensor(0.0).float().cuda(self.gpu_id)
        self.one = torch.tensor(1.0).float().cuda(self.gpu_id)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def forward(self, x):
        absolute_mean_depth_estimations, absolute_std_depth_estimations, input_sparse_depths, binary_sparse_masks = x
        # Use sparse depth values which are greater than a certain ratio of the mean value of the sparse depths to avoid
        # instability of scale recovery
        mean_sparse_depths = torch.sum(input_sparse_depths * binary_sparse_masks, dim=(1, 2, 3),
                                       keepdim=True) / torch.sum(binary_sparse_masks, dim=(1, 2, 3),
                                                                 keepdim=True)
        masks = ((input_sparse_depths > self.lower_limit * mean_sparse_depths) & (
                input_sparse_depths < self.upper_limit * mean_sparse_depths)).float()

        sparse_scale_maps = input_sparse_depths * masks / (self.epsilon + absolute_mean_depth_estimations)
        scales = torch.sum(sparse_scale_maps, dim=(1, 2, 3)) / torch.sum(masks, dim=(1, 2, 3))
        return torch.mul(scales.view(-1, 1, 1, 1), absolute_mean_depth_estimations), torch.mul(
            scales.view(-1, 1, 1, 1),
            absolute_std_depth_estimations)


class DepthWarpingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8, gpu_id=0):
        super(DepthWarpingLayer, self).__init__()
        self.gpu_id = gpu_id
        self.epsilon = torch.tensor(epsilon).float().cuda(self.gpu_id)

    def forward(self, x):
        depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        warped_depth_maps, intersect_masks = _depth_warping(depth_maps_1, depth_maps_2, img_masks,
                                                            translation_vectors,
                                                            rotation_matrices, intrinsic_matrices, self.epsilon,
                                                            gpu_id=self.gpu_id)
        return warped_depth_maps, intersect_masks


# Warping depth map in coordinate system 2 to coordinate system 1
def _depth_warping(depth_maps_1, depth_maps_2, img_masks, translation_vectors, rotation_matrices,
                   intrinsic_matrices, epsilon, gpu_id=0):
    # Generate a meshgrid for each depth map to calculate value
    depth_maps_1 = torch.mul(depth_maps_1, img_masks)
    depth_maps_2 = torch.mul(depth_maps_2, img_masks)
    # B x H x W x C
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    depth_maps_2 = depth_maps_2.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)

    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(gpu_id),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda(gpu_id)])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda(gpu_id)

    # intrinsic_matrix_inverse = intrinsic_matrix.inverse()
    eye = torch.eye(3).float().cuda(gpu_id).view(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse, _ = torch.solve(eye, intrinsic_matrices)
    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    # The following is when we have different intrinsic matrices for samples within a batch
    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).view(height, width, 3, 1)
    intermediate_result = torch.matmul(M.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3)

    depth_maps_2_calculate = W.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).view(-1, height,
                                                                   width, 1))

    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.where(img_masks > 0.5, depth_maps_2_calculate, epsilon)
    depth_maps_2_calculate = torch.where(depth_maps_2_calculate > torch.tensor(0.0).float().cuda(gpu_id),
                                         depth_maps_2_calculate, epsilon)

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.view(-1, 3).narrow(dim=-1, start=0, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=0,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / (
              depth_maps_2_calculate)

    v_2 = (W.view(-1, 3).narrow(dim=-1, start=1, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=1,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / (
              depth_maps_2_calculate)

    W_2 = torch.bmm(intrinsic_matrices, translation_vectors)
    M_2 = torch.bmm(torch.bmm(intrinsic_matrices, rotation_matrices), intrinsic_matrices_inverse)

    temp = torch.matmul(M_2.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3).narrow(dim=-1, start=2,
                                                                                               length=1).view(-1,
                                                                                                              height,
                                                                                                              width, 1)
    depth_maps_1_calculate = W_2.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_2, temp)
    depth_maps_1_calculate = torch.mul(img_masks, depth_maps_1_calculate)

    u_2_flat = u_2.view(-1)
    v_2_flat = v_2.view(-1)

    warped_depth_maps_2 = _bilinear_interpolate(depth_maps_1_calculate, u_2_flat,
                                                v_2_flat).view(
        num_batch, 1, height, width)
    intersect_masks = torch.where(
        _bilinear_interpolate(img_masks, u_2_flat, v_2_flat) * img_masks >= 0.9,
        torch.tensor(1.0).float().cuda(gpu_id),
        torch.tensor(0.0).float().cuda(gpu_id)).view(num_batch, 1, height, width)

    return [warped_depth_maps_2, intersect_masks]


def _bilinear_interpolate(im, x, y, padding_mode="zeros"):
    num_batch, height, width, channels = im.shape
    # Range [-1, 1]
    grid = torch.cat([torch.tensor(2.0).float().cuda() *
                      (x.view(num_batch, height, width, 1) / torch.tensor(width).float().cuda())
                      - torch.tensor(1.0).float().cuda(), torch.tensor(2.0).float().cuda() * (
                              y.view(num_batch, height, width, 1) / torch.tensor(height).float().cuda()) - torch.tensor(
        1.0).float().cuda()], dim=-1)

    return torch.nn.functional.grid_sample(input=im.permute(0, 3, 1, 2), grid=grid, mode='bilinear',
                                           padding_mode=padding_mode).permute(0, 2, 3, 1)


# Warp feature map from frame 2 to frame 1
class FeatureWarpingLayer(nn.Module):
    def __init__(self, epsilon=1.0e-8, gpu_id=0):
        super(FeatureWarpingLayer, self).__init__()
        self.gpu_id = gpu_id
        self.epsilon = torch.tensor(epsilon).float().cuda(self.gpu_id)
        self.one = torch.tensor(1.0).float().cuda(self.gpu_id)
        self.zero = torch.tensor(0.0).float().cuda(self.gpu_id)

    def forward(self, x):
        depth_maps_1, feature_maps_2, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x

        num_batch, feature_length, height, width = feature_maps_2.shape
        u_2, v_2 = _flow_coordinate_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                               intrinsic_matrices, gpu_id=self.gpu_id)

        u_2_flat = u_2.reshape(-1)
        v_2_flat = v_2.reshape(-1)

        warped_feature_maps_from_2_to_1 = _bilinear_interpolate(feature_maps_2.permute(0, 2, 3, 1), u_2_flat,
                                                                v_2_flat).reshape(
            num_batch,
            height,
            width,
            feature_length).permute(0, 3, 1, 2)

        intersect_masks = torch.where(
            _bilinear_interpolate(img_masks.permute(0, 2, 3, 1), u_2_flat, v_2_flat) * img_masks.permute(0, 2, 3,
                                                                                                         1) >= 0.9,
            self.one, self.zero).reshape(num_batch, 1, height, width)
        return warped_feature_maps_from_2_to_1, intersect_masks


def _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices,
                              gpu_id=0):
    # Generate a meshgrid for each depth map to calculate value
    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(gpu_id),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda(gpu_id)])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    ones_grid = torch.ones((1, height, width, 1), dtype=torch.float32).cuda(gpu_id)

    # intrinsic_matrix_inverse = intrinsic_matrix.inverse()
    eye = torch.eye(3).float().cuda(gpu_id).view(1, 3, 3).expand(intrinsic_matrices.shape[0], -1, -1)
    intrinsic_matrices_inverse, _ = torch.solve(eye, intrinsic_matrices)

    rotation_matrices_inverse = rotation_matrices.transpose(1, 2)

    # The following is when we have different intrinsic matrices for samples within a batch
    temp_mat = torch.bmm(intrinsic_matrices, rotation_matrices_inverse)
    W = torch.bmm(temp_mat, -translation_vectors)
    M = torch.bmm(temp_mat, intrinsic_matrices_inverse)

    mesh_grid = torch.cat((x_grid, y_grid, ones_grid), dim=-1).view(height, width, 3, 1)
    intermediate_result = torch.matmul(M.view(-1, 1, 1, 3, 3), mesh_grid).view(-1, height, width, 3)

    depth_maps_2_calculate = W.view(-1, 3).narrow(dim=-1, start=2, length=1).view(-1, 1, 1, 1) + torch.mul(
        depth_maps_1,
        intermediate_result.narrow(dim=-1, start=2, length=1).view(-1, height,
                                                                   width, 1))

    # expand operation doesn't allocate new memory (repeat does)
    depth_maps_2_calculate = torch.tensor(1.0e30).float().cuda(gpu_id) * (1.0 - img_masks) + \
                             img_masks * depth_maps_2_calculate

    # This is the source coordinate in coordinate system 2 but ordered in coordinate system 1 in order to warp image 2 to coordinate system 1
    u_2 = (W.view(-1, 3).narrow(dim=-1, start=0, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=0,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / depth_maps_2_calculate

    v_2 = (W.view(-1, 3).narrow(dim=-1, start=1, length=1).view(-1, 1, 1, 1) + torch.mul(depth_maps_1,
                                                                                         intermediate_result.narrow(
                                                                                             dim=-1, start=1,
                                                                                             length=1).view(-1,
                                                                                                            height,
                                                                                                            width,
                                                                                                            1))) / depth_maps_2_calculate
    return [u_2, v_2]


def _flow_coordinate_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices,
                                gpu_id=0):
    # B x H x W x C
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)
    u_2, v_2 = _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                         intrinsic_matrices, gpu_id=gpu_id)

    return u_2, v_2


# dense flow for frame 1 to frame 2
def _flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices, gpu_id=0):
    # B x H x W x C
    depth_maps_1 = depth_maps_1.permute(0, 2, 3, 1)
    img_masks = img_masks.permute(0, 2, 3, 1)
    num_batch, height, width, channels = depth_maps_1.shape

    y_grid, x_grid = torch.meshgrid(
        [torch.arange(start=0, end=height, dtype=torch.float32).cuda(gpu_id),
         torch.arange(start=0, end=width, dtype=torch.float32).cuda(gpu_id)])

    x_grid = x_grid.view(1, height, width, 1)
    y_grid = y_grid.view(1, height, width, 1)

    u_2, v_2 = _warp_coordinate_generate(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                         intrinsic_matrices, gpu_id=gpu_id)

    return torch.cat(
        [(u_2 - x_grid) / float(width), (v_2 - y_grid) / float(height)],
        dim=-1).permute(0, 3, 1, 2)


class FlowfromDepthLayer(nn.Module):
    def __init__(self, gpu_id=0):
        super(FlowfromDepthLayer, self).__init__()
        self.gpu_id = gpu_id

    def forward(self, x):
        depth_maps_1, img_masks, translation_vectors, rotation_matrices, intrinsic_matrices = x
        flow_image = _flow_from_depth(depth_maps_1, img_masks, translation_vectors, rotation_matrices,
                                      intrinsic_matrices, gpu_id=self.gpu_id)
        return flow_image


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, min(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(min(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FCDenseNetFeature(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, feature_length=128,
                 final_convs_filter_base=16):
        super(FCDenseNetFeature, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        self.SEBlocksDown = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
            self.SEBlocksDown.append(SELayer(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################
        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.SEBlocksUp = nn.ModuleList([])
        self.finalConvs = nn.ModuleList([])
        self.factors = []
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.SEBlocksUp.append(SELayer(cur_channels_count))
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            self.finalConvs.append(nn.Conv2d(in_channels=prev_block_channels,
                                             out_channels=final_convs_filter_base, kernel_size=1, stride=1,
                                             padding=0,
                                             bias=True))
            cur_channels_count += prev_block_channels
            self.factors.append(2 ** (len(up_blocks) - 1 - i))

        # Final DenseBlock
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        self.SEBlocksUp.append(SELayer(cur_channels_count))
        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        self.finalConvs.append(nn.Conv2d(in_channels=cur_channels_count,
                                         out_channels=final_convs_filter_base, kernel_size=1, stride=1, padding=0,
                                         bias=True))
        self.factors.append(1.0)

        self.out_conv = nn.Conv2d(in_channels=final_convs_filter_base * len(self.up_blocks),
                                  out_channels=feature_length,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0, bias=True)

    def forward(self, x):
        final_outputs = []
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            out = self.SEBlocksDown[i](out)

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.SEBlocksUp[i](out)
            out = self.denseBlocksUp[i](out)
            final_outputs.append(
                torch.nn.functional.interpolate(self.finalConvs[i](out), scale_factor=self.factors[i],
                                                mode="bilinear"))

        final_out = torch.cat(final_outputs, dim=1)
        final_out = self.out_conv(final_out)
        final_out = final_out / torch.norm(final_out, dim=1, keepdim=True)
        return final_out


class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class FeatureResponseGenerator(nn.Module):
    def __init__(self, scale=20.0, threshold=0.9):
        super(FeatureResponseGenerator, self).__init__()
        self.scale = scale
        self.threshold = threshold

    def forward(self, x):
        source_feature_map, target_feature_map, source_feature_1D_locations, boundaries = x

        # source_feature_map: B x C x H x W
        # source_feature_1D_locations: B x Sampling_size x 1
        batch_size, channel, height, width = source_feature_map.shape
        _, sampling_size, _ = source_feature_1D_locations.shape
        # B x C x Sampling_size
        source_feature_1D_locations = source_feature_1D_locations.view(batch_size, 1,
                                                                       sampling_size).expand(-1, channel, -1)
        # Extend 1D locations to B x C x Sampling_size
        # B x C x Sampling_size
        sampled_feature_vectors = torch.gather(source_feature_map.view(batch_size, channel, height * width), 2,
                                               source_feature_1D_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(batch_size, channel, sampling_size, 1,
                                                               1).permute(0, 2, 1, 3, 4).view(batch_size,
                                                                                              sampling_size,
                                                                                              channel,
                                                                                              1, 1)

        # Do convolution on target_feature_map with the sampled_feature_vectors as the kernels
        # We use the sampled feature vectors in a convolution operation where BC is the input channel dim and
        # Sampling_size as the output channel dim.
        temp = [None for _ in range(batch_size)]
        for i in range(batch_size):
            temp[i] = torch.nn.functional.conv2d(input=target_feature_map[i].view(1, channel, height, width),
                                                 weight=sampled_feature_vectors[i].view(sampling_size, channel,
                                                                                        1,
                                                                                        1),
                                                 padding=0)
        # B x Sampling_size x H x W
        cosine_distance_map = 0.5 * torch.cat(temp, dim=0) + 0.5
        # Normalized cosine distance map
        # B x Sampling_size x H x W
        cosine_distance_map = torch.exp(self.scale * (cosine_distance_map - self.threshold))
        cosine_distance_map = cosine_distance_map / torch.sum(cosine_distance_map, dim=(2, 3), keepdim=True)

        return cosine_distance_map


class FeatureResponseGeneratorNoSoftThresholding(nn.Module):
    def __init__(self):
        super(FeatureResponseGeneratorNoSoftThresholding, self).__init__()

    def forward(self, x):
        source_feature_map, target_feature_map, source_feature_1D_locations, boundaries = x

        # source_feature_map: B x C x H x W
        # source_feature_1D_locations: B x Sampling_size x 1
        batch_size, channel, height, width = source_feature_map.shape
        _, sampling_size, _ = source_feature_1D_locations.shape
        # B x C x Sampling_size
        source_feature_1D_locations = source_feature_1D_locations.view(batch_size, 1,
                                                                       sampling_size).expand(-1, channel, -1)
        # Extend 1D locations to B x C x Sampling_size
        # B x C x Sampling_size
        sampled_feature_vectors = torch.gather(source_feature_map.view(batch_size, channel, height * width), 2,
                                               source_feature_1D_locations.long())
        sampled_feature_vectors = sampled_feature_vectors.view(batch_size, channel, sampling_size, 1,
                                                               1).permute(0, 2, 1, 3, 4).view(batch_size,
                                                                                              sampling_size,
                                                                                              channel,
                                                                                              1, 1)

        # Do convolution on target_feature_map with the sampled_feature_vectors as the kernels
        # We use the sampled feature vectors in a convolution operation where BC is the input channel dim and
        # Sampling_size as the output channel dim.
        temp = [None for _ in range(batch_size)]
        for i in range(batch_size):
            temp[i] = torch.nn.functional.conv2d(input=target_feature_map[i].view(1, channel, height, width),
                                                 weight=sampled_feature_vectors[i].view(sampling_size, channel,
                                                                                        1, 1), padding=0)
        # B x Sampling_size x H x W
        cosine_distance_map = torch.cat(temp, dim=0)
        return cosine_distance_map
