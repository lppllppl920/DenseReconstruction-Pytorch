import torch
import numpy as np


# Maximum sparse log likelihood loss
class SparseLogProbLoss(torch.nn.Module):
    def __init__(self, epsilon=1.0e-8, gpu_id=0):
        super(SparseLogProbLoss, self).__init__()
        self.epsilon = epsilon
        self.gpu_id = gpu_id
        self.zero = torch.tensor(0.0).float().cuda(self.gpu_id)
        self.one = torch.tensor(1.0).float().cuda(self.gpu_id)
        self.offset = torch.tensor(0.5 * np.log(2.0 * np.pi)).float()

    def forward(self, x):
        mean_depth_maps, std_depth_maps, sparse_depth_maps, binary_sparse_masks = x

        # mean_sparse_depth = torch.sum(binary_sparse_masks * sparse_depth_maps, dim=(1, 2, 3)) / torch.sum(
        #     binary_sparse_masks, dim=(1, 2, 3))
        # std_depth_maps = torch.clamp(std_depth_maps, min=torch.min(mean_sparse_depth * 1.0e-3).item())
        std_depth_maps = torch.clamp(std_depth_maps, min=self.epsilon)

        temp = sparse_depth_maps - mean_depth_maps
        temp_2 = (0.5 * temp ** 2) / std_depth_maps ** 2

        temp_3 = torch.sum(binary_sparse_masks * (
                self.offset.to(mean_depth_maps.device) + torch.log(std_depth_maps) + temp_2), dim=(1, 2, 3))

        loss = temp_3 / (
                self.epsilon + torch.sum(binary_sparse_masks, dim=(1, 2, 3)))
        return torch.mean(loss)


# Maximum dense log likelihood loss
class DenseLogProbLoss(torch.nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(DenseLogProbLoss, self).__init__()
        self.epsilon = epsilon
        self.offset = torch.tensor(0.5 * np.log(2.0 * np.pi)).float()

    def forward(self, x):
        mean_depth_maps, std_depth_maps, warped_mean_depth_maps, intersect_masks = x

        # mean_depth = torch.sum(intersect_masks * (mean_depth_maps + warped_mean_depth_maps), dim=(1, 2, 3)) / torch.sum(
        #     intersect_masks, dim=(1, 2, 3))
        # std_depth_maps = torch.clamp(std_depth_maps, min=torch.min(mean_depth * 1.0e-3).item())
        std_depth_maps = torch.clamp(std_depth_maps, min=self.epsilon)

        temp = warped_mean_depth_maps - mean_depth_maps
        loss = torch.sum(intersect_masks * (self.offset.to(mean_depth_maps.device) + torch.log(
            std_depth_maps) + (0.5 * temp * temp) / (std_depth_maps * std_depth_maps)), dim=(1, 2, 3)) / (
                       self.epsilon + torch.sum(intersect_masks, dim=(1, 2, 3)))

        return torch.mean(loss)


class NormalizedSparseMaskedL1Loss(torch.nn.Module):
    def __init__(self, epsilon=1.0e-8):
        super(NormalizedSparseMaskedL1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        flows, flows_from_depth, sparse_masks = x

        mean_flow_magnitude = torch.sum(sparse_masks * torch.abs(flows), dim=(1, 2, 3)) / torch.sum(
            sparse_masks, dim=(1, 2, 3))
        loss = torch.sum(sparse_masks * torch.abs(flows - flows_from_depth),
                         dim=(1, 2, 3)) / (self.epsilon + mean_flow_magnitude * torch.sum(sparse_masks, dim=(1, 2, 3)))
        return torch.mean(loss)


class RelativeResponseLoss(torch.nn.Module):
    def __init__(self, eps=1.0e-10):
        super(RelativeResponseLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        response_map, source_feature_1d_locations, boundaries = x
        batch_size, sampling_size, height, width = response_map.shape

        response_map = response_map / torch.sum(response_map, dim=(2, 3), keepdim=True)
        # B x Sampling_size x 1
        sampled_cosine_distance = torch.gather(response_map.view(batch_size, sampling_size, height * width), 2,
                                               source_feature_1d_locations.view(batch_size, sampling_size,
                                                                                1).long())

        sampled_boundaries = torch.gather(
            boundaries.view(batch_size, 1, height * width).expand(-1, sampling_size, -1), 2,
            source_feature_1d_locations.view(batch_size, sampling_size,
                                             1).long())
        sampled_boundaries_sum = 1.0 + torch.sum(sampled_boundaries)

        rr_loss = torch.sum(
            sampled_boundaries * -torch.log(self.eps + sampled_cosine_distance)) / sampled_boundaries_sum

        return rr_loss


class MatchingAccuracyMetric(torch.nn.Module):
    def __init__(self, threshold):
        super(MatchingAccuracyMetric, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        response_map, source_feature_2d_locations, boundaries = x
        batch_size, sampling_size, height, width = response_map.shape

        _, detected_target_1d_locations = \
            torch.max(response_map.view(batch_size, sampling_size, height * width), dim=2, keepdim=True)

        detected_target_1d_locations = detected_target_1d_locations.float()
        detected_target_2d_locations = torch.cat(
            [torch.fmod(detected_target_1d_locations, width),
             torch.floor(detected_target_1d_locations / width)],
            dim=2).view(batch_size, sampling_size, 2).float()

        distance = torch.norm(detected_target_2d_locations - source_feature_2d_locations,
                              dim=2, keepdim=False)
        ratio_1 = torch.sum((distance < self.threshold).float()) / (batch_size * sampling_size)
        ratio_2 = torch.sum((distance < 2.0 * self.threshold).float()) / (batch_size * sampling_size)
        ratio_3 = torch.sum((distance < 4.0 * self.threshold).float()) / (batch_size * sampling_size)
        return ratio_1, ratio_2, ratio_3
