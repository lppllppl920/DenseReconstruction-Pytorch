'''
Author: Xingtong Liu, Maia Stiber, Jindan Huang, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import tqdm
import cv2
import numpy as np
from pathlib import Path
import torchsummary
import math
import torch
import random
from tensorboardX import SummaryWriter
import argparse
import datetime
import json
import os

# Local
import models
import losses
import utils
import dataset


def main():
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Probabilistic depth training with dense descriptor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adjacent_range', nargs='+', type=int, required=True,
                        help='frame interval range for sampling two frames')
    parser.add_argument('--image_downsampling', type=float, default=4.0,
                        help='image downsampling rate to speed up training and reduce overfitting')
    parser.add_argument('--network_downsampling', type=int, default=64,
                        help='network downsampling rate from input to bottleneck layer')
    parser.add_argument('--input_size', nargs='+', type=int, default=None,
                        help='input size for architecture summary')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
    parser.add_argument('--slp_weight', type=float, default=1.0, help='weight for sparse log prob loss')
    parser.add_argument('--dcl_weight', type=float, default=0.5, help='weight for depth consistency loss')
    parser.add_argument('--sfl_weight', type=float, default=100.0, help='weight for sparse flow loss')
    parser.add_argument('--dl_weight', type=float, default=20.0, help='weight for descriptor loss')
    parser.add_argument('--lr_range', nargs='+', type=float, default=[1.0e-4, 1.0e-3],
                        help='cyclic lr range (min, max)')
    parser.add_argument('--inlier_percentage', type=float, default=0.998,
                        help='percentage of inliers of SfM point clouds (to prune some outliers)')
    parser.add_argument('--display_interval', type=int, default=50, help='iteration interval for image display')
    parser.add_argument('--save_interval', type=int, default=2, help='interval for saving model')
    parser.add_argument('--visible_interval', type=int, default=5,
                        help='range for propagating point visibility information')
    parser.add_argument('--training_patient_id', nargs='+', type=int, required=True, help='id of the training patients')
    parser.add_argument('--load_intermediate_data', action='store_true',
                        help='whether or not to load pre-compute data')
    parser.add_argument('--load_trained_model', action='store_true',
                        help='whether to load pre-trained model')
    parser.add_argument('--trained_model_path', type=str, default=None, help='path to the trained model')
    parser.add_argument('--num_epoch', type=int, required=True, help='number of epochs in total')
    parser.add_argument('--architecture_summary', action='store_true', help='summarize the network architecture')
    parser.add_argument('--data_root', type=str, required=True, help='path to the training data')
    parser.add_argument('--log_root', type=str, required=True, help='root of the training logs')
    parser.add_argument('--precompute_root', type=str, required=True, help='root of the pre-compute data')
    parser.add_argument('--num_iter', type=int, default=1000,
                        help='number of iterations per epoch')
    parser.add_argument('--descriptor_model_path', type=str, required=True,
                        help='path to the trained feature matching model')

    args = parser.parse_args()

    # Fix randomness for reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(10085)
    np.random.seed(10085)
    random.seed(10085)

    date = datetime.datetime.now()
    log_root = Path(args.log_root) / "depth_train_{}_{}_{}_{}".format(
        date.month,
        date.day,
        date.hour,
        date.minute)
    if not log_root.exists():
        log_root.mkdir(parents=True)

    if not Path(args.precompute_root).exists():
        Path(args.precompute_root).mkdir(parents=True)

    with open(str(log_root / 'commandline_args'), 'w') as f:
        f.write("script: {}".format(str(os.path.realpath(__file__))))
        json.dump(args.__dict__, f, indent=2)

    writer = SummaryWriter(logdir=str(log_root))
    print("Tensorboard visualization at {}".format(str(log_root)))

    # Get color image filenames
    train_filenames = utils.get_color_file_names_by_bag(Path(args.data_root),
                                                        id_list=args.training_patient_id)
    folder_list = utils.get_parent_folder_names(Path(args.data_root), id_list=args.training_patient_id)

    # Build training and validation dataset
    train_dataset = dataset.DepthDataset(image_file_names=train_filenames,
                                         folder_list=folder_list,
                                         adjacent_range=args.adjacent_range,
                                         image_downsampling=args.image_downsampling,
                                         network_downsampling=args.network_downsampling,
                                         inlier_percentage=args.inlier_percentage,
                                         load_intermediate_data=args.load_intermediate_data,
                                         intermediate_data_root=Path(args.precompute_root),
                                         visible_interval=args.visible_interval,
                                         num_pre_workers=args.num_workers,
                                         num_iter=args.num_iter)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    depth_estimation_model = models.FCDenseNetStd(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48)
    # Initialize the depth estimation network with Kaiming He initialization
    utils.init_net(depth_estimation_model, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")

    # Multi-GPU running
    depth_estimation_model = torch.nn.DataParallel(depth_estimation_model)
    # Summary network architecture
    if args.architecture_summary:
        torchsummary.summary(depth_estimation_model, input_size=(3, args.input_size[0], args.input_size[1]))
    # Optimizer
    optimizer = torch.optim.SGD(depth_estimation_model.parameters(), lr=args.lr_range[1], momentum=0.9)
    lr_scheduler = models.CyclicLR(optimizer, base_lr=args.lr_range[0], max_lr=args.lr_range[1])

    # Custom layers
    depth_scaling_layer = models.DepthMeanStdScalingLayer()
    depth_warping_layer = models.DepthWarpingLayer()
    feature_warping_layer = models.FeatureWarpingLayer()
    flow_from_depth_layer = models.FlowfromDepthLayer()
    # Loss functions
    sparse_log_prob_loss = losses.SparseLogProbLoss(epsilon=1.0e-4)
    dense_log_prob_loss = losses.DenseLogProbLoss()
    sparse_masked_l1_loss = losses.NormalizedSparseMaskedL1Loss()

    # Load previous student model and so on
    if args.load_trained_model:
        if Path(args.trained_model_path).exists():
            print("Loading {:s} ...".format(str(args.trained_model_path)))
            state = torch.load(str(args.trained_model_path), encoding="latin1")
            step = state['step']
            epoch = state['epoch']
            depth_estimation_model.load_state_dict(state["model"])

            print('Restored model, epoch {}, step {}'.format(epoch, step))
        else:
            print("No pre-trained model detected")
            raise OSError
    else:
        epoch = 0
        step = 0

    descriptor_model = models.FCDenseNetFeature(
        in_channels=3, down_blocks=(3, 3, 3, 3, 3),
        up_blocks=(3, 3, 3, 3, 3), bottleneck_layers=4,
        growth_rate=8, out_chans_first_conv=16, feature_length=256, final_convs_filter_base=128)
    # Initialize the depth estimation network with Kaiming He initialization
    descriptor_model = utils.init_net(descriptor_model, type="kaiming", mode="fan_in",
                                      activation_mode="relu",
                                      distribution="normal")
    # Multi-GPU running
    descriptor_model = torch.nn.DataParallel(descriptor_model)
    descriptor_model.eval()

    if Path(args.descriptor_model_path).exists():
        print("Loading {:s} ...".format(str(args.descriptor_model_path)))
        state = torch.load(str(args.descriptor_model_path))
        descriptor_model.load_state_dict(state["model"])
    else:
        print("No pre-trained descriptor model detected")
        raise OSError

    for cur_epoch in range(epoch, args.num_epoch + 1):
        # Set the seed correlated to cur_epoch for reproducibility
        torch.manual_seed(10086 + cur_epoch)
        np.random.seed(10086 + cur_epoch)
        random.seed(10086 + cur_epoch)
        depth_estimation_model.train()

        if cur_epoch <= 10:
            dcl_weight = 0.0
            dl_weight = 0.0
        else:
            dcl_weight = args.dcl_weight
            dl_weight = args.dl_weight

        # Update progress bar
        tq = tqdm.tqdm(total=len(train_loader) * args.batch_size)
        for batch, (
                colors_1, colors_2, sparse_depths_1, sparse_depths_2,
                sparse_depth_masks_1, sparse_depth_masks_2,
                sparse_flows_1, sparse_flows_2, sparse_flow_masks_1, sparse_flow_masks_2, boundaries, shrink_boundaries,
                rotations_1_wrt_2,
                rotations_2_wrt_1, translations_1_wrt_2, translations_2_wrt_1, intrinsics, folders, file_names) in \
                enumerate(train_loader):

            # Update learning rate
            lr_scheduler.batch_step(batch_iteration=step)
            tq.set_description('Epoch {}, lr {}'.format(cur_epoch, lr_scheduler.get_lr()))

            with torch.no_grad():
                colors_1 = colors_1.cuda()
                colors_2 = colors_2.cuda()
                sparse_depths_1 = sparse_depths_1.cuda()
                sparse_depths_2 = sparse_depths_2.cuda()
                sparse_depth_masks_1 = sparse_depth_masks_1.cuda()
                sparse_depth_masks_2 = sparse_depth_masks_2.cuda()
                sparse_flows_1 = sparse_flows_1.cuda()
                sparse_flows_2 = sparse_flows_2.cuda()
                sparse_flow_masks_1 = sparse_flow_masks_1.cuda()
                sparse_flow_masks_2 = sparse_flow_masks_2.cuda()
                boundaries = boundaries.cuda()
                shrink_boundaries = shrink_boundaries.cuda()
                rotations_1_wrt_2 = rotations_1_wrt_2.cuda()
                rotations_2_wrt_1 = rotations_2_wrt_1.cuda()
                translations_1_wrt_2 = translations_1_wrt_2.cuda()
                translations_2_wrt_1 = translations_2_wrt_1.cuda()
                intrinsics = intrinsics.cuda()

                colors_1 = boundaries * colors_1
                colors_2 = boundaries * colors_2
                # Generate dense descriptor map
                feature_maps_1 = descriptor_model(colors_1)
                feature_maps_2 = descriptor_model(colors_2)
                feature_maps_1 = boundaries * feature_maps_1
                feature_maps_2 = boundaries * feature_maps_2

            # Predicted depth
            predicted_mean_depth_maps_1, predicted_std_depth_maps_1 = depth_estimation_model(colors_1)
            predicted_mean_depth_maps_2, predicted_std_depth_maps_2 = depth_estimation_model(colors_2)

            scaled_mean_depth_maps_1, scaled_std_depth_maps_1 = depth_scaling_layer(
                [predicted_mean_depth_maps_1, predicted_std_depth_maps_1,
                 sparse_depths_1, sparse_depth_masks_1])
            scaled_mean_depth_maps_2, scaled_std_depth_maps_2 = depth_scaling_layer(
                [predicted_mean_depth_maps_2, predicted_std_depth_maps_2,
                 sparse_depths_2, sparse_depth_masks_2])

            # Feature consistency loss
            warped_feature_maps_2_to_1, shrink_intersect_masks_1 = feature_warping_layer(
                [scaled_mean_depth_maps_1, feature_maps_2, shrink_boundaries, translations_1_wrt_2,
                 rotations_1_wrt_2, intrinsics])
            warped_feature_maps_1_to_2, shrink_intersect_masks_2 = feature_warping_layer(
                [scaled_mean_depth_maps_2, feature_maps_1, shrink_boundaries, translations_2_wrt_1,
                 rotations_2_wrt_1, intrinsics])

            descriptor_loss_1 = torch.mean((torch.sum(
                shrink_intersect_masks_1 * torch.abs(feature_maps_1 - warped_feature_maps_2_to_1), dim=(1, 2, 3))) / (
                                                   torch.sum(shrink_intersect_masks_1, dim=(1, 2, 3)) + 1.0e-8))
            descriptor_loss_2 = torch.mean((torch.sum(
                shrink_intersect_masks_2 * torch.abs(feature_maps_2 - warped_feature_maps_1_to_2), dim=(1, 2, 3))) / (
                                                   torch.sum(shrink_intersect_masks_2, dim=(1, 2, 3)) + 1.0e-8))
            dl_loss = dl_weight * (0.5 * descriptor_loss_1 + 0.5 * descriptor_loss_2)

            # Sparse log prob loss
            sd_loss = args.slp_weight * (0.5 * sparse_log_prob_loss([scaled_mean_depth_maps_1, scaled_std_depth_maps_1,
                                                                     sparse_depths_1, sparse_depth_masks_1]) + 0.5 *
                                         sparse_log_prob_loss([scaled_mean_depth_maps_2, scaled_std_depth_maps_2,
                                                               sparse_depths_2, sparse_depth_masks_2]))

            warped_mean_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                [scaled_mean_depth_maps_1, scaled_mean_depth_maps_2, boundaries, translations_1_wrt_2,
                 rotations_1_wrt_2,
                 intrinsics])
            warped_mean_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                [scaled_mean_depth_maps_2, scaled_mean_depth_maps_1, boundaries, translations_2_wrt_1,
                 rotations_2_wrt_1,
                 intrinsics])
            # Depth consistency loss
            dc_loss = dcl_weight * (0.5 * dense_log_prob_loss([scaled_mean_depth_maps_1, scaled_std_depth_maps_1,
                                                               warped_mean_depth_maps_2_to_1, intersect_masks_1]) +
                                    0.5 * dense_log_prob_loss([scaled_mean_depth_maps_2, scaled_std_depth_maps_2,
                                                               warped_mean_depth_maps_1_to_2, intersect_masks_2]))

            flows_from_depth_1 = flow_from_depth_layer(
                [scaled_mean_depth_maps_1, boundaries, translations_1_wrt_2, rotations_1_wrt_2,
                 intrinsics])
            flows_from_depth_2 = flow_from_depth_layer(
                [scaled_mean_depth_maps_2, boundaries, translations_2_wrt_1, rotations_2_wrt_1,
                 intrinsics])

            sparse_flow_masks_1 = sparse_flow_masks_1 * boundaries
            sparse_flow_masks_2 = sparse_flow_masks_2 * boundaries
            sparse_flows_1 = sparse_flows_1 * boundaries
            sparse_flows_2 = sparse_flows_2 * boundaries
            flows_from_depth_1 = flows_from_depth_1 * boundaries
            flows_from_depth_2 = flows_from_depth_2 * boundaries

            sf_loss = args.sfl_weight * 0.5 * (sparse_masked_l1_loss(
                [sparse_flows_1, flows_from_depth_1, sparse_flow_masks_1]) + sparse_masked_l1_loss(
                [sparse_flows_2, flows_from_depth_2, sparse_flow_masks_2]))

            # Overall Loss
            loss = sd_loss + dc_loss + sf_loss + dl_loss

            # Display depth and color at TensorboardX
            if batch % args.display_interval == 0:
                display_list_1 = \
                    utils.display_color_mean_std_depth_sparse_flow_dense_flow(colors_1,
                                                                              scaled_mean_depth_maps_1 * boundaries,
                                                                              scaled_std_depth_maps_1 * boundaries,
                                                                              sparse_flows_1,
                                                                              flows_from_depth_1)
                display_list_2 = \
                    utils.display_color_mean_std_depth_sparse_flow_dense_flow(colors_2,
                                                                              scaled_mean_depth_maps_2 * boundaries,
                                                                              scaled_std_depth_maps_2 * boundaries,
                                                                              sparse_flows_2,
                                                                              flows_from_depth_2)
                utils.stack_and_display(phase="Train",
                                        title="Results (c1, d1, sd1, sf1, df1, c2, d2, sd2, sf2, df2)",
                                        step=step, writer=writer,
                                        image_list=display_list_1 + display_list_2)

            # Handle nan/inf cases
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                optimizer.zero_grad()
                loss.backward()
                optimizer.zero_grad()
                continue
            else:
                optimizer.zero_grad()
                loss.backward()
                # Prevent one sample from having too much impact on the training
                torch.nn.utils.clip_grad_norm_(depth_estimation_model.parameters(), 10.0)
                optimizer.step()
                if batch == 0:
                    mean_loss = loss.item()
                    mean_sd_loss = sd_loss.item()
                    mean_dc_loss = dc_loss.item()
                    mean_sf_loss = sf_loss.item()
                    mean_dl_loss = dl_loss.item()
                else:
                    mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
                    mean_sd_loss = (mean_sd_loss * batch + sd_loss.item()) / (batch + 1.0)
                    mean_dc_loss = (mean_dc_loss * batch + dc_loss.item()) / (batch + 1.0)
                    mean_sf_loss = (mean_sf_loss * batch + sf_loss.item()) / (batch + 1.0)
                    mean_dl_loss = (mean_dl_loss * batch + dl_loss.item()) / (batch + 1.0)

            step += 1
            tq.update(colors_1.shape[0])
            tq.set_postfix(loss='avg: {:.3f}, cur: {:.3f}'.format(mean_loss, loss.item()),
                           sd_loss='avg: {:.3f}, cur: {:.3f}'.format(mean_sd_loss,
                                                                     sd_loss.item()),
                           dc_loss='avg: {:.3f}, cur: {:.3f}'.format(mean_dc_loss,
                                                                     dc_loss.item()),
                           sf_loss='avg: {:.3f}, cur: {:.3f}'.format(mean_sf_loss,
                                                                     sf_loss.item()),
                           dl_loss='avg: {:.3f}, cur: {:.3f}'.format(mean_dl_loss,
                                                                     dl_loss.item())
                           )
            # TensorboardX
            writer.add_scalars('Training', {'overall': mean_loss,
                                            'slp loss': mean_sd_loss,
                                            'dcl loss': mean_dc_loss,
                                            'sfl loss': mean_sf_loss,
                                            'dl loss': mean_dl_loss
                                            }, step)

        tq.close()

        if cur_epoch % args.save_interval == 0:
            writer.export_scalars_to_json(
                str(log_root / ("all_scalars_" + str(cur_epoch) + ".json")))
            model_path_epoch = log_root / 'checkpoint_model_epoch_{}_{}_{}_{}_{}_{}.pt'.format(cur_epoch,
                                                                                               mean_loss,
                                                                                               mean_sd_loss,
                                                                                               mean_dc_loss,
                                                                                               mean_sf_loss,
                                                                                               mean_dl_loss)
            utils.save_model(model=depth_estimation_model, optimizer=optimizer,
                             epoch=cur_epoch + 1, step=step, model_path=model_path_epoch,
                             validation_loss=mean_sf_loss)

    writer.close()


if __name__ == '__main__':
    main()
