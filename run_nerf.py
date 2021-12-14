import os
import time
import torch
import imageio
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from utils.flow_utils import flow_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300000,
                        help='exponential learning rate decay')
    parser.add_argument("--chunk", type=int, default=1024*128,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=1,
                        help='fix random seed for repeatability')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_viewdirsDyn", action='store_true',
                        help='use full 5D input instead of 3D for D-NeRF')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--N_iters", type=int, default=1000000,
                        help='number of training iterations')

    # Dynamic NeRF lambdas
    parser.add_argument("--dynamic_loss_lambda", type=float, default=1.,
                        help='lambda of dynamic loss')
    parser.add_argument("--static_loss_lambda", type=float, default=1.,
                        help='lambda of static loss')
    parser.add_argument("--full_loss_lambda", type=float, default=3.,
                        help='lambda of full loss')
    parser.add_argument("--depth_loss_lambda", type=float, default=0.04,
                        help='lambda of depth loss')
    parser.add_argument("--order_loss_lambda", type=float, default=0.1,
                        help='lambda of order loss')
    parser.add_argument("--flow_loss_lambda", type=float, default=0.02,
                        help='lambda of optical flow loss')
    parser.add_argument("--slow_loss_lambda", type=float, default=0.1,
                        help='lambda of sf slow regularization')
    parser.add_argument("--smooth_loss_lambda", type=float, default=0.1,
                        help='lambda of sf smooth regularization')
    parser.add_argument("--consistency_loss_lambda", type=float, default=0.1,
                        help='lambda of sf cycle consistency regularization')
    parser.add_argument("--mask_loss_lambda", type=float, default=0.1,
                        help='lambda of the mask loss')
    parser.add_argument("--sparse_loss_lambda", type=float, default=0.1,
                        help='lambda of sparse loss')
    parser.add_argument("--DyNeRF_blending", action='store_true',
                        help='use Dynamic NeRF to predict blending weight')
    parser.add_argument("--pretrain", action='store_true',
                        help='Pretrain the StaticneRF')
    parser.add_argument("--ft_path_S", type=str, default=None,
                        help='specific weights npy file to reload for StaticNeRF')

    # For rendering teasers
    parser.add_argument("--frame2dolly", type=int, default=-1,
                        help='choose frame to perform dolly zoom')
    parser.add_argument("--x_trans_multiplier", type=float, default=1.,
                        help='x_trans_multiplier')
    parser.add_argument("--y_trans_multiplier", type=float, default=0.33,
                        help='y_trans_multiplier')
    parser.add_argument("--z_trans_multiplier", type=float, default=5.,
                        help='z_trans_multiplier')
    parser.add_argument("--num_novelviews", type=int, default=60,
                        help='num_novelviews')
    parser.add_argument("--focal_decrease", type=float, default=200,
                        help='focal_decrease')
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)

    # Load data
    if args.dataset_type == 'llff':
        frame2dolly = args.frame2dolly
        images, invdepths, masks, poses, bds, \
        render_poses, render_focals, grids = load_llff_data(args, args.datadir,
                                                            args.factor,
                                                            frame2dolly=frame2dolly,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        num_img = float(poses.shape[0])
        assert len(poses) == len(images)
        print('Loaded llff', images.shape,
            render_poses.shape, hwf, args.datadir)

        # Use all views to train
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            raise NotImplementedError
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    if not args.render_only:
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
        'num_img': num_img,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        i = start - 1

        # Change time and change view at the same time.
        time2render = np.concatenate((np.repeat((i_train / float(num_img) * 2. - 1.0), 4),
                                      np.repeat((i_train / float(num_img) * 2. - 1.0)[::-1][1:-1], 4)))
        if len(time2render) > len(render_poses):
            pose2render = np.tile(render_poses, (int(np.ceil(len(time2render) / len(render_poses))), 1, 1))
            pose2render = pose2render[:len(time2render)]
            pose2render = torch.Tensor(pose2render)
        else:
            time2render = np.tile(time2render, int(np.ceil(len(render_poses) / len(time2render))))
            time2render = time2render[:len(render_poses)]
            pose2render = torch.Tensor(render_poses)
        result_type = 'novelviewtime'

        testsavedir = os.path.join(
            basedir, expname, result_type + '_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            ret = render_path(pose2render, time2render,
                              hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
        moviebase = os.path.join(
            testsavedir, '{}_{}_{:06d}_'.format(expname, result_type, i))
        save_res(moviebase, ret)

        # Fix view (first view) and change time.
        pose2render = torch.Tensor(poses[0:1, ...]).expand([int(num_img), 3, 4])
        time2render = i_train / float(num_img) * 2. - 1.0
        result_type = 'testset_view000'

        testsavedir = os.path.join(
            basedir, expname, result_type + '_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            ret = render_path(pose2render, time2render,
                              hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
        moviebase = os.path.join(
            testsavedir, '{}_{}_{:06d}_'.format(expname, result_type, i))
        save_res(moviebase, ret)

        return

    N_rand = args.N_rand

    # Move training data to GPU
    images = torch.Tensor(images)
    invdepths = torch.Tensor(invdepths)
    masks = 1.0 - torch.Tensor(masks)
    poses = torch.Tensor(poses)
    grids = torch.Tensor(grids)

    print('Begin')
    print('TRAIN views are', i_train)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    decay_iteration = max(25, num_img)

    # Pre-train StaticNeRF
    if args.pretrain:
        render_kwargs_train.update({'pretrain': True})

        # Pre-train StaticNeRF first and use DynamicNeRF to blend
        assert args.DyNeRF_blending == True

        if args.ft_path_S is not None and args.ft_path_S != 'None':
            # Load Pre-trained StaticNeRF
            ckpt_path = args.ft_path_S
            print('Reloading StaticNeRF from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            render_kwargs_train['network_fn_s'].load_state_dict(ckpt['network_fn_s_state_dict'])
        else:
            # Train StaticNeRF from scratch
            for i in range(args.N_iters):
                time0 = time.time()

                # No raybatching as we need to take random rays from one image at a time
                img_i = np.random.choice(i_train)
                t = img_i / num_img * 2. - 1.0 # time of the current frame
                target = images[img_i]
                pose = poses[img_i, :3, :4]
                mask = masks[img_i] # Static region mask

                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose)) # (H, W, 3), (H, W, 3)
                coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
                select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand], replace=False)
                select_coords = coords_s[select_inds_s]

                def select_batch(value, select_coords=select_coords):
                    return value[select_coords[:, 0], select_coords[:, 1]]

                rays_o = select_batch(rays_o) # (N_rand, 3)
                rays_d = select_batch(rays_d) # (N_rand, 3)
                target_rgb = select_batch(target)
                batch_mask = select_batch(mask[..., None])
                batch_rays = torch.stack([rays_o, rays_d], 0)

                #####  Core optimization loop  #####
                ret = render(t,
                             False,
                             H, W, focal,
                             chunk=args.chunk,
                             rays=batch_rays,
                             **render_kwargs_train)

                optimizer.zero_grad()

                # Compute MSE loss between rgb_s and true RGB.
                img_s_loss = img2mse(ret['rgb_map_s'], target_rgb)
                psnr_s = mse2psnr(img_s_loss)
                loss = args.static_loss_lambda * img_s_loss

                loss.backward()
                optimizer.step()

                # Learning rate decay.
                decay_rate = 0.1
                decay_steps = args.lrate_decay
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

                dt = time.time() - time0

                print(f"Pretraining step: {global_step}, Loss: {loss}, Time: {dt}, expname: {expname}")

                if i % args.i_print == 0:
                    writer.add_scalar("loss", loss.item(), i)
                    writer.add_scalar("lr", new_lrate, i)
                    writer.add_scalar("psnr_s", psnr_s.item(), i)

                if i % args.i_img == 0:
                    target = images[img_i]
                    pose = poses[img_i, :3, :4]
                    mask = masks[img_i]

                    with torch.no_grad():
                        ret = render(t,
                                     False,
                                     H, W, focal,
                                     chunk=1024*16,
                                     c2w=pose,
                                     **render_kwargs_test)

                        # Save out the validation image for Tensorboard-free monitoring
                        writer.add_image("rgb_holdout", target, global_step=i, dataformats='HWC')
                        writer.add_image("mask", mask, global_step=i, dataformats='HW')
                        writer.add_image("rgb_s", torch.clamp(ret['rgb_map_s'], 0., 1.), global_step=i, dataformats='HWC')
                        writer.add_image("depth_s", normalize_depth(ret['depth_map_s']), global_step=i, dataformats='HW')
                        writer.add_image("acc_s", ret['acc_map_s'], global_step=i, dataformats='HW')

                global_step += 1

        # Save the pretrained weight
        torch.save({
            'global_step': global_step,
            'network_fn_s_state_dict': render_kwargs_train['network_fn_s'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(basedir, expname, 'Pretrained_S.tar'))

        # Reset
        render_kwargs_train.update({'pretrain': False})
        global_step = start

        # Fix the StaticNeRF and only train the DynamicNeRF
        grad_vars_d = list(render_kwargs_train['network_fn_d'].parameters())
        optimizer = torch.optim.Adam(params=grad_vars_d, lr=args.lrate, betas=(0.9, 0.999))

    for i in range(start, args.N_iters):
        time0 = time.time()

        # Use frames at t-2, t-1, t, t+1, t+2 (adapted from NSFF)
        if i < decay_iteration * 2000:
            chain_5frames = False
        else:
            chain_5frames = True

        # Lambda decay.
        Temp = 1. / (10 ** (i // (decay_iteration * 1000)))

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        # No raybatching as we need to take random rays from one image at a time
        img_i = np.random.choice(i_train)
        t = img_i / num_img * 2. - 1.0 # time of the current frame
        target = images[img_i]
        pose = poses[img_i, :3, :4]
        mask = masks[img_i] # Static region mask
        invdepth = invdepths[img_i]
        grid = grids[img_i]

        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose)) # (H, W, 3), (H, W, 3)
        coords_d = torch.stack((torch.where(mask < 0.5)), -1)
        coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
        coords = torch.stack((torch.where(mask > -1)), -1)

        # Evenly sample dynamic region and static region
        select_inds_d = np.random.choice(coords_d.shape[0], size=[min(len(coords_d), N_rand//2)], replace=False)
        select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand//2], replace=False)
        select_coords = torch.cat([coords_s[select_inds_s],
                                   coords_d[select_inds_d]], 0)

        def select_batch(value, select_coords=select_coords):
            return value[select_coords[:, 0], select_coords[:, 1]]

        rays_o = select_batch(rays_o) # (N_rand, 3)
        rays_d = select_batch(rays_d) # (N_rand, 3)
        target_rgb = select_batch(target)
        batch_grid = select_batch(grid) # (N_rand, 8)
        batch_mask = select_batch(mask[..., None])
        batch_invdepth = select_batch(invdepth)
        batch_rays = torch.stack([rays_o, rays_d], 0)

        #####  Core optimization loop  #####
        ret = render(t,
                     chain_5frames,
                     H, W, focal,
                     chunk=args.chunk,
                     rays=batch_rays,
                     **render_kwargs_train)

        optimizer.zero_grad()
        loss = 0
        loss_dict = {}

        # Compute MSE loss between rgb_full and true RGB.
        img_loss = img2mse(ret['rgb_map_full'], target_rgb)
        psnr = mse2psnr(img_loss)
        loss_dict['psnr'] = psnr
        loss_dict['img_loss'] = img_loss
        loss += args.full_loss_lambda * loss_dict['img_loss']

        # Compute MSE loss between rgb_s and true RGB.
        img_s_loss = img2mse(ret['rgb_map_s'], target_rgb, batch_mask)
        psnr_s = mse2psnr(img_s_loss)
        loss_dict['psnr_s'] = psnr_s
        loss_dict['img_s_loss'] = img_s_loss
        loss += args.static_loss_lambda * loss_dict['img_s_loss']

        # Compute MSE loss between rgb_d and true RGB.
        img_d_loss = img2mse(ret['rgb_map_d'], target_rgb)
        psnr_d = mse2psnr(img_d_loss)
        loss_dict['psnr_d'] = psnr_d
        loss_dict['img_d_loss'] = img_d_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_loss']

        # Compute MSE loss between rgb_d_f and true RGB.
        img_d_f_loss = img2mse(ret['rgb_map_d_f'], target_rgb)
        psnr_d_f = mse2psnr(img_d_f_loss)
        loss_dict['psnr_d_f'] = psnr_d_f
        loss_dict['img_d_f_loss'] = img_d_f_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_f_loss']

        # Compute MSE loss between rgb_d_b and true RGB.
        img_d_b_loss = img2mse(ret['rgb_map_d_b'], target_rgb)
        psnr_d_b = mse2psnr(img_d_b_loss)
        loss_dict['psnr_d_b'] = psnr_d_b
        loss_dict['img_d_b_loss'] = img_d_b_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_b_loss']

        # Motion loss.
        # Compuate EPE between induced flow and true flow (forward flow).
        # The last frame does not have forward flow.
        if img_i < num_img - 1:
            pts_f = ret['raw_pts_f']
            weight = ret['weights_d']
            pose_f = poses[img_i + 1, :3, :4]
            induced_flow_f = induce_flow(H, W, focal, pose_f, weight, pts_f, batch_grid[..., :2])
            flow_f_loss = img2mae(induced_flow_f, batch_grid[:, 2:4], batch_grid[:, 4:5])
            loss_dict['flow_f_loss'] = flow_f_loss
            loss += args.flow_loss_lambda * Temp * loss_dict['flow_f_loss']

        # Compuate EPE between induced flow and true flow (backward flow).
        # The first frame does not have backward flow.
        if img_i > 0:
            pts_b = ret['raw_pts_b']
            weight = ret['weights_d']
            pose_b = poses[img_i - 1, :3, :4]
            induced_flow_b = induce_flow(H, W, focal, pose_b, weight, pts_b, batch_grid[..., :2])
            flow_b_loss = img2mae(induced_flow_b, batch_grid[:, 5:7], batch_grid[:, 7:8])
            loss_dict['flow_b_loss'] = flow_b_loss
            loss += args.flow_loss_lambda * Temp * loss_dict['flow_b_loss']

        # Slow scene flow. The forward and backward sceneflow should be small.
        slow_loss = L1(ret['sceneflow_b']) + L1(ret['sceneflow_f'])
        loss_dict['slow_loss'] = slow_loss
        loss += args.slow_loss_lambda * loss_dict['slow_loss']

        # Smooth scene flow. The summation of the forward and backward sceneflow should be small.
        smooth_loss = compute_sf_smooth_loss(ret['raw_pts'],
                                             ret['raw_pts_f'],
                                             ret['raw_pts_b'],
                                             H, W, focal)
        loss_dict['smooth_loss'] = smooth_loss
        loss += args.smooth_loss_lambda * loss_dict['smooth_loss']

        # Spatial smooth scene flow. (loss adapted from NSFF)
        sp_smooth_loss = compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_f'], H, W, focal) \
                       + compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_b'], H, W, focal)
        loss_dict['sp_smooth_loss'] = sp_smooth_loss
        loss += args.smooth_loss_lambda * loss_dict['sp_smooth_loss']

        # Consistency loss.
        consistency_loss = L1(ret['sceneflow_f'] + ret['sceneflow_f_b']) + \
                           L1(ret['sceneflow_b'] + ret['sceneflow_b_f'])
        loss_dict['consistency_loss'] = consistency_loss
        loss += args.consistency_loss_lambda * loss_dict['consistency_loss']

        # Mask loss.
        mask_loss = L1(ret['blending'][batch_mask[:, 0].type(torch.bool)]) + \
                    img2mae(ret['dynamicness_map'][..., None], 1 - batch_mask)
        loss_dict['mask_loss'] = mask_loss
        if i < decay_iteration * 1000:
            loss += args.mask_loss_lambda * loss_dict['mask_loss']

        # Sparsity loss.
        sparse_loss = entropy(ret['weights_d']) + entropy(ret['blending'])
        loss_dict['sparse_loss'] = sparse_loss
        loss += args.sparse_loss_lambda * loss_dict['sparse_loss']

        # Depth constraint
        # Depth in NDC space equals to negative disparity in Euclidean space.
        depth_loss = compute_depth_loss(ret['depth_map_d'], -batch_invdepth)
        loss_dict['depth_loss'] = depth_loss
        loss += args.depth_loss_lambda * Temp * loss_dict['depth_loss']

        # Order loss
        order_loss = torch.mean(torch.square(ret['depth_map_d'][batch_mask[:, 0].type(torch.bool)] - \
                                             ret['depth_map_s'].detach()[batch_mask[:, 0].type(torch.bool)]))
        loss_dict['order_loss'] = order_loss
        loss += args.order_loss_lambda * loss_dict['order_loss']

        sf_smooth_loss = compute_sf_smooth_loss(ret['raw_pts_b'],
                                                ret['raw_pts'],
                                                ret['raw_pts_b_b'],
                                                H, W, focal) + \
                         compute_sf_smooth_loss(ret['raw_pts_f'],
                                                ret['raw_pts_f_f'],
                                                ret['raw_pts'],
                                                H, W, focal)
        loss_dict['sf_smooth_loss'] = sf_smooth_loss
        loss += args.smooth_loss_lambda * loss_dict['sf_smooth_loss']

        if chain_5frames:
            img_d_b_b_loss = img2mse(ret['rgb_map_d_b_b'], target_rgb)
            loss_dict['img_d_b_b_loss'] = img_d_b_b_loss
            loss += args.dynamic_loss_lambda * loss_dict['img_d_b_b_loss']

            img_d_f_f_loss = img2mse(ret['rgb_map_d_f_f'], target_rgb)
            loss_dict['img_d_f_f_loss'] = img_d_f_f_loss
            loss += args.dynamic_loss_lambda * loss_dict['img_d_f_f_loss']

        loss.backward()
        optimizer.step()

        # Learning rate decay.
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0

        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}, chain_5frames: {chain_5frames}, expname: {expname}")

        # Rest is logging
        if i % args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:
                raise NotImplementedError
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_d_state_dict': render_kwargs_train['network_fn_d'].state_dict(),
                    'network_fn_s_state_dict': render_kwargs_train['network_fn_s'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

            print('Saved weights at', path)

        if i % args.i_video == 0 and i > 0:

            # Change time and change view at the same time.
            time2render = np.concatenate((np.repeat((i_train / float(num_img) * 2. - 1.0), 4),
                                          np.repeat((i_train / float(num_img) * 2. - 1.0)[::-1][1:-1], 4)))
            if len(time2render) > len(render_poses):
                pose2render = np.tile(render_poses, (int(np.ceil(len(time2render) / len(render_poses))), 1, 1))
                pose2render = pose2render[:len(time2render)]
                pose2render = torch.Tensor(pose2render)
            else:
                time2render = np.tile(time2render, int(np.ceil(len(render_poses) / len(time2render))))
                time2render = time2render[:len(render_poses)]
                pose2render = torch.Tensor(render_poses)
            result_type = 'novelviewtime'

            testsavedir = os.path.join(
                basedir, expname, result_type + '_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                ret = render_path(pose2render, time2render,
                                  hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
            moviebase = os.path.join(
                testsavedir, '{}_{}_{:06d}_'.format(expname, result_type, i))
            save_res(moviebase, ret)

        if i % args.i_testset == 0 and i > 0:

            # Change view and time.
            pose2render = torch.Tensor(poses)
            time2render = i_train / float(num_img) * 2. - 1.0
            result_type = 'testset'

            testsavedir = os.path.join(
                basedir, expname, result_type + '_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                ret = render_path(pose2render, time2render,
                                  hwf, args.chunk, render_kwargs_test, savedir=testsavedir,
                                  flows_gt_f=grids[:, :, :, 2:4], flows_gt_b=grids[:, :, :, 5:7])
            moviebase = os.path.join(
                testsavedir, '{}_{}_{:06d}_'.format(expname, result_type, i))
            save_res(moviebase, ret)

            # Fix view (first view) and change time.
            pose2render = torch.Tensor(poses[0:1, ...].expand([int(num_img), 3, 4]))
            time2render = i_train / float(num_img) * 2. - 1.0
            result_type = 'testset_view000'

            testsavedir = os.path.join(
                basedir, expname, result_type + '_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                ret = render_path(pose2render, time2render,
                                  hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
            moviebase = os.path.join(
                testsavedir, '{}_{}_{:06d}_'.format(expname, result_type, i))
            save_res(moviebase, ret)

            # Fix time (the first timestamp) and change view.
            pose2render = torch.Tensor(poses)
            time2render = np.tile(i_train[0], [int(num_img)]) / float(num_img) * 2. - 1.0
            result_type = 'testset_time000'

            testsavedir = os.path.join(
                basedir, expname, result_type + '_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                ret = render_path(pose2render, time2render,
                                  hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
            moviebase = os.path.join(
                testsavedir, '{}_{}_{:06d}_'.format(expname, result_type, i))
            save_res(moviebase, ret)

        if i % args.i_print == 0:
            writer.add_scalar("loss", loss.item(), i)
            writer.add_scalar("lr", new_lrate, i)
            writer.add_scalar("Temp", Temp, i)
            for loss_key in loss_dict:
                writer.add_scalar(loss_key, loss_dict[loss_key].item(), i)

        if i % args.i_img == 0:
            # Log a rendered training view to Tensorboard.
            # img_i = np.random.choice(i_train[1:-1])
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            mask = masks[img_i]
            grid = grids[img_i]
            invdepth = invdepths[img_i]

            flow_f_img = flow_to_image(grid[..., 2:4].cpu().numpy())
            flow_b_img = flow_to_image(grid[..., 5:7].cpu().numpy())

            with torch.no_grad():
                ret = render(t,
                             False,
                             H, W, focal,
                             chunk=1024*16,
                             c2w=pose,
                             **render_kwargs_test)

                # The last frame does not have forward flow.
                pose_f = poses[min(img_i + 1, int(num_img) - 1), :3, :4]
                induced_flow_f = induce_flow(H, W, focal, pose_f, ret['weights_d'], ret['raw_pts_f'], grid[..., :2])

                # The first frame does not have backward flow.
                pose_b = poses[max(img_i - 1, 0), :3, :4]
                induced_flow_b = induce_flow(H, W, focal, pose_b, ret['weights_d'], ret['raw_pts_b'], grid[..., :2])

                induced_flow_f_img = flow_to_image(induced_flow_f.cpu().numpy())
                induced_flow_b_img = flow_to_image(induced_flow_b.cpu().numpy())

                psnr = mse2psnr(img2mse(ret['rgb_map_full'], target))

                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i == 0:
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(ret['rgb_map_full'].cpu().numpy()))

                writer.add_scalar("psnr_holdout", psnr.item(), i)
                writer.add_image("rgb_holdout", target, global_step=i, dataformats='HWC')
                writer.add_image("mask", mask, global_step=i, dataformats='HW')
                writer.add_image("disp", torch.clamp(invdepth / percentile(invdepth, 97), 0., 1.), global_step=i, dataformats='HW')

                writer.add_image("rgb", torch.clamp(ret['rgb_map_full'], 0., 1.), global_step=i, dataformats='HWC')
                writer.add_image("depth", normalize_depth(ret['depth_map_full']), global_step=i, dataformats='HW')
                writer.add_image("acc", ret['acc_map_full'], global_step=i, dataformats='HW')

                writer.add_image("rgb_s", torch.clamp(ret['rgb_map_s'], 0., 1.), global_step=i, dataformats='HWC')
                writer.add_image("depth_s", normalize_depth(ret['depth_map_s']), global_step=i, dataformats='HW')
                writer.add_image("acc_s", ret['acc_map_s'], global_step=i, dataformats='HW')

                writer.add_image("rgb_d", torch.clamp(ret['rgb_map_d'], 0., 1.), global_step=i, dataformats='HWC')
                writer.add_image("depth_d", normalize_depth(ret['depth_map_d']), global_step=i, dataformats='HW')
                writer.add_image("acc_d", ret['acc_map_d'], global_step=i, dataformats='HW')

                writer.add_image("induced_flow_f", induced_flow_f_img, global_step=i, dataformats='HWC')
                writer.add_image("induced_flow_b", induced_flow_b_img, global_step=i, dataformats='HWC')
                writer.add_image("flow_f_gt", flow_f_img, global_step=i, dataformats='HWC')
                writer.add_image("flow_b_gt", flow_b_img, global_step=i, dataformats='HWC')

                writer.add_image("dynamicness", ret['dynamicness_map'], global_step=i, dataformats='HW')

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
