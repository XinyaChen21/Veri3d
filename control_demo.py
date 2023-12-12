import os
import torch
import imageio
import numpy as np
from munch import *
from tqdm import tqdm
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

from options import BaseOptions
from model import VoxelHumanGenerator as Generator
from dataset import DeepFashionDataset, DemoDataset, AISTDataset, AISTDemoDataset
from utils import requires_grad
import sklearn.decomposition
import joblib
import copy


torch.random.manual_seed(1008611)
import random
random.seed(1008611)


def view_shape_appearance_control(opt, g_ema, device, sample_z, person_idx,
                            sample_trans, sample_beta, sample_theta,sample_cam_extrinsics, sample_focals,sample_intrinsic,
                            dataset_name='DeepFashion'):
    if dataset_name=='DeepFashion':
        panning_angle = np.pi / 3
    else:
        panning_angle = 2*np.pi 
                  
    requires_grad(g_ema, False)
    g_ema.is_train = False

    result_dir = os.path.join(opt.results_dst_dir,'sup_video', f'{person_idx}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    #-----------------view control------------------
    video_list = []
    for k in tqdm(range(120)):
        if k < 30:
            angle = (panning_angle / 2) * (k / 30)
        elif k >= 30 and k < 90:
            angle = panning_angle / 2 - panning_angle * ((k - 30) / 60)
        else:
            angle = -panning_angle / 2 * ((120 - k) / 30)
        delta = R.from_rotvec(angle * np.array([0, 1, 0]))
        if opt.move_camera:
            r = R.from_rotvec(-np.pi * np.array([0, 0, 1]))
            delta = delta*r
            delta = torch.from_numpy(R.as_matrix(delta)).float().to(device)
            new_sample_cam_extrinsics = sample_cam_extrinsics.clone()
            new_sample_cam_extrinsics[0,:3,:3] = torch.matmul(delta,sample_cam_extrinsics[0,:3,:3])
            new_sample_cam_extrinsics[0,:3,3] -= (torch.matmul(delta,sample_trans[0]) + sample_trans[0])
            new_sample_cam_extrinsics = torch.inverse(new_sample_cam_extrinsics)
            new_sample_theta = sample_theta.clone()
        else:
            r = R.from_rotvec(sample_theta[0, :3].cpu().numpy())
            new_r = delta * r
            new_sample_theta = sample_theta.clone()
            new_sample_theta[0, :3] = torch.from_numpy(new_r.as_rotvec()).to(device)
            new_sample_cam_extrinsics = sample_cam_extrinsics.clone()
        with torch.no_grad():
            j = 0
            chunk = 1
            if dataset_name=='DeepFashion' or dataset_name=='Surreal':
                out = g_ema([sample_z[j:j+chunk]],
                            new_sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            sample_beta[j:j+chunk],
                            new_sample_theta[j:j+chunk],
                            sample_trans[j:j+chunk],
                            truncation=opt.truncation_ratio)
            elif dataset_name=='AIST++':
                out = g_ema([sample_z[j:j+chunk]],
                        sample_cam_extrinsics[j:j+chunk],
                        sample_intrinsic[j:j+chunk],
                        sample_beta[j:j+chunk],
                        new_sample_theta[j:j+chunk],
                        sample_trans[j:j+chunk],
                        truncation=opt.truncation_ratio)
        rgb_images_thumbs = out[1].detach().cpu()[..., :3]
        g_ema.zero_grad()
        video_list.append((rgb_images_thumbs.numpy() + 1) / 2. * 255. + 0.5)
    all_img = np.concatenate(video_list, 0).astype(np.uint8)
    imageio.mimwrite(os.path.join(result_dir,'view_{}.mp4'.format(str(person_idx).zfill(7))), all_img, fps=30, quality=8)
    
    #-----------------shape control------------------
    video_list = []
    sample_num=40
    pc_low=-3
    pc_high=3
    start = 0

    step = (pc_high - start)/sample_num
    pc1_list = list(np.arange(start,pc_high,step))
    step = (pc_high - pc_low)/(sample_num*2)
    pc1_list += list(np.arange(pc_low,pc_high,step))[::-1]
    step = (start - pc_low)/sample_num    
    pc1_list += list(np.arange(pc_low,start,step))

    step = (pc_high - start)/sample_num
    pc2_list = list(np.arange(start,pc_high,step))
    step = (pc_high - pc_low)/(sample_num*2)
    pc2_list += list(np.arange(pc_low,pc_high,step))[::-1]
    step = (start - pc_low)/sample_num    
    pc2_list += list(np.arange(pc_low,start,step))
    pc_list = pc1_list + pc2_list
    for idx in tqdm(range(len(pc_list))):
        cur_sample_beta = sample_beta.clone() * 0
        if idx < len(pc_list)//2:
            cur_sample_beta[0,0] = pc_list[idx]
        else:
            cur_sample_beta[0,1] = pc_list[idx]
        with torch.no_grad():
            j = 0
            chunk = 1
            if dataset_name=='DeepFashion' or dataset_name=='Surreal':
                out = g_ema([sample_z[j:j+chunk]],
                            sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            cur_sample_beta[j:j+chunk],
                            new_sample_theta[j:j+chunk],
                            sample_trans[j:j+chunk],
                            truncation=opt.truncation_ratio)
            elif dataset_name=='AIST++':
                out = g_ema([sample_z[j:j+chunk]],
                        sample_cam_extrinsics[j:j+chunk],
                        sample_intrinsic[j:j+chunk],
                        cur_sample_beta[j:j+chunk],
                        new_sample_theta[j:j+chunk],
                        sample_trans[j:j+chunk],
                        truncation=opt.truncation_ratio)
        rgb_images_thumbs = out[1].detach().cpu()[..., :3]
        g_ema.zero_grad()
        video_list.append((rgb_images_thumbs.numpy() + 1) / 2. * 255. + 0.5)
    shape_all_img = np.concatenate(video_list, 0).astype(np.uint8)
    imageio.mimwrite(os.path.join(result_dir,'shape_{}.mp4'.format(str(person_idx).zfill(7))), shape_all_img, fps=30, quality=8)

    #-----------------appearance control------------------
    video_list = []
    sample_num=120
    if not os.path.exists(os.path.join(os.path.join(result_dir,'sample_z_2.npy'))):
        sample_z_2 = torch.randn(1, opt.style_dim, device=device)
        if (sample_z_2-sample_z).sum()==0:
            sample_z_2 = torch.randn(1, opt.style_dim, device=device)
        np.save(os.path.join(result_dir,'sample_z_2.npy'),sample_z_2.data.cpu().numpy())
    else:
        sample_z_2 = np.load(os.path.join(result_dir,'sample_z_2.npy'))
        sample_z_2 = torch.from_numpy(sample_z_2).float().to(sample_beta.device)
    for idx in tqdm(range(sample_num*2)):
        if idx < sample_num:
            z_interpolate = (sample_z_2-sample_z)/sample_num * idx + sample_z
        else:
            z_interpolate = (sample_z-sample_z_2)/sample_num * (idx-sample_num) + sample_z_2
        with torch.no_grad():
            j = 0
            chunk = 1
            if dataset_name=='DeepFashion' or dataset_name=='Surreal':
                out = g_ema([z_interpolate[j:j+chunk]],
                            sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            sample_beta[j:j+chunk],
                            new_sample_theta[j:j+chunk],
                            sample_trans[j:j+chunk],
                            truncation=opt.truncation_ratio)
            elif dataset_name=='AIST++':
                out = g_ema([z_interpolate[j:j+chunk]],
                        sample_cam_extrinsics[j:j+chunk],
                        sample_intrinsic[j:j+chunk],
                        sample_beta[j:j+chunk],
                        new_sample_theta[j:j+chunk],
                        sample_trans[j:j+chunk],
                        truncation=opt.truncation_ratio)
        rgb_images_thumbs = out[1].detach().cpu()[..., :3]
        g_ema.zero_grad()
        video_list.append((rgb_images_thumbs.numpy() + 1) / 2. * 255. + 0.5)
    code_all_img = np.concatenate(video_list, 0).astype(np.uint8)
    imageio.mimwrite(os.path.join(result_dir,'appearance_{}.mp4'.format(str(person_idx).zfill(7))), code_all_img, fps=30, quality=8)

    overview_all_img = np.concatenate([all_img,shape_all_img,code_all_img], 0).astype(np.uint8)
    imageio.mimwrite(os.path.join(result_dir,'overview_{}.mp4'.format(str(person_idx).zfill(7))), overview_all_img, fps=30, quality=8)


def part_control(opt, g_ema, device, given_sample_z, sample_trans, sample_beta, sample_theta, sample_cam_extrinsics, sample_focals, pca_num_sample=8000,cluster_part = 2, base_num = 30, person_idx=0,visual_num_sample=6):
    requires_grad(g_ema, False)
    g_ema.is_train = False
    #-------setting-------
    pca_save_dir = os.path.join(opt.results_dst_dir, f'part_control/pca_{cluster_part}')
    if not os.path.exists(pca_save_dir):
        os.makedirs(pca_save_dir)
    visual_base_num = 10
    pca_component_file = f'{pca_save_dir}/pca_part_{cluster_part}_num_sample_{pca_num_sample}.m'
    part_statistic_file = f'{pca_save_dir}/part_statistic_part_{cluster_part}_num_sample_{pca_num_sample}.npy'
    #-------setting-------
    fine2coarse_part = {0:0,3:0,6:0,9:0,13:0,14:0,12:0,
                        16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,
                        # 16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,
                        15:2,
                        1:3,2:3,4:3,5:3,7:3,8:3,11:3,10:3}
    fine_part = g_ema.renderer.smpl_seg.data.cpu().numpy()
    coarse_part = np.zeros_like(fine_part)
    for i in range(fine_part.shape[0]):
        coarse_part[i] = fine2coarse_part[fine_part[i]]
    if not os.path.exists(pca_component_file) or not os.path.exists(part_statistic_file):
        sample_z_list = {}
        vertices_feature_list = []
        for i in tqdm(range(pca_num_sample)):

            sample_z = torch.randn(1, opt.style_dim, device=device)
            sample_z_list[str(i).zfill(7)] = sample_z.cpu().numpy()
            with torch.no_grad():
                j = 0
                chunk = 1
                vertices_feature = g_ema.get_vertices_feature([sample_z[j:j+chunk]])
                vertices_feature_list.append(vertices_feature.data.cpu().numpy())
        
        vertices_features = np.concatenate(vertices_feature_list,axis = 0)
        
        part_vertices_features = vertices_features.transpose(2,0,1,3)[coarse_part == cluster_part].transpose(1,0,2,3).reshape(pca_num_sample,-1)

        pca = sklearn.decomposition.PCA(n_components=base_num)
        part_vertices_features_transformed = pca.fit_transform(part_vertices_features)
        joblib.dump(pca, pca_component_file)
        part_statistic = {}
        part_vertices_feature_base = pca.components_
        # explained_variance =  pca.explained_variance_
        # explained_variance_ratio =  pca.explained_variance_ratio_

        for j in range(len(part_vertices_feature_base)):
            low = int(part_vertices_features_transformed[:,j].min())
            high = int(part_vertices_features_transformed[:,j].max())
            part_statistic[j] = {}
            part_statistic[j]['low'] = low
            part_statistic[j]['high'] = high
        np.save(part_statistic_file,part_statistic)
    else:
        pca = joblib.load(pca_component_file)
        part_statistic = np.load(part_statistic_file,allow_pickle=True).item()
        
    #-----------------part control------------------
    vertices_feature = g_ema.get_vertices_feature([given_sample_z])

    img_set = []
    ori_vertices_feature = vertices_feature[:1].data.cpu().numpy()

    ori_coor = pca.transform(ori_vertices_feature.transpose(2,0,1,3)[coarse_part == cluster_part].transpose(1,0,2,3).reshape(1,-1))[0]
    visual_dir = os.path.join(opt.results_dst_dir, 'sup_video', f'{person_idx}')
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    
    img_set = []
    for j in range(visual_base_num):
        if cluster_part==2:
            if j not in [0,4,5,6]:
                continue
        if cluster_part==3:
            if j not in [0,1,2,3]:
                continue
        if cluster_part==0:
            if j not in [5,6,7,8]:
                continue
        input_vertices_feature_list = []

        ori_coor_copy = copy.deepcopy(ori_coor)
        low = part_statistic[j]['low']
        high = part_statistic[j]['high']
        skip = (high-low)/2#/3*2
        step = max((high - low - skip) / visual_num_sample,0.00001)

        for value in np.arange(low+skip/2, high-skip/2, step):
            ori_coor_copy[j] = value
            cur_vertices_feature = np.dot(ori_coor_copy, pca.components_) + pca.mean_
            changed_vertices_feature = copy.deepcopy(ori_vertices_feature)
            changed_vertices_feature.squeeze().transpose(1,0)[coarse_part == cluster_part] = cur_vertices_feature.reshape(-1,vertices_feature.shape[1])
            input_vertices_feature_list.append(changed_vertices_feature)

        for input_vertices_feature in input_vertices_feature_list:
            input_vertices_feature = torch.from_numpy(input_vertices_feature).to(given_sample_z.device)
            with torch.no_grad():
                jj = 0
                chunk = 1
                out = g_ema([given_sample_z[jj:jj+chunk]],
                            sample_cam_extrinsics[jj:jj+chunk],
                            sample_focals[jj:jj+chunk],
                            sample_beta[jj:jj+chunk],
                            sample_theta[jj:jj+chunk],
                            sample_trans[jj:jj+chunk],
                            truncation=opt.truncation_ratio,
                            input_vertices_feature = input_vertices_feature)
            rgb_images_thumbs = out[1].detach().cpu()[..., :3]
            g_ema.zero_grad()
            rgb_images_thumbs = (rgb_images_thumbs.numpy() + 1) / 2. * 255. + 0.5
            img_set.append(rgb_images_thumbs)
            
    code_all_img = np.concatenate(img_set, 0).astype(np.uint8)
    imageio.mimwrite(os.path.join(visual_dir,f'part_{cluster_part}_{pca_num_sample}.mp4'.format(str(person_idx).zfill(7))), code_all_img, fps=30, quality=8)
        
        
def aistpose_control(opt, dataset, g_ema, device,sample_z, video_idx=7, person_idx=0):
    requires_grad(g_ema, False)
    g_ema.is_train = False
    sample_trans, sample_beta, sample_theta, sample_intrinsic, sample_cam_extrinsics = dataset.get_smpl_camera_param(device,video_idx=video_idx)

    visual_dir = os.path.join(opt.results_dst_dir, 'sup_video', f'{person_idx}')
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    img_set = []
    for jj in tqdm(range(sample_beta.shape[0])):
        chunk = 1
        with torch.no_grad():   
            
            out = g_ema([sample_z],
                        sample_cam_extrinsics[jj:jj+chunk],
                        sample_intrinsic[jj:jj+chunk],
                        sample_beta[jj:jj+chunk],
                        sample_theta[jj:jj+chunk].clone(),
                        sample_trans[jj:jj+chunk],
                        truncation=opt.truncation_ratio)
            rgb_images_thumbs = out[1].detach().cpu()[..., :3]
            g_ema.zero_grad()
            rgb_images_thumbs = (rgb_images_thumbs.numpy() + 1) / 2. * 255. + 0.5
            img_set.append(rgb_images_thumbs)
    all_img = np.concatenate(img_set, 0).astype(np.uint8)
    imageio.mimwrite(os.path.join(visual_dir,'pose_person{}_video{}.mp4'.format(str(person_idx).zfill(7),str(video_idx))), all_img, fps=30, quality=8)


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    
    opt.rendering.perturb = 0
    # opt.inference.size = opt.model.size
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    
    checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname, 'volume_renderer')
    checkpoint_path = os.path.join(checkpoints_dir,
                                    'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
    # define results directory name
    result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    # create results directory
    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir, opt.rendering.dataset)
    if opt.inference.fixed_camera_angles:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'fixed_angles')
    else:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'random_angles')
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    # load generation model
    g_ema = Generator(opt.model, opt.rendering, full_pipeline=False).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if k not in model_dict:
            continue
        if v.size() == model_dict[k].size():
            model_dict[k] = v
        else:
            print(k)

    g_ema.load_state_dict(model_dict)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    
    if 'DeepFashion' in opt.dataset.dataset_path:
        file_list = os.path.join(opt.dataset.dataset_path, 'train_list.txt')
    else:
        file_list = None
    if opt.rendering.dataset=='DeepFashion':
        dataset = DeepFashionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                     opt.model.renderer_spatial_output_dim, file_list,
                                     white_bg=opt.rendering.white_bg)
    elif opt.rendering.dataset=='AIST++':
        dataset = AISTDataset(opt.dataset.dataset_path, transform,
                                    nerf_resolution=opt.model.renderer_spatial_output_dim,
                                    white_bg=opt.rendering.white_bg,
                                    random_flip=opt.dataset.random_flip)
    elif opt.rendering.dataset=="Surreal":
        dataset = DeepFashionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                     opt.model.renderer_spatial_output_dim, file_list,
                                     white_bg=opt.rendering.white_bg)
    else:
        dataset = DemoDataset()

    g_ema.renderer.is_train = False
    g_ema.renderer.perturb = 0
    g_ema.eval()

    for person_idx in range(0,opt.inference.identities):
        result_dir = os.path.join(opt.inference.results_dst_dir,'sup_video', f'{person_idx}')  
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)  
        if not os.path.exists(os.path.join(os.path.join(result_dir,'sample_z.npy'))):
            sample_z = torch.randn(1, opt.inference.style_dim, device=device)
            np.save(os.path.join(result_dir,'sample_z.npy'),sample_z.data.cpu().numpy())
        else:
            sample_z = np.load(os.path.join(result_dir,'sample_z.npy'))
            sample_z = torch.from_numpy(sample_z).float().to(device)

        if opt.rendering.dataset=='DeepFashion':
            opt.model.renderer_spatial_output_dim = (512, 256)
            opt.model.smpl_gender = 'neural'
            sample_trans, sample_beta, sample_theta = dataset.sample_smpl_param(1, device, val=False)
            sample_cam_extrinsics, sample_focals = dataset.get_camera_extrinsics(1, device, val=False)
            sample_intrinsic=None
        elif opt.rendering.dataset=='AIST++':
            sample_focals=None
            sample_trans, sample_beta, sample_theta, sample_intrinsic, sample_cam_extrinsics = dataset.sample_smpl_param(1, device)
        else:
            opt.model.renderer_spatial_output_dim = (128, 64)
            opt.model.smpl_gender = 'neural'
            sample_trans, sample_beta, sample_theta = dataset.sample_smpl_param(1, device, val=False)
            sample_cam_extrinsics, sample_focals = dataset.get_camera_extrinsics(1, device, val=False)
            sample_intrinsic=None
            
        #view & shape & appearance control
        view_shape_appearance_control(opt.inference, g_ema, device, sample_z, person_idx, 
        sample_trans, sample_beta, sample_theta,sample_cam_extrinsics, sample_focals,sample_intrinsic,
        dataset_name=opt.rendering.dataset)
        
        #part control
        if opt.rendering.dataset=='DeepFashion':
            for cluster_part in [0,2,3]:
                part_control(opt.inference, g_ema, device, sample_z, sample_trans, sample_beta, sample_theta, sample_cam_extrinsics, sample_focals,\
                            pca_num_sample=1000,cluster_part = cluster_part, base_num = 30, person_idx=person_idx,visual_num_sample=40)
        
        #pose control with aist++ pose
        '''
        if opt.rendering.dataset=='DeepFashion':
            opt.model.renderer_spatial_output_dim = (512, 512)
        elif opt.rendering.dataset=='Surreal':
            opt.model.renderer_spatial_output_dim = (128, 128)
        opt.model.smpl_gender = 'male'
        aist_g_ema = Generator(opt.model, opt.rendering, full_pipeline=False).to(device)
        pretrained_weights_dict = checkpoint["g_ema"]
        model_dict = aist_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k not in model_dict:
                continue
            if v.size() == model_dict[k].size():
                model_dict[k] = v
            else:
                print(k)

        aist_g_ema.load_state_dict(model_dict)
        aist_demo_dataset = AISTDemoDataset(nerf_resolution=opt.model.renderer_spatial_output_dim)
        aist_g_ema.renderer.is_train = False
        aist_g_ema.renderer.perturb = 0
        aist_g_ema.eval()
        for video_idx in [2,4,7]:
            aistpose_control(opt.inference, aist_demo_dataset, aist_g_ema, device, sample_z, person_idx=person_idx, video_idx=video_idx)
        '''
