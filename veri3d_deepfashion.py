import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as st
from smplx.lbs import transform_mat
from pytorch3d.ops.knn import knn_points
import dnnlib
from pytorch3d.structures import Meshes
from network_util import initseq
from fourier import get_embedder


from smpl_utils import init_smpl, get_J, batch_rodrigues


class MLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, input_ch=3, skips=None, output_ch=4):
        super(MLP, self).__init__()

        if skips is None:
            skips = [4]

        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch

        self.code_size = mlp_width

        pts_block_mlps = [nn.Linear(input_ch, mlp_width), nn.ReLU()]

        layers_to_cat_input = []
        for i in range(mlp_depth - 1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [
                    nn.Linear(mlp_width + input_ch, mlp_width),
                    nn.ReLU(),
                ]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        self.output_linear = nn.Sequential(nn.Linear(mlp_width, output_ch))
        initseq(self.output_linear)

    def forward(self, pos_embed):
        h = pos_embed
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)

        outputs = self.output_linear(h)

        return outputs


class UVMapping:
    def __init__(self):
        self.fs = np.loadtxt("./assets/smpl_f_ft_vt/smpl_f.txt").astype(np.int32)
        self.fts = np.loadtxt("./assets/smpl_f_ft_vt/smpl_ft.txt").astype(np.int32)
        self.vts = np.loadtxt("./assets/smpl_f_ft_vt/smpl_vt.txt").astype(np.float32)

        self.uv_img_size = 128

    def get_vetidx_to_uvidx(self):
        vetidx_to_uvidx = {}
        for face_idx in range(self.fs.shape[0]):
            f = self.fs[face_idx]
            ft = self.fts[face_idx]
            for i in range(f.shape[0]):
                vet_idx = f[i]
                uv_idx = ft[i]
                if vet_idx not in vetidx_to_uvidx.keys():
                    vetidx_to_uvidx[vet_idx] = []
                    vetidx_to_uvidx[vet_idx].append(uv_idx)
                else:
                    if uv_idx not in vetidx_to_uvidx[vet_idx]:
                        vetidx_to_uvidx[vet_idx].append(uv_idx)
        vetidx_to_uvidx_mat = np.zeros((len(vetidx_to_uvidx.keys()), 4))
        vetidx_to_uvidx_mask = np.zeros((len(vetidx_to_uvidx.keys()), 4))
        for vetidx in vetidx_to_uvidx.keys():
            # if len(vetidx_to_uvidx[vetidx])>2:
            #     print(len(vetidx_to_uvidx[vetidx]))
            for j in range(len(vetidx_to_uvidx[vetidx])):
                uvidx = vetidx_to_uvidx[vetidx][j]
                vetidx_to_uvidx_mat[vetidx, j] = uvidx
                vetidx_to_uvidx_mask[vetidx, j] = 1
        return vetidx_to_uvidx_mat, vetidx_to_uvidx_mask


class VoxelHuman(nn.Module):
    def __init__(self, opt, smpl_cfgs, style_dim, out_im_res=(128, 64), mode="train"):
        super(VoxelHuman, self).__init__()
        self.smpl_cfgs = smpl_cfgs
        self.style_dim = style_dim
        self.opt = opt
        self.out_im_res = out_im_res
        self.is_train = mode == "train"

        self.register_buffer("inf", torch.Tensor([1e10]), persistent=False)
        self.register_buffer("zero_idx", torch.LongTensor([0]), persistent=False)

        # create meshgrid to generate rays
        i, j = torch.meshgrid(
            torch.linspace(0.5, self.out_im_res[1] - 0.5, self.out_im_res[1]),
            torch.linspace(0.5, self.out_im_res[0] - 0.5, self.out_im_res[0]),
        )

        self.register_buffer("i", i.t().unsqueeze(0), persistent=False)
        self.register_buffer("j", j.t().unsqueeze(0), persistent=False)

        self.N_samples = opt.N_samples
        self.t_vals = (
            torch.linspace(0.0, 1.0 - 1 / self.N_samples, steps=self.N_samples)
            .view(-1)
            .cuda()
        )
        self.perturb = opt.perturb

        self.smpl_model = init_smpl(
            model_folder=smpl_cfgs["model_folder"],
            model_type=smpl_cfgs["model_type"],
            gender=smpl_cfgs["gender"],
            num_betas=smpl_cfgs["num_betas"],
        )

        self.embed_size = 64
        input_ch = 211
        self.multires = 10
        i_embed = 0
        self.uv_num = 1
        self.mean_neighbor = 3
        self.multiple_sample = self.opt.multiple_sample
        self.encoder = self.prepare_stylegan(self.embed_size)

        self.cnl_mlp = MLP(input_ch=input_ch, mlp_depth=2, mlp_width=256, output_ch=4)

        uv_mapping = UVMapping()
        self.uvidx_to_uvcoor = uv_mapping.vts
        self.vetidx_to_uvidx, _ = uv_mapping.get_vetidx_to_uvidx()
        self.uvidx_to_uvcoor = torch.from_numpy(self.uvidx_to_uvcoor)
        self.vetidx_to_uvidx = torch.from_numpy(self.vetidx_to_uvidx)

        self.pos_embed_fn_nerf, pos_embed_size = get_embedder(self.multires, i_embed)

        # create human model
        zero_betas = torch.zeros(10)
        self.zero_init_J = get_J(zero_betas, self.smpl_model)
        parents = self.smpl_model.parents.cpu().numpy()
        self.parents = parents
        self.num_joints = num_joints = parents.shape[0]

        self.vox_index = []
        self.smpl_index = []
        self.actual_vox_bbox = []

        self.smpl_seg = self.smpl_model.lbs_weights.argmax(-1)

        for j in range(num_joints):
            flag, cur_index = self.predefined_bbox(j)
            if flag == False:
                continue

            self.vox_index.append(j)
            self.smpl_index.append(cur_index)

    def prepare_stylegan(self, in_channels):
        def prepare_generator(z_dim, w_dim, out_channels, c_dim=0):
            G_kwargs = dnnlib.EasyDict(
                class_name="networks.Generator",
                z_dim=z_dim,
                w_dim=w_dim,
                mapping_kwargs=dnnlib.EasyDict(),
                synthesis_kwargs=dnnlib.EasyDict(use_noise=False),
            )
            G_kwargs.synthesis_kwargs.channel_base = 32768
            G_kwargs.synthesis_kwargs.channel_max = 512
            G_kwargs.mapping_kwargs.num_layers = 8
            G_kwargs.synthesis_kwargs.num_fp16_res = 0
            G_kwargs.synthesis_kwargs.conv_clamp = None

            uv_feature_size = 256
            g_common_kwargs = dict(
                c_dim=c_dim, img_resolution=uv_feature_size, img_channels=out_channels
            )
            gen = dnnlib.util.construct_class_by_name(**G_kwargs, **g_common_kwargs)
            return gen

        w_dim = 512
        z_dim = 256
        c_dim = 0
        return prepare_generator(z_dim, w_dim, in_channels, c_dim)

    def compute_actual_bbox(self, beta):
        actual_vox_bbox = []
        init_J = get_J(beta.reshape(1, 10), self.smpl_model)
        for j in range(self.num_joints):
            pj = self.parents[j]
            j_coor = init_J[j]
            pj_coor = init_J[pj]
            mid = (j_coor + pj_coor) / 2.0

            # spine direction
            if j in [15, 12, 6, 3]:
                h = np.abs(j_coor[1] - pj_coor[1])
                w = 0.3
                delta = np.array([w, 0.8 * h, w])

            elif j in [4, 5, 7, 8]:
                h = np.abs(j_coor[1] - pj_coor[1])
                w = 0.15
                delta = np.array([w, 0.6 * h, w])

            # arms direction
            elif j in [22, 20, 18, 23, 21, 19]:
                h = np.abs(j_coor[0] - pj_coor[0])
                w = 0.12
                delta = np.array([0.8 * h, w, w])

            # foot direction
            elif j in [10, 11]:
                h = np.abs(j_coor[2] - pj_coor[2])
                w = 0.08
                delta = np.array([w, w, 0.6 * h])

            else:
                continue

            xyz_min = mid - delta
            xyz_max = mid + delta

            if j == 15:
                xyz_max += np.array([0, 0.25, 0])
            elif j == 22:
                xyz_max += np.array([0.25, 0, 0])
            elif j == 23:
                xyz_min -= np.array([0.25, 0, 0])
            elif j == 3:
                xyz_min -= np.array([0, 0.1, 0])
            elif j == 12:
                xyz_min -= np.array([0, 0.25, 0])

            actual_vox_bbox.append(
                (
                    torch.from_numpy(xyz_min).float().cuda(),
                    torch.from_numpy(xyz_max).float().cuda(),
                )
            )

        return actual_vox_bbox

    def smpl_index_by_joint(self, joint_list):
        start_index = self.smpl_seg == joint_list[0]
        if len(joint_list) > 1:
            for i in range(1, len(joint_list)):
                start_index += self.smpl_seg == joint_list[i]
            return start_index > 0
        else:
            return start_index

    def predefined_bbox(self, j, only_cur_index=False):
        flag = True
        if j == 15:
            cur_index = self.smpl_index_by_joint([15])
        elif j == 12:
            cur_index = self.smpl_index_by_joint([9, 13, 14, 6, 16, 17, 12, 15])
        elif j == 9 and only_cur_index:
            flag = False
            cur_index = self.smpl_index_by_joint([9, 13, 14, 6, 16, 17, 3])
        elif j == 6:
            cur_index = self.smpl_index_by_joint([3, 6, 0, 9])
        elif j == 3:
            cur_index = self.smpl_index_by_joint([3, 0, 1, 2, 6])
        elif j == 18:
            cur_index = self.smpl_index_by_joint([13, 18, 16])
        elif j == 20:
            cur_index = self.smpl_index_by_joint([16, 20, 18])
        elif j == 22:
            cur_index = self.smpl_index_by_joint([22, 20, 18])
        elif j == 19:
            cur_index = self.smpl_index_by_joint([14, 17, 19])
        elif j == 21:
            cur_index = self.smpl_index_by_joint([17, 19, 21])
        elif j == 23:
            cur_index = self.smpl_index_by_joint([23, 21, 19])
        elif j == 4:
            cur_index = self.smpl_index_by_joint([0, 1, 4])
        elif j == 7:
            cur_index = self.smpl_index_by_joint([4, 1, 7])
        elif j == 10:
            cur_index = self.smpl_index_by_joint([7, 10, 4])
        elif j == 5:
            cur_index = self.smpl_index_by_joint([0, 2, 5])
        elif j == 8:
            cur_index = self.smpl_index_by_joint([2, 5, 8])
        elif j == 11:
            cur_index = self.smpl_index_by_joint([8, 11, 5])
        else:
            cur_index = None
            flag = False

        if only_cur_index:
            return cur_index

        return flag, cur_index

    def batch_rigid_transform(self, rot_mats, init_J):
        joints = torch.from_numpy(init_J.reshape(1, -1, 3, 1)).cuda()
        parents = self.parents

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
        ).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
        )

        return posed_joints, rel_transforms

    def transform_to_vox_local(self, rays_o, rays_d, transforms_mat, trans):
        inv_transforms_mat = torch.inverse(transforms_mat)[:3, :3].unsqueeze(0)
        rays_o_local = rays_o - trans
        rays_o_local = torch.matmul(
            inv_transforms_mat, (rays_o_local - transforms_mat[:3, -1]).unsqueeze(-1)
        )[..., 0]
        rays_d_local = torch.matmul(inv_transforms_mat, rays_d.unsqueeze(-1))[..., 0]

        return rays_o_local, rays_d_local

    def forward_transform_bbox(self, i, transforms_mat, xyz_min, xyz_max):
        # xyz_min = self.vox_list[i].xyz_min
        # xyz_max = self.vox_list[i].xyz_max
        new_xyz_min = (
            torch.matmul(transforms_mat[:3, :3], xyz_min.reshape(3, 1))[..., 0]
            + transforms_mat[:3, -1]
        )
        new_xyz_max = (
            torch.matmul(transforms_mat[:3, :3], xyz_max.reshape(3, 1))[..., 0]
            + transforms_mat[:3, -1]
        )
        return new_xyz_min.detach().cpu().numpy(), new_xyz_max.detach().cpu().numpy()

    def sample_ray_bbox_intersect(self, rays_o, rays_d, xyz_min, xyz_max):
        _rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        vec = torch.where(_rays_d == 0, torch.full_like(_rays_d, 1e-6), _rays_d)
        rate_a = (xyz_max - rays_o) / vec
        rate_b = (xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1)
        t_max = torch.maximum(rate_a, rate_b).amin(-1)
        mask_outbbox = t_max <= t_min
        t_min[mask_outbbox] = 10086
        t_max[mask_outbbox] = -10086
        return t_min.view(-1), t_max.view(-1), mask_outbbox.view(-1)

    def sample_ray(
        self,
        beta,
        theta,
        trans,
        rays_o,
        rays_d,
        focals,
        device,
        vertices=None,
        obv2cnl_tfm=None,
        tpose_vertices=None,
    ):
        if obv2cnl_tfm is None:
            _theta = theta.reshape(1, 24, 3, 3)
            so = self.smpl_model(
                betas=beta.reshape(1, 10),
                body_pose=_theta[:, 1:],
                global_orient=_theta[:, 0].view(1, 1, 3, 3),
            )
            smpl_v = so["vertices"].clone().reshape(-1, 3)
            del so
            so_T = self.smpl_model(
                betas=beta.reshape(1, 10),
                body_pose=_theta[:, 1:] * 0 + torch.eye(3)[None, None, :, :].to(device),
                global_orient=_theta[:, 0].view(1, 1, 3, 3) * 0
                + torch.eye(3)[None, None, :, :].to(device),
            )
            smpl_v_T = so_T["vertices"].clone().reshape(-1, 3)
            del so_T
            init_J = get_J(beta.reshape(1, 10), self.smpl_model)

            _, rel_transforms = self.batch_rigid_transform(theta, init_J)
        else:
            rel_transforms = torch.inverse(obv2cnl_tfm)
            smpl_v = vertices.clone().reshape(-1, 3)
            smpl_v_T = tpose_vertices.clone().reshape(-1, 3)
            trans = torch.zeros((1, 3)).float().to(device)

        # first sample ray pts for each voxel
        mask_outbbox_list = []

        actual_vox_bbox = self.compute_actual_bbox(beta)

        t_min_list = []
        t_max_list = []
        mask_outbbox_list = []
        bbox_transformation_list = []

        for i, vox_i in enumerate(self.vox_index):
            if vox_i == 15:
                cur_transforms_mat = rel_transforms[0, vox_i]
            elif vox_i == 12:
                cur_transforms_mat = rel_transforms[0, 6]
            else:
                cur_transforms_mat = rel_transforms[0, self.parents[vox_i]]

            rays_o_local, rays_d_local = self.transform_to_vox_local(
                rays_o, rays_d, cur_transforms_mat, trans
            )
            bbox_transformation_list.append(cur_transforms_mat)

            cur_xyz_min, cur_xyz_max = actual_vox_bbox[i]
            cur_t_min, cur_t_max, cur_mask_outbbox = self.sample_ray_bbox_intersect(
                rays_o_local, rays_d_local, cur_xyz_min, cur_xyz_max
            )
            t_min_list.append(cur_t_min)
            t_max_list.append(cur_t_max)
            mask_outbbox_list.append(cur_mask_outbbox)

        ### cumulate t_min, t_max, mask_oubbox for all vox
        all_t_min = torch.stack(t_min_list, -1)
        all_t_max = torch.stack(t_max_list, -1)
        all_mask_outbbox = torch.stack(mask_outbbox_list, -1)
        t_min = torch.min(all_t_min, -1)[0]
        t_max = torch.max(all_t_max, -1)[0]
        # st()
        mask_outbbox = torch.all(all_mask_outbbox, -1)
        assert torch.all((t_min == 10086) == (t_max == -10086))
        assert torch.all((t_min == 10086) == mask_outbbox)

        valid_t_min = t_min[~mask_outbbox].view(-1, 1)
        valid_t_max = t_max[~mask_outbbox].view(-1, 1)
        z_vals = valid_t_min * (
            1.0 - self.t_vals.view(1, -1)
        ) + valid_t_max * self.t_vals.view(1, -1)
        if self.perturb > 0:
            upper = torch.cat([z_vals[..., 1:], valid_t_max], -1)
            lower = z_vals.detach()
            t_rand = torch.rand(*z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand
        _rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_pts_global = rays_o[~mask_outbbox].unsqueeze(1) + _rays_d[
            ~mask_outbbox
        ].unsqueeze(1) * z_vals.view(-1, self.N_samples, 1)
        if self.multiple_sample:
            # zone_radius = 1*z_vals/focals[0]
            rays_pts_global_ori = rays_pts_global.clone()
            if len(focals.shape) != 3:
                zone_radius = 1 * rays_pts_global[:, :, -1] / focals[0] / 2
            else:
                zone_radius = 1 * rays_pts_global[:, :, -1] / focals[0, 0, 0] / 2

            self.sample_num = 5
            zone_radius = zone_radius[..., None]
            coefficient = (
                torch.tensor([[0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0]])
                .float()
                .to(device)
            )
            sample_points = (
                coefficient[None, None] * zone_radius[..., None]
                + rays_pts_global_ori[:, :, None, :]
            )
            rays_pts_global = torch.cat(
                [rays_pts_global_ori[:, :, None, :], sample_points], dim=-2
            )
        else:
            rays_pts_global = rays_pts_global[:, :, None, :]
            self.sample_num = 1

        smpl_v_w = smpl_v
        if obv2cnl_tfm is None:
            rel_transforms[:, :, :3, 3] += trans
            smpl_v_w = smpl_v + trans

        meshes = Meshes(
            verts=smpl_v_T[None],
            faces=torch.from_numpy(self.smpl_model.faces[None].astype(np.float32)).to(
                device
            ),
        )
        self.tpose_verts_normals = meshes.verts_normals_packed()

        # rotation of local coordinate system under the T pose to world coordinate system
        eps = 1e-5
        up = [[0, 0, 1]]
        z_axis = self.tpose_verts_normals
        z_axis_num = z_axis.shape[0]
        eps = torch.tensor([[eps]]).repeat(z_axis_num, 1).type(torch.float32).to(device)
        up = torch.tensor(up).repeat(z_axis_num, 1).type(torch.float32).to(device)
        x_axis = torch.cross(up, z_axis)
        x_axis /= torch.max(
            torch.stack([torch.norm(x_axis, dim=1, keepdim=True), eps]), dim=0
        )[0]
        y_axis = torch.cross(z_axis, x_axis)
        y_axis /= torch.max(
            torch.stack([torch.norm(y_axis, dim=1, keepdim=True), eps]), dim=0
        )[0]
        local2world_Tpose = torch.cat(
            (
                x_axis.reshape(-1, 3, 1),
                y_axis.reshape(-1, 3, 1),
                z_axis.reshape(-1, 3, 1),
            ),
            dim=2,
        )

        return (
            rays_pts_global,
            smpl_v_w,
            z_vals,
            rel_transforms,
            local2world_Tpose,
            mask_outbbox,
        )

    def get_rays(self, focal, c2w):
        dirs = torch.stack(
            [
                (self.i - self.out_im_res[1] * 0.5) / focal[0],
                (self.j - self.out_im_res[0] * 0.5) / focal[0],
                torch.ones_like(self.i).expand(
                    1, self.out_im_res[0], self.out_im_res[1]
                ),
            ],
            -1,
        ).repeat(focal.shape[0], 1, 1, 1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(
            dirs[..., None, :] * c2w[:, None, None, :3, :3], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)
        viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def get_rays_from_KE(self, intrinsic, c2w):
        dirs = torch.stack(
            [
                (self.i - intrinsic[0, 0, 2]) / intrinsic[0, 0, 0],
                (self.j - intrinsic[0, 1, 2]) / intrinsic[0, 1, 1],
                torch.ones_like(self.i).expand(
                    1, self.out_im_res[0], self.out_im_res[1]
                ),
            ],
            -1,
        ).repeat(intrinsic.shape[0], 1, 1, 1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(
            dirs[..., None, :] * c2w[:, None, None, :3, :3], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)
        viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def get_vertices_feature(self, styles, truncation=1):
        uv_feature, ws = self.encoder(styles, None, truncation_psi=truncation)
        uv_feature = uv_feature[-1]
        vetidx_to_uvidx = self.vetidx_to_uvidx.type(torch.long).to(uv_feature.device)
        uvidx_to_uvcoor = self.uvidx_to_uvcoor.type(torch.float32).to(uv_feature.device)

        uv_indices = vetidx_to_uvidx[:, : self.uv_num]
        uv_coordinates = uvidx_to_uvcoor[uv_indices.reshape(-1)]
        uv_coordinates[:, 1] = 1 - uv_coordinates[:, 1]
        vertices_feature = F.grid_sample(
            uv_feature, uv_coordinates[None, :, None, :] * 2 - 1
        )
        return vertices_feature

    def forward(
        self,
        cam_poses,
        focals,
        beta,
        theta,
        trans,
        styles=None,
        vertices_feature=None,
        truncation=1,
        vertices=None,
        obv2cnl_tfm=None,
        tpose_vertices=None,
    ):
        device = cam_poses.device
        batch_size = cam_poses.shape[0]
        beta = beta[:1]
        if theta is not None:
            theta = theta[:1]
            trans = trans[:1]
            theta_rodrigues = batch_rodrigues(theta.reshape(-1, 3)).reshape(1, 24, 3, 3)
        else:
            theta_rodrigues = None

        # get rays
        if len(focals.shape) != 3:
            rays_o, rays_d, viewdirs = self.get_rays(focals, cam_poses)
        else:
            rays_o, rays_d, viewdirs = self.get_rays_from_KE(focals, cam_poses)
        rays_o = rays_o.reshape(batch_size, -1, 3)
        rays_d = rays_d.reshape(batch_size, -1, 3)

        # sample points on rays
        with torch.no_grad():
            (
                rays_pts_global,
                smpl_verts,
                z_vals,
                rel_transforms,
                local2world_Tpose,
                mask_outbbox,
            ) = self.sample_ray(
                beta,
                theta_rodrigues,
                trans,
                rays_o[0],
                rays_d[0],
                focals,
                device,
                vertices=vertices,
                obv2cnl_tfm=obv2cnl_tfm,
                tpose_vertices=tpose_vertices,
            )

        skip_dist = self.opt.skip_dist**2
        batch_index = 0

        # generate vertices features
        if vertices_feature is None:
            vertices_feature = self.get_vertices_feature(styles, truncation)

        rays_pts_global_all = rays_pts_global.clone()
        raw = torch.zeros_like(rays_pts_global[..., 0, 0]).unsqueeze(-1).repeat(1, 1, 4)
        for j in range(self.sample_num):
            rays_pts_global = rays_pts_global_all[:, :, j, :]

            chunk = rays_pts_global.shape[0]
            for i in range(0, rays_pts_global.shape[0], chunk):
                pts = rays_pts_global[i * chunk : (i + 1) * chunk].reshape(-1, 3)
                if pts.shape[0] == 0:
                    continue

                # find nearest neighbors
                with torch.no_grad():
                    nn = knn_points(
                        pts[None], smpl_verts.reshape(1, -1, 3), K=self.mean_neighbor
                    )
                    dists = nn.dists[0]
                    indices = nn.idx[0]

                # skip empty area
                cur_raw = torch.zeros_like(pts[..., 0]).unsqueeze(-1).repeat(1, 4)
                valid = dists[:, 0] < skip_dist
                dists = dists[valid]
                indices = indices[valid]
                pts = pts[valid]

                batch_size = vertices_feature.shape[0]
                pts_num = pts.shape[0]

                # neighbor vertices feature
                vertices_feature_channel = vertices_feature.shape[1]
                neighbor_vertices_feature = (
                    vertices_feature[batch_index]
                    .permute(1, 0, 2)[indices.reshape(-1)]
                    .permute(1, 0, 2)
                )
                neighbor_vertices_feature = neighbor_vertices_feature.reshape(
                    vertices_feature_channel, -1, self.mean_neighbor
                )

                # distance statistics
                mean_dists = dists.mean(dim=-1, keepdim=True)
                max_dists = dists.max(dim=-1, keepdim=True)[0]
                min_dists = dists.min(dim=-1, keepdim=True)[0]
                if self.mean_neighbor == 1:
                    var_dists = dists * 0
                else:
                    var_dists = dists.var(dim=-1, keepdim=True)

                # transform to local coordinate system
                neighbor_smpl_verts = smpl_verts[indices.reshape(-1)].reshape(
                    batch_size, pts_num, self.mean_neighbor, -1
                )
                offset_dir_w = pts[:, None, :] - neighbor_smpl_verts[batch_index, :]
                cnl2obv_tfm = (
                    rel_transforms[batch_index, :, :3, :3][None]
                    .mul(self.smpl_model.lbs_weights[:, :, None, None])
                    .sum(dim=1, keepdim=False)
                )  # rotation
                local2world = torch.matmul(cnl2obv_tfm, local2world_Tpose)
                world2local = torch.inverse(local2world)[indices.reshape(-1)]
                offset_dir_l = torch.einsum(
                    "bij,bj->bi", world2local, offset_dir_w.reshape(-1, 3)
                )
                offset_dir_l = offset_dir_l.reshape(-1, self.mean_neighbor, 3)
                offset_dir_l = offset_dir_l / (
                    torch.norm(offset_dir_l, dim=-1, keepdim=True) + 1e-6
                )
                mean_offset_dir_l = offset_dir_l.mean(dim=1, keepdim=False)

                # weighted neighbor vertices features
                dists = torch.clamp(dists, 0.0001)
                ws = 1.0 / dists
                ws = ws / ws.sum(-1, keepdim=True)
                weighted_vertices_feature = (
                    (neighbor_vertices_feature * ws).sum(-1).permute(1, 0)
                )

                # local coordinate information: 3-dimensional averaged directions and 4-dimensional norm statistics
                local_coord_info = torch.cat(
                    [mean_dists, max_dists, min_dists, var_dists, mean_offset_dir_l],
                    dim=-1,
                )
                local_coord_info = self.pos_embed_fn_nerf(local_coord_info)

                # feature used to predict density and color
                embedded_feature = torch.cat(
                    [weighted_vertices_feature, local_coord_info], dim=-1
                )
                embedded_feature = torch.reshape(
                    embedded_feature, [-1, embedded_feature.shape[-1]]
                )

                # predict density & color
                raw_valid = self.cnl_mlp(embedded_feature)
                cur_raw[valid] = raw_valid
                raw[i * chunk : (i + 1) * chunk] += cur_raw.view(-1, self.N_samples, 4)

        raw = raw / (self.sample_num)

        rgb_map, xyz, depth, mask, rgba_map = self.volume_rendering(
            raw, z_vals, mask_outbbox, rays_pts_global, batch_size, device
        )

        rgb_map = -1 + 2 * rgb_map

        return rgb_map, mask, [xyz, depth], smpl_verts, rgba_map

    def volume_rendering(
        self, raw, z_vals, mask_outbbox, rays_pts_global, batch_size, device
    ):
        valid_rays_num = rays_pts_global.shape[0]
        rays_pts_global = rays_pts_global.repeat(batch_size, 1, 1)
        mask_outbbox = mask_outbbox.repeat(batch_size)

        z_vals = z_vals.repeat(batch_size, 1)
        dists = (z_vals[..., 1:] - z_vals[..., :-1]).reshape(
            batch_size, valid_rays_num, -1
        )
        dists = torch.cat([dists, dists[..., -1:]], -1)
        sigma = raw[..., -1]
        # Render All
        sigma = 1 - torch.exp(-F.relu(sigma) * dists)

        visibility = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(torch.index_select(sigma, 2, self.zero_idx)),
                    1.0 - sigma + 1e-10,
                ],
                2,
            ),
            2,
        )
        visibility = visibility[..., :-1]
        _weights = sigma * visibility

        _rgb_map = torch.sum(
            _weights.unsqueeze(-1)
            * torch.sigmoid(raw[..., :3].view(batch_size, valid_rays_num, -1, 3)),
            2,
        )
        _xyz_map = torch.sum(
            _weights.unsqueeze(-1)
            * rays_pts_global.view(batch_size, valid_rays_num, -1, 3),
            2,
        )
        _weights_normalized = _weights / torch.sum(
            _weights + 1e-10, dim=-1, keepdim=True
        )
        _depth_map = torch.sum(
            _weights_normalized * z_vals.view(batch_size, valid_rays_num, -1), -1
        )
        # _depth_map = torch.sum(_weights * z_vals.view(batch_size, valid_rays_num, -1), -1)
        _weights_map = _weights.sum(dim=-1, keepdim=True)

        H, W = self.out_im_res

        rgb_map = torch.zeros(batch_size * H * W, 3).to(device)
        if self.opt.white_bg:
            rgb_map += 1
            _rgb_map = _rgb_map + 1 - _weights_map.view(batch_size, valid_rays_num, 1)

        rgb_map[~mask_outbbox] = _rgb_map.view(-1, 3)
        rgb_map = rgb_map.view(batch_size, H, W, 3)

        xyz_map = torch.zeros(batch_size * H * W, 3).to(device)
        xyz_map[~mask_outbbox] = _xyz_map.view(-1, 3)
        xyz = xyz_map.view(batch_size, H, W, 3)

        depth_map = torch.zeros(batch_size * H * W, 1).to(device) + _depth_map.max()
        depth_map[~mask_outbbox] = _depth_map.view(-1, 1)
        depth = depth_map.view(batch_size, H, W)

        weights_map = torch.zeros(batch_size * H * W, 1).to(device)
        weights_map[~mask_outbbox] = _weights_map.view(-1, 1)
        mask = weights_map.view(batch_size, H, W)

        rgba_map = torch.cat([rgb_map, mask[..., None]], dim=-1)

        if torch.any(torch.isnan(rgb_map)):
            st()
        return rgb_map, xyz, depth, mask, rgba_map
