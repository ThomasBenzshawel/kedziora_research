import gc
import importlib
from contextlib import contextmanager
import os

import fvdb
from fvdb.nn import VDBTensor
from fvdb import GridBatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import collections
from pathlib import Path
from pytorch_lightning.utilities.distributed import rank_zero_only


from xcube_refactored.utils import exp
from xcube_refactored.utils.vis_util import vis_pcs


from xcube_refactored.modules.diffusionmodules.schedulers.scheduling_ddim import DDIMScheduler
from xcube_refactored.modules.diffusionmodules.schedulers.scheduling_ddpm import DDPMScheduler
from xcube_refactored.modules.diffusionmodules.schedulers.scheduling_dpmpp_2m import DPMSolverMultistepScheduler


from xcube_refactored.modules.diffusionmodules.ema import LitEma


# Why aren't these used??????
from xcube_refactored.modules.diffusionmodules.openaimodel.unet_dense import UNetModel as UNetModel_Dense
from xcube_refactored.modules.diffusionmodules.openaimodel.unet_sparse import UNetModel as UNetModel_Sparse
from xcube_refactored.modules.diffusionmodules.openaimodel.unet_sparse_crossattn import UNetModel as UNetModel_Sparse_CrossAttn


# Why aren't these used??????
from xcube_refactored.modules.encoders import (SemanticEncoder, ClassEmbedder, PointNetEncoder,
                                    StructEncoder, StructEncoder3D, StructEncoder3D_remain_h, StructEncoder3D_v2)

class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.best_metrics = AverageMeter()
        # step -> log_name -> log_value (list of ordered-dict)
        self.test_logged_values = []
        self.record_folder = None
        self.record_headers = []
        self.record_data_cache = {}
        self.last_test_valid = False
        self.render_backend = 'pyrender'
        self.num_oom = 0

    @staticmethod
    def load_module(spec_path, weight_path=None, overwrite_config=None):
        """
        Load a module given spec_path
        :param spec_path: Path to a model yaml file or ckpt. If is a ckpt file, then weight will also be loaded.
        :param weight_path: Path to the model weight. If explicitly set to 'NO_LOAD', then even if ckpt is provided to
            spec_path, no weights will be loaded into the model.
        :param overwrite_config: argparse.Namespace object, if you want to overwrite the original config.
        :return: the module class, possibly with weight loaded.
        """
        if spec_path is not None:
            spec_path = Path(spec_path)
            if spec_path.suffix == ".ckpt":
                # Boil down spec path using glob.
                import glob2
                possible_paths = glob2.glob(str(spec_path))
                if len(possible_paths) == 1:
                    spec_path = Path(possible_paths[0])
                else:
                    raise AssertionError
                config_yaml_path = spec_path.parent.parent / "hparams.yaml"
                if weight_path == "NO_LOAD":
                    weight_path = None
                elif weight_path is None:
                    weight_path = spec_path
            elif spec_path.suffix == ".yaml":
                config_yaml_path = spec_path
            else:
                raise NotImplementedError

            config_args = exp.parse_config_yaml(config_yaml_path, overwrite_config, override=False)
        else:
            config_args = overwrite_config

        if "model" not in config_args.keys():
            print("No model found.")
            return None

        basis_net_module = importlib.import_module("xcube.models." + config_args.model).Model

        if weight_path is not None:
            net_module = basis_net_module.load_from_checkpoint(weight_path, hparams=config_args)
        else:
            net_module = basis_net_module(config_args)

        return net_module

    def training_step(self, *args, **kwargs):
        try:
            return self.train_val_step(is_val=False, *args, **kwargs)
        except RuntimeError:
            # Compare to post-mortem, this would allow training to continue...
            exp.logger.warning(f"Training-step OOM. Skipping.")

            try:
                from xcube.data.base import DatasetSpec as DS
                exp.logger.warning(f"The problematic batch is: {args[0][DS.SHAPE_NAME]}")
            except:
                pass

            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            self.num_oom += 1.0
            self.log("num_oom", self.num_oom)
            return None



    def train_dataloader(self):
        # Note:
        import xcube.data as dataset
        train_set = dataset.build_dataset(
            self.hparams.train_dataset, self.get_dataset_spec(), self.hparams, self.hparams.train_kwargs)
        torch.manual_seed(0)
        return DataLoader(train_set, batch_size=self.hparams.batch_size // self.trainer.world_size, shuffle=True,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        import xcube.data as dataset
        val_set = dataset.build_dataset(
            self.hparams.val_dataset, self.get_dataset_spec(), self.hparams, self.hparams.val_kwargs)
        return DataLoader(val_set, batch_size=self.hparams.batch_size // self.trainer.world_size, shuffle=False,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def test_dataloader(self):
        import xcube.data as dataset
        self.hparams.test_kwargs.resolution = self.hparams.resolution # ! use for testing when training on X^3 but testing on Y^3

        test_set = dataset.build_dataset(
            self.hparams.test_dataset, self.get_dataset_spec(), self.hparams, self.hparams.test_kwargs)
        if self.hparams.test_set_shuffle:
            torch.manual_seed(0)
        return DataLoader(test_set, batch_size=1, shuffle=self.hparams.test_set_shuffle, 
                          num_workers=0, collate_fn=self.get_collate_fn())
    
class Embedder(nn.Module):
    def __init__(self, include_input=True, input_dims=3, max_freq_log2=10, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        super().__init__()
        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            out_dim += d
            
        max_freq = max_freq_log2
        N_freqs = num_freqs
        
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for _ in periodic_fns:
                out_dim += d
        
        self.include_input = include_input
        self.freq_bands = freq_bands
        self.periodic_fns = periodic_fns
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def forward(self, inputs):
        output_list = [inputs]
        for fn in self.embed_fns:
            output_list.append(fn(inputs))
            
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                output_list.append(p_fn(inputs * freq))
        
        return torch.cat(output_list, -1)

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embedder_obj = Embedder(max_freq_log2=multires-1, num_freqs=multires, input_dims=input_dims)
    return embedder_obj, embedder_obj.out_dim

class DatasetSpec(Enum):
    SHAPE_NAME = 100
    INPUT_PC = 200
    TARGET_NORMAL = 300
    INPUT_COLOR = 350
    INPUT_INTENSITY = 360
    GT_DENSE_PC = 400
    GT_DENSE_NORMAL = 500
    GT_DENSE_COLOR = 550
    GT_MESH = 600
    GT_MESH_SOUP = 650
    GT_ONET_SAMPLE = 700
    GT_GEOMETRY = 800
    DATASET_CFG = 1000
    GT_DYN_FLAG = 1100
    GT_SEMANTIC = 1200
    LATENT_SEMANTIC = 1300
    SINGLE_SCAN_CROP = 1400
    SINGLE_SCAN_INTENSITY_CROP = 1410
    SINGLE_SCAN = 1450
    SINGLE_SCAN_INTENSITY = 1460
    CLASS = 1500
    TEXT_EMBEDDING = 1600
    TEXT_EMBEDDING_MASK = 1610
    TEXT = 1620
    MICRO = 1630

def list_collate(batch):
    """
    This just do not stack batch dimension.
    """
    from fvdb import GridBatch, JaggedTensor

    elem = None
    for e in batch:
        if e is not None:
            elem = e
            break
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return list_collate([torch.as_tensor(b) if b is not None else None for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, DictConfig) or isinstance(elem, ListConfig):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    elif isinstance(elem, GridBatch):
        return fvdb.cat(batch)
    
    # elif isinstance(elem, pathlib.Path):
    #     return batch
    # elif elem is None:
    #     return batch

    # raise NotImplementedError
    return batch


def is_rank_node_zero():
    return os.environ.get('NODE_RANK', '0') == '0'

class Model(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        if not hasattr(self.hparams, 'ema'):
            self.hparams.ema = False
        if not hasattr(self.hparams, 'use_ddim'):
            self.hparams.use_ddim = False
        if not hasattr(self.hparams, 'scale_by_std'):
            self.hparams.scale_by_std = False
        if not hasattr(self.hparams, 'scale_factor'):
            self.hparams.scale_factor = 1.0
        if not hasattr(self.hparams, 'num_inference_steps'):
            self.hparams.num_inference_steps = 1000

        
        self.hparams.conditioning_key = "none"
        
        self.hparams.log_image = True

        # position embedding
        self.hparams.use_pos_embed = False
        self.hparams.use_pos_embed_high = False
        if not hasattr(self.hparams, 'use_pos_embed_world'):
            self.hparams.use_pos_embed_world = False
        self.hparams.use_pos_embed_world_high = False

        # get vae model vae and unet depend on eachother
        model_yaml_path = Path(self.hparams.vae_config)
        model_args = exp.parse_config_yaml(model_yaml_path)
        net_module = importlib.import_module("xcube.models." + model_args.model).Model
        args_ckpt = Path(self.hparams.vae_checkpoint)
        assert args_ckpt.exists(), "Selected VAE checkpoint does not exist!"
        self.vae = net_module.load_from_checkpoint(args_ckpt, hparams=model_args).eval()
        self.vae.requires_grad_(False)

        # setup diffusion unet (Uses the same num of blocks and f_maps as the VAE)
        unet_num_blocks = self.vae.hparams.network.unet.params.num_blocks
        num_input_channels = self.vae.hparams.network.unet.params.f_maps * 2 ** (unet_num_blocks - 1) # Fix by using VAE hparams
        num_input_channels = int(num_input_channels / self.vae.hparams.cut_ratio)

        out_channels = num_input_channels
        num_classes = None
        use_spatial_transformer = False
        context_dim=None
        concat_dim=None
    
        if self.hparams.use_pos_embed:
            num_input_channels += 3
        elif self.hparams.use_pos_embed_high:
            embed_fn, input_ch = get_embedder(5)
            self.pos_embedder = embed_fn
            num_input_channels += input_ch
        elif self.hparams.use_pos_embed_world:
            num_input_channels += 3
        elif self.hparams.use_pos_embed_world_high:
            embed_fn, input_ch = get_embedder(5)
            self.pos_embedder = embed_fn
            num_input_channels += input_ch

        # eval(NAME) can be unet sparse, unet dense, unet sparse crossattn, and attention dense/sparse
        self.unet = eval(self.hparams.network.diffuser_name)(num_input_channels=num_input_channels, 
                                                             out_channels=out_channels, 
                                                             num_classes=num_classes,
                                                             use_spatial_transformer=use_spatial_transformer,
                                                             context_dim=context_dim,
                                                             **self.hparams.network.diffuser)
                
        # get the schedulers # important for getting noise for diffusion
        self.noise_scheduler = DDPMScheduler(**self.hparams.network.scheduler)
        self.ddim_scheduler = DDIMScheduler(**self.hparams.network.scheduler)

        # setup diffusion condition
        self.hparams.use_mask_cond = False
        self.hparams.use_point_cond = False
        self.hparams.use_semantic_cond = False
        self.hparams.use_normal_concat_cond = False 
            
        self.hparams.use_single_scan_concat_cond = False
        self.hparams.encode_single_scan_by_points = False
        
        self.hparams.use_class_cond = False
        self.hparams.use_micro_cond = False
        self.hparams.use_text_cond = False

        self.hparams.use_noise_offset = False
            
        # classifier-free config
        self.hparams.use_classifier_free = False # text cond in not influenced by this flag
        self.hparams.classifier_free_prob = 0.1 # prob to drop the label
            
        # finetune config
        if not hasattr(self.hparams, 'pretrained_model_name_or_path'):
            self.hparams.pretrained_model_name_or_path = None
        if not hasattr(self.hparams, 'ignore_mismatched_size'):
            self.hparams.ignore_mismatched_size = False

        # mask or point or semantic condition 
        # Never used
        if self.hparams.use_mask_cond or self.hparams.use_point_cond or self.hparams.use_semantic_cond or self.hparams.use_class_cond:
            self.cond_stage_model = eval(self.hparams.network.cond_stage_model.target)(**self.hparams.network.cond_stage_model.params)
        
    
        # single scan concat condition
    
        # load pretrained unet weight (ema version)
        # build ema
        if self.hparams.ema:
            self.unet_ema = LitEma(self.unet, decay=self.hparams.ema_decay)
            
        # scale by std
        if not self.hparams.scale_by_std:
            self.scale_factor = self.hparams.scale_factor
            assert self.scale_factor == 1., 'when not using scale_by_std, scale_factor should be 1.'
        else:
            self.register_buffer('scale_factor', torch.tensor(self.hparams.scale_factor).float())


# End HPARAMS and INIT ____________________________________________________________________________________-

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.hparams.scale_by_std and self.global_step == 0 and batch_idx == 0 and self.scale_factor == 1.:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            latents = self.extract_latent(batch)
            z = latents.feature.jdata.detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())

    def on_train_batch_end(self, *args, **kwargs):
        if self.hparams.ema:
            self.unet_ema(self.unet)

    @contextmanager
    def ema_scope(self):
        if self.hparams.ema:
            self.unet_ema.store(self.unet.parameters())
            self.unet_ema.copy_to(self.unet)
        try:
            yield None
        finally:
            if self.hparams.ema:
                self.unet_ema.restore(self.unet.parameters())
                
    def get_pos_embed(self, h):
        return h[:, :3]
    
    def get_pos_embed_high(self, h):
        xyz = h[:, :3] # N, 3
        xyz = self.pos_embedder(xyz) # N, C
        return xyz
    
    def conduct_classifier_free(self, cond, batch_size, device, is_testing=False):
        if isinstance(cond, VDBTensor):
            cond = cond.feature
        assert isinstance(cond, fvdb.JaggedTensor), "cond should be JaggedTensor"

        mask = torch.rand(batch_size, device=device) < self.hparams.classifier_free_prob 
        new_cond = []
        for idx in range(batch_size):
            if mask[idx] or is_testing:
                # during testing, use this function to zero the condition
                new_cond.append(torch.zeros_like(cond[idx].jdata))
            else:
                new_cond.append(cond[idx].jdata)
        new_cond = fvdb.JaggedTensor(new_cond)
        return new_cond
    

    @exp.mem_profile(every=1)
    def forward(self, batch, out: dict):


        # first get latent from vae, the latent is the input to the diffusion model
        # A latent is the encoded feature from the input
        with torch.no_grad():
            latents = self.vae._encode(batch, use_mode=False)

        # To Do: scale the latent
        if self.hparams.scale_by_std:
            latents = latents * self.scale_factor

        # then get the noise
        latent_data = latents.feature.jdata
        noise = torch.randn_like(latent_data) # N, C

        # bsz is the batch size ???? TODO
        bsz = latents.grid.grid_count
        
        # Sample a random timestep for each latent
        # A timestep is a random point in the training schedule

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # B
        timesteps_sparse = timesteps.long()
        timesteps_sparse = timesteps_sparse[latents.feature.jidx.long()] # N, 1

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latent_data, noise, timesteps_sparse)
        
        # Predict the target for the noise residual (this is the backward diffusion process for training)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        # Currently Used ------------------ Very Important ------------------
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent_data, noise, timesteps_sparse)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        # forward_cond function use batch-level timesteps
        noisy_latents = VDBTensor(grid=latents.grid, feature=latents.grid.jagged_like(noisy_latents))


        cond_dict = None
        is_testing=False
        guidance_scale=1.0
   
        # Classifier free guidance is when the model is trained with a classifier, 
        # but at inference time, the classifier is not used
        do_classifier_free_guidance = guidance_scale != 1.0

        # ! corssattn part
        # text condition
        if self.hparams.use_text_cond:
            # traing-time: get text from batch
            if batch is not None:
                text_emb = torch.stack(batch[DatasetSpec.TEXT_EMBEDDING]) # B, 77, 1024
                mask = torch.stack(batch[DatasetSpec.TEXT_EMBEDDING_MASK]) # B, 77
            else:
                text_emb = cond_dict['text_emb']
                mask = cond_dict['text_emb_mask']                
            context = text_emb
            if do_classifier_free_guidance:
                context_copy = cond_dict['text_emb_null']
                mask_copy = cond_dict['text_emb_mask_null']

        
        # ! concat part            
        concat_list = []        
        # semantic condition
        if self.hparams.use_semantic_cond:
            # traing-time: get semantic from batch
            if batch is not None:
                input_semantic = fvdb.JaggedTensor(batch[DatasetSpec.LATENT_SEMANTIC])
            else:
                input_semantic = cond_dict['semantics']
            semantic_cond = self.cond_stage_model(input_semantic.jdata.long())
            if not is_testing and self.hparams.use_classifier_free: # if VDBtensor, convert to JaggedTensor
                semantic_cond = self.conduct_classifier_free(semantic_cond, noisy_latents.grid.grid_count, noisy_latents.grid.device)  
            concat_list.append(semantic_cond) # ! tensor type
        
        # ! corssattn part
        # text condition
        if self.hparams.use_text_cond:
            # traing-time: get text from batch
            if batch is not None:
                text_emb = torch.stack(batch[DatasetSpec.TEXT_EMBEDDING]) # B, 77, 1024
                mask = torch.stack(batch[DatasetSpec.TEXT_EMBEDDING_MASK]) # B, 77
            else:
                text_emb = cond_dict['text_emb']
                mask = cond_dict['text_emb_mask']                
            context = text_emb
            if do_classifier_free_guidance:
                context_copy = cond_dict['text_emb_null']
                mask_copy = cond_dict['text_emb_mask_null']

    
        if self.hparams.conditioning_key == 'none':
            # no condition is used -------------------- VERY IMPORTANT --------------------
            model_pred = self.unet(noisy_latents, timesteps)
        elif self.hparams.conditioning_key == 'c_crossattn':
            assert len(concat_list) > 0, "concat_list should not be empty"
            assert context is not None, "context should not be None"
            noisy_latents_in = VDBTensor.cat([noisy_latents] + concat_list, dim=1)
            model_pred = self.unet(noisy_latents_in, timesteps, context=context, mask=mask)
            
        else:
            raise NotImplementedError
        

        out.update({'pred': model_pred.feature.jdata})
        out.update({'target': target})

        return out
    

    # Pcs stands for point clouds ???
    def get_random_sample_pcs(self, ijk: fvdb.JaggedTensor, batch_size=1, M=3, use_center=False):
        # M: sample per point
        # !: be careful about the batch_size
        output_ijk = []
        for idx in range(batch_size):
            current_ijk = ijk[idx].jdata.float() # N, 3
            if use_center:
                output_ijk.append(current_ijk)
            else:            
                N = current_ijk.shape[0]
                # create offsets of size M*N x 3 with values in range [-0.5, 0.5]
                offsets = torch.FloatTensor(N * M, 3).uniform_(-0.5, 0.5).to(current_ijk.device)
                # duplicate your original point cloud M times
                expanded_point_cloud = current_ijk.repeat(M, 1)
                # add offsets to duplicated points
                expanded_point_cloud += offsets
                output_ijk.append(expanded_point_cloud)
        return fvdb.JaggedTensor(output_ijk)
            
    def decode_to_points(self, latents):
        res = self.vae.unet.FeaturesSet()
        res, output_x = self.vae.unet.decode(res, latents, is_testing=True)
        grid_tree = res.structure_grid
        fine_list = []
        coarse_list = []
        for batch_idx in range(output_x.grid.grid_count):
            # fineest level
            pd_grid_0 = grid_tree[0]
            pd_xyz_0 = pd_grid_0.grid_to_world(self.get_random_sample_pcs(pd_grid_0.ijk[batch_idx], M=8))
            fine_list.append(pd_xyz_0.jdata.cpu().numpy())
            # coarsest level
            pd_grid_1 = grid_tree[len(grid_tree.keys()) - 1]
            pd_xyz_1 = pd_grid_1.grid_to_world(self.get_random_sample_pcs(pd_grid_1.ijk[batch_idx], M=3))
            coarse_list.append(pd_xyz_1.jdata.cpu().numpy())
        # plot the fine and coarse level
        viz_fine = vis_pcs(fine_list)
        viz_coarse = vis_pcs(coarse_list)
        decode_results = np.concatenate([viz_fine, viz_coarse], axis=0)
        return decode_results
    
    @exp.mem_profile(every=1)
    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        # compute the MSE loss
        if self.hparams.supervision.mse_weight > 0.0:
            loss_dict.add_loss("mse", F.mse_loss(out["pred"], out["target"]), self.hparams.supervision.mse_weight)
        if compute_metric: # currently use MSE as metric
            metric_dict.add_loss("mse", F.mse_loss(out["pred"], out["target"]))

        return loss_dict, metric_dict 
    

    # This is an exaple of our training step ______________________________________________________________________________________
    def train_val_step(self, batch, batch_idx, is_val):
        if batch_idx % 100 == 0:
            # Squeeze memory really hard :)   # Why was this commented??????
            gc.collect()
            torch.cuda.empty_cache()

        out = {'idx': batch_idx}
        with exp.pt_profile_named("forward"):
            out = self(batch, out)

        if out is None and not is_val:
            return None

        # Compute metric in train would be the out['pred'] and out['target']
        loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=is_val)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)

            if self.hparams.log_image:
                cond_dict = {}
                
                if self.hparams.use_text_cond:
                    text_emb = torch.stack(batch[DatasetSpec.TEXT_EMBEDDING]) # B, 77, 1024
                    mask = torch.stack(batch[DatasetSpec.TEXT_EMBEDDING_MASK]) # B, 77
                    cond_dict['text_emb'] = text_emb
                    cond_dict['text_emb_mask'] = mask
                
                if self.trainer.global_rank == 0: # only log the image on rank 0
                    if batch_idx == 0:
                        with self.ema_scope("Plotting"):
                            # first extract latent

                            clean_latents = self.extract_latent(batch)
                            grids = clean_latents.grid

                            # sample latents
                            sample_latents = self.random_sample_latents(grids, use_ddim=self.hparams.use_ddim, ddim_step=100, cond_dict=cond_dict) # TODO: change this ddim_step to variable
                            
                            # decode clean latents first
                            decode_clean = self.decode_to_points(clean_latents)
                            # Decode sample latents
                            decode_sample = self.decode_to_points(sample_latents)
                            sample = np.concatenate([decode_clean, decode_sample], axis=0)
                            self.log_image("img/sample", sample)
                            # clean matplotlib opens
                            plt.close('all')
                else:
                    if batch_idx == 0:
                        clean_latents = self.extract_latent(batch)
                        grids = clean_latents.grid
                        _ = self.random_sample_latents(grids, use_ddim=self.hparams.use_ddim, ddim_step=100, cond_dict=cond_dict) # TODO: change this ddim_step to variable

        loss_sum = loss_dict.get_sum()
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)
        self.log('val_step', self.global_step)

        return loss_sum
    
    @torch.no_grad()
    def extract_latent(self, batch):
        return self.vae._encode(batch, use_mode=False)
    

    def _forward_cond(self, noisy_latents: VDBTensor, timesteps: torch.Tensor, 
                      batch = None, cond_dict = None, is_testing=False, guidance_scale=1.0) -> VDBTensor:
        do_classifier_free_guidance = guidance_scale != 1.0
        # ! adm part
        # mask condition
        if self.hparams.use_mask_cond:
            coords = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float())
            coords = VDBTensor(noisy_latents.grid, coords)
            cond = self.cond_stage_model(coords)
        # point condition
        if self.hparams.use_point_cond:
            coords = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float()) # JaggedTensor
            if self.hparams.network.cond_stage_model.use_normal:
                if batch is not None: # training-time: get normal from batch
                    ref_xyz = fvdb.JaggedTensor(batch[DS.INPUT_PC])
                    # splatting normal
                    input_normal = noisy_latents.grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
                    # normalize normal
                    input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan
            else:
                input_normal = None
            cond = self.cond_stage_model(coords, input_normal)
        # class condition:
        if self.hparams.use_class_cond:
            if batch is not None:
                cond = self.cond_stage_model(batch, key=DS.CLASS)
            else:
                cond = self.cond_stage_model(cond_dict, key="class") # not checked yet
        # micro condition
        if self.hparams.use_micro_cond:
            if batch is not None:
                micro = batch[DS.MICRO]
                micro = torch.stack(micro).float()
            else:
                micro = cond_dict['micro']
            micro = self.micro_pos_embedder(micro)
            cond = self.micro_cond_model(micro)
        
        # ! concat part            
        concat_list = []        
        # semantic condition
        if self.hparams.use_semantic_cond:
            # traing-time: get semantic from batch
            if batch is not None:
                input_semantic = fvdb.JaggedTensor(batch[DS.LATENT_SEMANTIC])
            else:
                input_semantic = cond_dict['semantics']
            semantic_cond = self.cond_stage_model(input_semantic.jdata.long())
            if not is_testing and self.hparams.use_classifier_free: # if VDBtensor, convert to JaggedTensor
                semantic_cond = self.conduct_classifier_free(semantic_cond, noisy_latents.grid.grid_count, noisy_latents.grid.device)  
            concat_list.append(semantic_cond) # ! tensor type
        # single scan concat condition
        if self.hparams.use_single_scan_concat_cond:
            # traing-time: get single scan crop from batch
            if batch is not None:
                single_scan = fvdb.JaggedTensor(batch[DS.SINGLE_SCAN_CROP])
                single_scan_intensity = fvdb.JaggedTensor(batch[DS.SINGLE_SCAN_INTENSITY_CROP])
            else:
                single_scan = cond_dict['single_scan']
                single_scan_intensity = cond_dict['single_scan_intensity']
                
            # here use splatting to build the single scan grid tree
            single_scan_hash_tree = self.vae.build_normal_hash_tree(single_scan)
            single_scan_grid = single_scan_hash_tree[0]            
            if self.hparams.encode_single_scan_by_points:
                single_scan_feat = self.single_scan_pos_embedder(single_scan, single_scan_intensity, single_scan_grid)
                single_scan_feat = VDBTensor(single_scan_grid, single_scan_feat)
            else:
                single_scan_coords = single_scan_grid.grid_to_world(single_scan_grid.ijk.float()).jdata
                single_scan_feat = self.single_scan_pos_embedder(single_scan_coords)
                single_scan_feat = VDBTensor(single_scan_grid, single_scan_grid.jagged_like(single_scan_feat))
            single_scan_cond = self.single_scan_cond_model(single_scan_feat, single_scan_hash_tree)
            # align this feature to the latent
            single_scan_cond = noisy_latents.grid.fill_to_grid(single_scan_cond.feature, single_scan_cond.grid, 0.0)
            if not is_testing and self.hparams.use_classifier_free:
                single_scan_cond = self.conduct_classifier_free(single_scan_cond, noisy_latents.grid.grid_count, noisy_latents.grid.device)             
            concat_list.append(single_scan_cond)
        if self.hparams.use_normal_concat_cond:
            # traing-time: get single scan crop from batch
            if batch is not None:
                # assert self.hparams.use_fvdb_loader is True, "use_fvdb_loader should be True for normal concat condition"
                ref_grid = fvdb.cat(batch[DS.INPUT_PC])    
                ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
                concat_normal = noisy_latents.grid.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
            else:
                concat_normal = cond_dict['normal']
            concat_normal.jdata /= (concat_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan
            if not is_testing and self.hparams.use_classifier_free:
                concat_normal = self.conduct_classifier_free(concat_normal, noisy_latents.grid.grid_count, noisy_latents.grid.device)            
            concat_list.append(concat_normal)

        if do_classifier_free_guidance and len(concat_list) > 0: # ! not tested yet
            if not self.hparams.use_classifier_free:
                logger.info("Applying classifier-free guidance without doing it for concat condition")
                concat_list_copy = concat_list
            else:
                # logger.info("Applying classifier-free guidance for concat condition")    
                # assert self.hparams.use_classifier_free, "do_classifier_free_guidance should be used with use_classifier_free"
                concat_list_copy = []
                for cond in concat_list:
                    cond = self.conduct_classifier_free(cond, noisy_latents.grid.grid_count, noisy_latents.grid.device, is_testing=True)
                    concat_list_copy.append(cond)
        
        # ! corssattn part
        # text condition
        if self.hparams.use_text_cond:
            # traing-time: get text from batch
            if batch is not None:
                text_emb = torch.stack(batch[DS.TEXT_EMBEDDING]) # B, 77, 1024
                mask = torch.stack(batch[DS.TEXT_EMBEDDING_MASK]) # B, 77
            else:
                text_emb = cond_dict['text_emb']
                mask = cond_dict['text_emb_mask']                
            context = text_emb
            if do_classifier_free_guidance:
                context_copy = cond_dict['text_emb_null']
                mask_copy = cond_dict['text_emb_mask_null']

        # concat pos_emb
        if self.hparams.use_pos_embed:
            pos_embed = noisy_latents.grid.ijk
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)
        elif self.hparams.use_pos_embed_high:
            pos_embed = self.get_pos_embed_high(noisy_latents.grid.ijk.jdata)
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)
        elif self.hparams.use_pos_embed_world:
            pos_embed = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float())
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)
        elif self.hparams.use_pos_embed_world_high:
            pos_embed = noisy_latents.grid.grid_to_world(noisy_latents.grid.ijk.float())
            pos_embed = self.get_pos_embed_high(pos_embed.jdata)
            noisy_latents = VDBTensor.cat([noisy_latents, pos_embed], dim=1)

        if self.hparams.conditioning_key == 'none':
            model_pred = self.unet(noisy_latents, timesteps)
        elif self.hparams.conditioning_key == 'concat':
            assert len(concat_list) > 0, "concat_list should not be empty"
            noisy_latents_in = VDBTensor.cat([noisy_latents] + concat_list, dim=1)
            model_pred = self.unet(noisy_latents_in, timesteps)
            
            if do_classifier_free_guidance:
                noisy_latents_in_copy = VDBTensor.cat([noisy_latents] + concat_list_copy, dim=1)
                model_pred_copy = self.unet(noisy_latents_in_copy, timesteps)
                model_pred = VDBTensor(model_pred.grid, model_pred.grid.jagged_like(model_pred.feature.jdata + guidance_scale * (model_pred.feature.jdata - model_pred_copy.feature.jdata)))
        elif self.hparams.conditioning_key == 'adm':
            assert cond is not None, "cond should not be None"
            model_pred = self.unet(noisy_latents, timesteps, y=cond)
        elif self.hparams.conditioning_key == 'crossattn':
            assert context is not None, "context should not be None"
            model_pred = self.unet(noisy_latents, timesteps, context=context, mask=mask)
            
            if do_classifier_free_guidance:
                model_pred_copy = self.unet(noisy_latents, timesteps, context=context_copy, mask=mask_copy)
                model_pred = VDBTensor(model_pred.grid, model_pred.grid.jagged_like(model_pred.feature.jdata + guidance_scale * (model_pred.feature.jdata - model_pred_copy.feature.jdata)))
        elif self.hparams.conditioning_key == 'c_crossattn':
            assert len(concat_list) > 0, "concat_list should not be empty"
            assert context is not None, "context should not be None"
            noisy_latents_in = VDBTensor.cat([noisy_latents] + concat_list, dim=1)
            model_pred = self.unet(noisy_latents_in, timesteps, context=context, mask=mask)
            
            if do_classifier_free_guidance:
                noisy_latents_in_copy = VDBTensor.cat([noisy_latents] + concat_list_copy, dim=1)
                model_pred_copy = self.unet(noisy_latents_in_copy, timesteps, context=context_copy, mask=mask_copy)
                model_pred = VDBTensor(model_pred.grid, model_pred.grid.jagged_like(model_pred.feature.jdata + guidance_scale * (model_pred.feature.jdata - model_pred_copy.feature.jdata)))
        else:
            raise NotImplementedError

        return model_pred
    
    # Used for inference / evaluation ______________________________________________________________________________________
    def evaluation_api(self, batch = None, grids: GridBatch = None, batch_size: int = None, latent_prev: VDBTensor = None, 
                       use_ddim=False, ddim_step=100, use_ema=True, use_dpm=False, use_karras=False, solver_order=3,
                       h_stride=1, guidance_scale: float = 1.0, 
                       cond_dict=None, res_coarse=None, guided_grid=None):
        """
        * @param grids: GridBatch from previous stage for conditional diffusion
        * @param batch_size: batch_size for unconditional diffusion
        * @param latent_prev: previous stage latent for conditional diffusion; not implemented yet
        * @param use_ddim: use DDIM or not
        * @param ddim_step: number of steps for DDIM
        * @param use_dpm: use DPM++ solver or not
        * @param use_karras: use Karras noise schedule or not 
        * @param solver_order: order of the solver; 3 for unconditional diffusion, 2 for guided sampling
        * @param use_ema: use EMA or not
        * @param h_stride: flag for remain_h VAE to create a anisotropic latent grid
        * @param cond_dict: conditional dictionary -> only pass if manully effort needed
        * @param res_coarse: previous stage result (semantics, normals, etc) for conditional diffusion
        """
        if grids is None: 
            if batch is not None:
                latents = self.extract_latent(batch)
                grids = latents.grid
            else:
                # use dense diffusion
                # create a dense grid
                assert batch_size is not None, "batch_size should be provided"

                # Haven't seen this before #TODO
                feat_depth = self.vae.hparams.tree_depth - 1
                gap_stride = 2 ** feat_depth
                gap_strides = [gap_stride, gap_stride, gap_stride // h_stride]



                if isinstance(self.hparams.network.diffuser.image_size, int):
                    neck_bound = int(self.hparams.network.diffuser.image_size / 2)
                    low_bound = [-neck_bound] * 3
                    voxel_bound = [neck_bound * 2] * 3
                else:        
                    voxel_bound = self.hparams.network.diffuser.image_size
                    low_bound = [- int(res / 2) for res in self.hparams.network.diffuser.image_size]
                
                # sv is the voxel size
                voxel_sizes = [sv * gap for sv, gap in zip(self.vae.hparams.voxel_size, gap_strides)] # !: carefully setup # Why was this commented??? 
                origins = [sv / 2. for sv in voxel_sizes]
                grids = fvdb.sparse_grid_from_dense(
                                batch_size, 
                                voxel_bound, 
                                low_bound, 
                                device="cpu", # hack to fix bugs
                                voxel_sizes=voxel_sizes,
                                origins=origins).to(self.device)
        # parse the cond_dict
        if cond_dict is None:
            cond_dict = {}

        # mask condition
        if self.hparams.use_semantic_cond:
            # check if semantics is in cond_dict
            if 'semantics' not in cond_dict:
                if batch is not None:
                    cond_dict['semantics'] = fvdb.JaggedTensor(batch[DS.LATENT_SEMANTIC])
                elif res_coarse is not None:
                    cond_semantic = res_coarse.semantic_features[-1].feature.jdata # N, class_num
                    cond_semantic = torch.argmax(cond_semantic, dim=1)
                    cond_dict['semantics'] = grids.jagged_like(cond_semantic)
                else:
                    raise NotImplementedError("No semantics provided")
                

        # single scan concat condition
        if self.hparams.use_normal_concat_cond:
            # traing-time: get single scan crop from batch
            if batch is not None:
                ref_grid = fvdb.cat(batch[DS.INPUT_PC])    
                ref_xyz = ref_grid.grid_to_world(ref_grid.ijk.float()) 
                concat_normal = grids.splat_trilinear(ref_xyz, fvdb.JaggedTensor(batch[DS.TARGET_NORMAL]))
            elif res_coarse is not None:
                concat_normal = res_coarse.normal_features[-1].feature # N, 3
                concat_normal.jdata /= (concat_normal.jdata.norm(dim=1, keepdim=True) + 1e-6) # avoid nan
            else:
                raise NotImplementedError("No normal provided")
            cond_dict['normal'] = concat_normal                
        
        # diffusion process starts here ______________________________________________________________________________________
        if use_ema:
            with self.ema_scope("Evaluation API"):
                latents = self.random_sample_latents(grids, use_ddim=use_ddim, ddim_step=ddim_step, use_dpm=use_dpm, use_karras=use_karras, solver_order=solver_order,
                                                     cond_dict=cond_dict, guidance_scale=guidance_scale)
        else:
            latents = self.random_sample_latents(grids, use_ddim=use_ddim, ddim_step=ddim_step, use_dpm=use_dpm, use_karras=use_karras, solver_order=solver_order,
                                                     cond_dict=cond_dict, guidance_scale=guidance_scale)
        # decode
        res = self.vae.unet.FeaturesSet()
        if guided_grid is None:
            res, output_x = self.vae.unet.decode(res, latents, is_testing=True)
        else:
            res, output_x = self.vae.unet.decode(res, latents, guided_grid)
        # TODO: add SDF output
        return res, output_x
    

    # 
    def random_sample_latents(self, grids: GridBatch, generator: torch.Generator = None, 
                              use_ddim=False, ddim_step=None, use_dpm=False, use_karras=False, solver_order=3,
                              cond_dict=None, guidance_scale=1.0) -> VDBTensor:
        if use_ddim:
            if ddim_step is None:
                ddim_step = self.hparams.num_inference_steps
            self.ddim_scheduler.set_timesteps(ddim_step, device=grids.device)
            timesteps = self.ddim_scheduler.timesteps
            scheduler = self.ddim_scheduler

        elif use_dpm:
            if ddim_step is None:
                ddim_step = self.hparams.num_inference_steps
            try:
                self.dpm_scheduler.set_timesteps(ddim_step, device=grids.device)
            except:
                # create a new dpm scheduler
                self.dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=self.hparams.network.scheduler.num_train_timesteps,
                    beta_start=self.hparams.network.scheduler.beta_start,
                    beta_end=self.hparams.network.scheduler.beta_end,
                    beta_schedule=self.hparams.network.scheduler.beta_schedule,
                    solver_order=solver_order,
                    prediction_type=self.hparams.network.scheduler.prediction_type,
                    algorithm_type="dpmsolver++",
                    use_karras_sigmas=use_karras,
                )
                self.dpm_scheduler.set_timesteps(ddim_step, device=grids.device)
            timesteps = self.dpm_scheduler.timesteps
            scheduler = self.dpm_scheduler
        else:
            timesteps = self.noise_scheduler.timesteps
            scheduler = self.noise_scheduler
        
        # prepare the latents
        latents = torch.randn(grids.total_voxels, self.unet.out_channels, device=grids.device, generator=generator)
        
        for i, t in tqdm(enumerate(timesteps)):
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = VDBTensor(grid=grids, feature=grids.jagged_like(latent_model_input))
            # Predict the noise residual
            noise_pred = self._forward_cond(latent_model_input, t, cond_dict=cond_dict, is_testing=True, guidance_scale=guidance_scale) # TODO: cond
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred.feature.jdata, t, latents).prev_sample # TODO: when there is scale model input, why there is latents
            
        # scale the latents to the original scale
        if self.hparams.scale_by_std:
            latents = 1. / self.scale_factor * latents
        
        return VDBTensor(grid=grids, feature=grids.jagged_like(latents))
    










    def get_dataset_spec(self):
        all_specs = self.vae.get_dataset_spec()
        # further add new specs
        if self.hparams.use_text_cond:
            all_specs.append(DatasetSpec.TEXT_EMBEDDING)
            all_specs.append(DatasetSpec.TEXT_EMBEDDING_MASK)
        return all_specs
    
    def get_collate_fn(self):
        return list_collate
    
    def get_hparams_metrics(self):
        return [('val_loss', True)]

    def train_dataloader(self):
        import xcube.data as dataset
        from torch.utils.data import DataLoader

        shuffle = True
        train_set = dataset.build_dataset(
            self.hparams.train_dataset, self.get_dataset_spec(), self.hparams, self.hparams.train_kwargs, duplicate_num=self.hparams.duplicate_num) # !: A change here for adding duplicate num for trainset without lantet

        batch_size = self.hparams.batch_size
        
        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        import xcube.data as dataset
        from torch.utils.data import DataLoader
        val_set = dataset.build_dataset(
            self.hparams.val_dataset, self.get_dataset_spec(), self.hparams, self.hparams.val_kwargs)
        return DataLoader(val_set, batch_size=self.hparams.batch_size_val, shuffle=False,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())
