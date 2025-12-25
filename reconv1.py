import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import logging
import hydra
import shutil
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from tqdm import tqdm
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True
from mridataset_npy import MRISliceDataset 
from mridataset import MRIDataset
from config import MRINeRF_Config
from model import CCM, CSM, mambalayer
from util_plot import plot_gt_pred
from util_eval import psnr, ssim, nmse as calc_nmse, print_metric
from mr_utils import (kspace_to_img_shifted_mc, img_to_kspace_shifted_mc, 
                      coil_combine, coil_unfold, k_loss_l1, 
                      process_and_undersample_k_space, get_mask, train)
from utils import (normalize_np, set_seed, random_2d_indices, dss, total_variation_loss_cc_jh,
                   total_variation_loss_cmap_jh, cc_loss)

torch.fx.wrap('base_cpp.forward')
torch.autograd.set_detect_anomaly(True)
cs = ConfigStore.instance()
cs.store(name="mri_nerf_config", node=MRINeRF_Config)
log = logging.getLogger(__name__)

def reconstruct_step(cfg, batch_data, device, batch_idx):
    fname = batch_data['fname'][0] 
    k_np = batch_data['kspace'][0].numpy() 
    target = batch_data['rss'][0].numpy()
    maxval = batch_data['max_val'][0].item() 
    
    if 'shape_raw' in batch_data:
        shape_raw = batch_data['shape_raw'][0].numpy()
    else:
        shape_raw = target.shape

    set_seed(2027)
    save_root = cfg.paths.save_folder
    visual_dir = os.path.join(save_root, 'visual')
    
    reduction_factor = cfg.params.reduction_factor
    acs = cfg.params.acs
    height, width = k_np.shape[-2], k_np.shape[-1]

    if cfg.params.mask =='uniform1d' and reduction_factor==4:
        mask = np.zeros((height, width), dtype=float)
        start_idx = (width - acs) // 2
        end_idx = start_idx + acs
        mask[:, start_idx:end_idx] = 1 
        mask[:, ::(cfg.params.reduction_factor+2)] = 1
    
    elif cfg.params.mask =='gaussian1d' and reduction_factor==4:
        mask_shape = torch.zeros([1, 1, height, width])
        mask = get_mask(mask_shape, width, 1, type='gaussian1d', 
                        acc_factor=reduction_factor, center_fraction=cfg.params.center_fraction)
        mask = mask.cpu().detach().numpy()[0][0]
        
    elif cfg.params.mask =='equispaced1d' and reduction_factor==4:
        mask_shape = torch.zeros([1, 1, height, width])
        mask_tensor = get_mask(mask_shape, width, 1, type='equispaced1d', 
                               acc_factor=reduction_factor, center_fraction=cfg.params.center_fraction)
        mask = mask_tensor.cpu().detach().numpy()[0][0]

    k_space, mask, undersampled_k_space = process_and_undersample_k_space(k_np, mask, device)

    # Net2 ACS Input
    mrim = kspace_to_img_shifted_mc(undersampled_k_space.clone().detach().to(dtype=torch.complex128))
    mrim = torch.sqrt(torch.sum(torch.abs(mrim)**2, dim=0))
    mrim_uk_max = torch.max(torch.abs(mrim))
    undersampled_k_space = undersampled_k_space / mrim_uk_max
    mrim_test_ = kspace_to_img_shifted_mc(undersampled_k_space)
    mrim_test = torch.sqrt(torch.sum(torch.abs(mrim_test_)**2, dim=0))
    mrim_test_phase = torch.angle(torch.sum(mrim_test_, dim=0))
    
    # ACS Mask
    acs_start = (k_np.shape[-1] - acs) // 2
    acs_end = (k_np.shape[-1] + acs) // 2 - 1
    maska = torch.zeros_like(mask)
    for i in range(acs_start, acs_end+1, 1):
        maska[:,:,i] = 1
    acs_k_space = undersampled_k_space * maska
    maska = maska.to(device)
    
    acs_image = kspace_to_img_shifted_mc(acs_k_space)
    acs_image = torch.cat([acs_image.real, acs_image.imag]).float().to(device)
    acs_image = acs_image / torch.max(torch.abs(acs_image))

    kwargs = {'log': log, 'k': k_np.shape[0]} 
    if cfg.models.model == 'joint_training':
        net = CCM(**kwargs).to(device)
        net2 = CSM(**kwargs).to(device)
        optimizer2 = optim.Adam(net2.parameters(), lr=cfg.hyper_params.lr)
    
    mambamodule = mambalayer().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.hyper_params.lr)
    optimizer3 = optim.Adam(mambamodule.parameters(), lr=cfg.hyper_params.aksmlr)
    
    initial_input = torch.complex(mrim_test*torch.cos(mrim_test_phase), mrim_test*torch.sin(mrim_test_phase))
    nomalized_initial_input = initial_input / torch.max(torch.abs(mrim_test))

    width, height = k_space[0,:,:].shape
    kspace_repository_r = torch.zeros_like(undersampled_k_space, dtype=torch.float32)
    kspace_repository_i = torch.zeros_like(undersampled_k_space, dtype=torch.float32)
    
    epochs = cfg.hyper_params.epochs
    predicted_c_c_c = None 

    forward_kwargs = {
        'net': net,
        'net2': net2,
        'u_k': acs_image,
        'coords': torch.cat([nomalized_initial_input.real.unsqueeze(0),
                             nomalized_initial_input.imag.unsqueeze(0)],dim=0).to(device),
        'width': width,
        'height': height,
        'step': 0
    }

    for epoch in range(epochs):
        predicted_c_c, predicted_cmap = train(**forward_kwargs)
        
        predicted_cmap = dss(predicted_cmap)
        mr_image = coil_unfold(predicted_c_c, predicted_cmap)
        predicted_c_c_c_c = predicted_c_c.detach().clone()
        predicted_c_c_s   = predicted_c_c.clone()
        predicted_k_space = img_to_kspace_shifted_mc(mr_image)

        loss4 = total_variation_loss_cc_jh(torch.abs(predicted_c_c_s)) 
        loss3 = total_variation_loss_cmap_jh(torch.abs(predicted_cmap))
        loss2 = cc_loss(predicted_cmap)
        loss1 = k_loss_l1(undersampled_k_space, predicted_k_space, mask)
        loss6 = loss1.clone().detach()
        net1_loss = loss1 + loss4
        net2_loss = loss6 + loss2 + loss3

        net1_loss.backward(retain_graph=True)
        net2_loss.backward()
        
        optimizer.step(); optimizer2.step()
        optimizer.zero_grad(); optimizer2.zero_grad()
        
        # Mamba / AKSM
        predicted_k_spaces = predicted_k_space.detach().clone() 
        mamba_in = torch.cat([predicted_k_spaces.real.unsqueeze(0).detach(),
                              predicted_k_spaces.imag.unsqueeze(0).detach()], dim=0).float().to(device)
        predicted_k_space_att, att_score = mambamodule(mamba_in)
        att_score = torch.sigmoid(att_score)

        coils = att_score.size(1) // 2
        att_score = torch.complex(att_score[0,:coils], att_score[0,coils:])
        
        predicted_k_space_t = att_score * predicted_k_spaces
        mr = kspace_to_img_shifted_mc(predicted_k_space_t)
        ccimage = coil_combine(mr, predicted_cmap.detach())
        ccimage = ccimage / torch.max(torch.abs(ccimage))
        aksm_loss = total_variation_loss_cc_jh(ccimage)
        
        optimizer3.zero_grad(); aksm_loss.backward(); optimizer3.step()

        # Repository Update
        top_k_val = k_np.shape[1] if epoch == 0 else cfg.params.selective_top_k
        top_values, top_indices_r = torch.topk(predicted_k_space_att.detach(), top_k_val, dim=-2)

        for i in range(predicted_k_spaces.size(0)):
            kspace_repository_r[i].scatter_(-2, top_indices_r[i], predicted_k_spaces.real[i].gather(-2, top_indices_r[i]).float())
            kspace_repository_i[i].scatter_(-2, top_indices_r[i], predicted_k_spaces.imag[i].gather(-2, top_indices_r[i]).float())
        predicted_k_space_final = torch.complex(kspace_repository_r, kspace_repository_i)
        
        # Data Consistency
        if epoch == 0 or (epoch+1) % cfg.params.csm_update_iteration == 0:
            random_rows, random_cols, random_chas = random_2d_indices(predicted_c_c_s, cfg.params.kspace_masking, epoch, k_np.shape[0])
            predicted_k_space_k = predicted_k_space.detach().clone()
            predicted_k_space_k[random_chas, random_rows, random_cols] = 0
            predicted_k_space_k = predicted_k_space_k / mrim_uk_max
            predicted_k_space_k = predicted_k_space_k * (1-mask) + undersampled_k_space * mask
            
            predicted_c_c_c = kspace_to_img_shifted_mc(predicted_k_space_k)
            temp_combine = coil_combine(predicted_c_c_c, predicted_cmap)
            predicted_c_c_c = predicted_c_c_c / torch.max(torch.abs(temp_combine))
            
            predicted_c_c_c_c = coil_combine(predicted_c_c_c, predicted_cmap)
            predicted_c_c_c_c = predicted_c_c_c_c / torch.max(torch.abs(predicted_c_c_c_c))
        else:
            predicted_c_c_c_c = coil_combine(predicted_c_c_c, predicted_cmap)
            predicted_c_c_c_c = predicted_c_c_c_c / torch.max(torch.abs(predicted_c_c_c_c))

        if epoch < epochs - 1:
            forward_kwargs = {
                'net': net, 'net2': net2,
                'u_k': torch.cat([predicted_c_c_c.real.detach(), predicted_c_c_c.imag.detach()], dim=0).float().to(device),
                'coords': torch.cat([predicted_c_c_c_c.real.unsqueeze(0).detach(),
                                     predicted_c_c_c_c.imag.unsqueeze(0).detach()], dim=0).float().to(device),
                'width': width, 'height': height, 'step': epoch + 1
            }

    final_kwargs = {
        'net': net,
        'net2': net2,
        'u_k': torch.cat([predicted_c_c_c.real.detach(), predicted_c_c_c.imag.detach()], dim=0).float().to(device),
        'coords': torch.cat([predicted_c_c_c_c.real.unsqueeze(0).detach(), predicted_c_c_c_c.imag.unsqueeze(0).detach()], dim=0).float().to(device),
        'width': width,
        'height': height,
        'step': epochs,
    }

    final_c_c, final_cmap = train(**final_kwargs)
    # final_cmap = dss(final_cmap)
    final_img_np = final_c_c.detach().cpu().numpy()

    maxval0 = float(1.0) # for 2d eval
    maxval1 = float(maxval / np.max(target)) # for 3d eval
    x = normalize_np(np.abs(final_img_np))
    y = normalize_np(target)
    
    psnr2d_val = psnr(y, x, maxval0)
    ssim2d_val = ssim(y, x, maxval0)
    psnr3d_val = psnr(y, x, maxval1) 
    ssim3d_val = ssim(y, x, maxval1)
    nmse_val = calc_nmse(y, x)
    
    if batch_idx < 100:
        os.makedirs(visual_dir, exist_ok=True)
        plot_gt_pred(gt=y, 
                     pred=x, 
                     shape_raw=shape_raw, 
                     max_value=maxval1, 
                     output_dir=visual_dir, 
                     name_ids=batch_idx)

    return psnr2d_val, ssim2d_val, psnr3d_val, ssim3d_val, nmse_val, save_root

@hydra.main(config_path="conf", config_name='AeSPa', version_base=None)
def main(cfg: MRINeRF_Config):
    print(cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + cfg.hyper_params.gpu)
        print("Running on the GPU" + cfg.hyper_params.gpu)
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    data_path = cfg.files.data_path
    target_size = cfg.files.target_size
    which_data = cfg.files.which_data 
    data_norm_type = cfg.files.data_norm_type

    #dataset = MRISliceDataset(root=data_path, target_size=320)  # npy dataset
    dataset = MRIDataset(data_path, target_size, which_data, data_norm_type)  # h5 dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    
    results_log = []
    psnr2d_list = []
    ssim2d_list = []
    psnr3d_list = []
    ssim3d_list = []
    nmse_list = []
    
    loop = tqdm(dataloader, desc="Reconstructing", unit="slice")
    
    save_dir = cfg.paths.save_folder
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(loop):
        cur_psnr2d, cur_ssim2d, cur_psnr3d, cur_ssim3d, cur_nmse, cur_save_root = reconstruct_step(cfg, batch, device, i)
        save_dir = cur_save_root
        
        psnr2d_list.append(cur_psnr2d)
        ssim2d_list.append(cur_ssim2d)
        psnr3d_list.append(cur_psnr3d)
        ssim3d_list.append(cur_ssim3d)
        nmse_list.append(cur_nmse)
        
        results_log.append((i, cur_psnr2d, cur_ssim2d, cur_psnr3d, cur_ssim3d, cur_nmse))
        #loop.set_postfix_str(f"PSNR2d: {cur_psnr2d:.2f} | SSIM2d: {cur_ssim2d:.4f}")

    psnr2d_array = np.array(psnr2d_list)
    ssim2d_array = np.array(ssim2d_list)
    psnr3d_array = np.array(psnr3d_list)
    ssim3d_array = np.array(ssim3d_list)
    nmse_array = np.array(nmse_list)

    print_metric("NMSE", nmse_array)
    print_metric("PSNR2d", psnr2d_array)
    print_metric("SSIM2d", ssim2d_array)
    print_metric("PSNR3d", psnr3d_array)
    print_metric("SSIM3d", ssim3d_array)
    mean_psnr2d = np.mean(psnr2d_array)
    std_psnr2d = np.std(psnr2d_array)
    mean_ssim2d = np.mean(ssim2d_array)
    std_ssim2d = np.std(ssim2d_array)

    csv_path = os.path.join(save_dir, 'eval_metrics.csv')
    os.makedirs(save_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Slice', 'PSNR2d', 'SSIM2d', 'PSNR3d', 'SSIM3d', 'NMSE'])
        writer.writerows(results_log)
        writer.writerow([])
        writer.writerow(['Average', mean_psnr2d, mean_ssim2d, np.mean(psnr3d_array), np.mean(ssim3d_array), np.mean(nmse_array)])
        writer.writerow(['Std', std_psnr2d, std_ssim2d, np.std(psnr3d_array), np.std(ssim3d_array), np.std(nmse_array)])
        
    print(f"Metrics saved to: {csv_path}")   

if __name__ == "__main__":
    seed = 2027
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    main()