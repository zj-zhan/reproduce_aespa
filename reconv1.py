import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import logging
import hydra
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from mridataset_npy import MRISliceDataset 

from config import MRINeRF_Config
from model import CCM, CSM, mambalayer
from mr_utils import (kspace_to_img_shifted_mc, img_to_kspace_shifted_mc, 
                      coil_combine, coil_unfold, k_loss_l1, 
                      process_and_undersample_k_space, get_mask, train)
from utils import (kspace_to_target, normalize_np, 
                   set_seed, crop_center, random_2d_indices, dss, 
                   total_variation_loss_cc_jh, total_variation_loss_cmap_jh, cc_loss)

torch.fx.wrap('base_cpp.forward')
torch.autograd.set_detect_anomaly(True)
cs = ConfigStore.instance()
cs.store(name="mri_nerf_config", node=MRINeRF_Config)
log = logging.getLogger(__name__)

def reconstruct_step(cfg, batch_data, device, progress_str):
    fname = batch_data['fname'][0] 
    k_np = batch_data['k_space'][0].numpy() 

    cfg.files.slice = fname
    set_seed(2027) 

    target = kspace_to_target(k_np)
    
    reduction_factor = cfg.params.reduction_factor
    save_folder = ""
    acs = cfg.params.acs

    if cfg.params.mask =='uniform1d' and reduction_factor==4:
        mask = np.zeros((k_np.shape[-2], k_np.shape[-1]), dtype=float)
        mask[:,(k_np.shape[-1]-acs)//2:(k_np.shape[-1]+acs)//2-1] = 1
        mask[:,::(cfg.params.reduction_factor+2)] = 1
        save_folder = os.path.join(cfg.paths.save_folder, f'accel{reduction_factor}_norm', fname)
    
    elif cfg.params.mask =='gaussian1d' and reduction_factor==4:
        mask = get_mask(torch.zeros([1, 1, 640, 320]),320, 1,type='gaussian1d',acc_factor=reduction_factor,center_fraction=cfg.params.center_fraction)
        mask = mask.cpu().detach().numpy()[0][0]
        save_folder = os.path.join(cfg.paths.save_folder, 'gaussian1d',f'accel{reduction_factor}_norm', fname)
        
    elif cfg.params.mask =='equispaced1d' and reduction_factor==4:
        mask_tensor = get_mask(torch.zeros([1, 1, 640, 320]),320,1, type='equispaced1d', acc_factor=reduction_factor, center_fraction=cfg.params.center_fraction)
        mask = mask_tensor.cpu().detach().numpy()[0][0]
        save_folder = os.path.join(cfg.paths.save_folder, 'equispaced1d',f'accel{reduction_factor}_norm', fname)

    os.makedirs(f'{save_folder}/c_est', exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'weights'), exist_ok=True)

    k_space, mask, undersampled_k_space = process_and_undersample_k_space(k_np, mask, device)

    # Initial Guess
    mrim = kspace_to_img_shifted_mc(undersampled_k_space.clone().detach().to(dtype=torch.complex128))
    mrim = torch.sqrt(torch.sum(torch.abs(mrim)**2, dim=0))
    mrim_uk_max = torch.max(torch.abs(mrim))
    undersampled_k_space = undersampled_k_space / mrim_uk_max

    # Net2 ACS Input
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
    writer = SummaryWriter(log_dir=f'{save_folder}/runs')

    print(f"[{progress_str}] Processing {fname} ...")
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
        
        # Forward Pass
        predicted_c_c, predicted_cmap = train(**forward_kwargs)
        
        # Backward Pass
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
        
        #  Data Consistency (DC) & Input Update
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

        if epoch % 50 == 0:
            writer.add_scalar('Loss/Net1', net1_loss.item(), epoch)
            if epoch % 200 == 0:
                vis_img_raw = coil_combine(predicted_c_c, predicted_cmap)
                vis_img = torch.abs(vis_img_raw).detach().cpu().unsqueeze(0)
                vis_img = vis_img / torch.max(vis_img)
                writer.add_image(f'Reconstruction/{fname}', vis_img, epoch)

    print(f"[{progress_str}] Loop finished. Running final inference...")

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
    final_cmap = dss(final_cmap)
    final_img_np = final_c_c.detach().cpu().numpy()
    
    np.save(os.path.join(save_folder, 'final_recon.npy'), final_img_np)
    c_est_np = final_cmap.detach().cpu().numpy()
    np.save(os.path.join(save_folder, 'c_est', 'sensitivity_map.npy'), c_est_np)
    
    try:
        plt.imsave(os.path.join(save_folder, 'images', 'final_recon.png'), 
                   np.abs(final_img_np), cmap='gray')
    except Exception as e:
        print(f"Failed to save PNG: {e}")

    recon_norm = np.abs(final_img_np)
    target_norm = target
    psnr_val = peak_signal_noise_ratio(target_norm, recon_norm, data_range=float(1.0))
    ssim_val = structural_similarity(target_norm, recon_norm, data_range=1.0)
    
    writer.close()

    #torch.save(net.state_dict(), os.path.join(save_folder, 'weights', 'ccm_model.pth'))
    #torch.save(net2.state_dict(), os.path.join(save_folder, 'weights', 'csm_model.pth'))
    #torch.save(mambamodule.state_dict(), os.path.join(save_folder, 'weights', 'aksm_model.pth'))

    return psnr_val, ssim_val

@hydra.main(config_path="conf", config_name='AeSPa', version_base=None)
def main(cfg: MRINeRF_Config):
    print(cfg)

    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda:" + cfg.hyper_params.gpu)
        print("Running on the GPU" + cfg.hyper_params.gpu)
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    data_path = cfg.files.data_path

    dataset = MRISliceDataset(root_dir=data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Found {len(dataset)} slices. Start Batch Reconstruction...")
    
    results_log = []
    psnr_values = []
    ssim_values = []

    for i, batch in enumerate(dataloader):
        progress_str = f"{i + 1}/{len(dataset)}"
        fname = batch['fname'][0]
        
        try:
            curr_psnr,curr_ssim = reconstruct_step(cfg, batch, device, progress_str)
            
            if curr_psnr > 0:
                psnr_values.append(curr_psnr)
                ssim_values.append(curr_ssim)
                results_log.append((fname, curr_psnr, curr_ssim))
                print(f"Slice {fname} Done. PSNR: {curr_psnr:.5f} | SSIM: {curr_ssim:.5f}")
            else:
                print(f"Slice {batch['fname'][0]} Failed or Skipped.")
                
        except Exception as e:
            print(f"[{progress_str}] Error processing batch: {e}")
            import traceback
            traceback.print_exc()

    if len(psnr_values) > 0:
        psnr_array = np.array(psnr_values)
        ssim_array = np.array(ssim_values)
        mean_psnr = np.mean(psnr_array)
        std_psnr = np.std(psnr_array)
        mean_ssim = np.mean(ssim_array)
        std_ssim = np.std(ssim_array)
            
        print(f"{'SLICE NAME':<35} | {'PSNR':<10}")
        for name, p_val, s_val in results_log:
            print(f"{name:<35} | {p_val:.5f}     | {s_val:.5f}")

        print(f"Batch Reconstruction Finished for Subject: {cfg.files.subject}")
        print(f"Total Slices: {len(dataset)}")
        print(f"Average PSNR: {mean_psnr:.5f} ± {std_psnr:.5f}")
        print(f"Average SSIM: {mean_ssim:.5f} ± {std_ssim:.5f}")
    else:
        print("No slices were successfully reconstructed.")

if __name__ == "__main__":
    seed = 2027
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()