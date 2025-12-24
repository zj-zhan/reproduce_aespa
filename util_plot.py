import os
import numpy as np
import matplotlib.pyplot as plt
from util_eval import nmse, psnr, ssim

def plot_gt_pred(gt, pred, shape_raw, max_value, output_dir, name_ids, escale=1.):
    """
    Plot ground truth image with predicted image, both with 1 channel.
    Image black border is cut if exists in GT. Metrics are saved in filename.

    Parameters:
        gt: np.array [H, W]
        pred: np.array [H, W]
        shape_raw: [Nbs, 2], reference shape for center crop
        max_value: float
        output_dir: str, path to save figures
        escale: float, error display range scaling factor
    """
    os.makedirs(output_dir, exist_ok=True)
    erro = np.abs(gt - pred)

    plt.rcParams.update({'font.size': 14})
    title_list = ["True", "Predict", "Error"]

    with plt.ioff():
        fig, axs = plt.subplot_mosaic([['a'], ['b'], ['c']], layout='constrained',
                                        figsize=(6, 15), dpi=300)
        target_shape = shape_raw
        #gts = center_crop(gt, target_shape)
        #preds = center_crop(pred, target_shape)
        #erros = center_crop(erro, target_shape)
        #gts, preds, erros = img3_rm_black_border(gts, preds, erros)
        maxval = max_value
        vr = [0.0, np.max(gt)]

        for i, (label, ax) in enumerate(axs.items()):
            if i == 0:
                a = ax.imshow(gt, vmin=vr[0], vmax=vr[1], cmap='gray')
            elif i == 1:
                a = ax.imshow(pred, vmin=vr[0], vmax=vr[1], cmap='gray')
            elif i == 2:
                a = ax.imshow(erro, vmin=vr[0], vmax=vr[1]/escale, cmap='jet')
            ax.set_axis_off()
            ax.set_title(title_list[i])
            plt.colorbar(a, location='right')

        nmse_val = nmse(gt, pred)
        psnr_val = psnr(gt, pred, maxval)
        ssim_val = ssim(gt, pred, maxval)
        metric_str = f"nmse_{nmse_val:.5f}_psnr_{psnr_val:.4f}_ssim_{ssim_val:.4f}"
        savename = os.path.join(output_dir, f"data_{name_ids}_{metric_str}.png")
        plt.savefig(savename, bbox_inches='tight')
        plt.close()

    return None