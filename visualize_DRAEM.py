import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEM_Test_Visual_Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, StudentReconstructiveSubNetwork
import os
import matplotlib.pyplot as plt


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    # 建立主存檔資料夾
    save_root = "./save_files"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []

    print("🔄 開始測試，共有物件類別:", len(obj_names))

    for obj_idx, obj_name in enumerate(obj_names):
        print(f"\n▶️ [{obj_idx+1}/{len(obj_names)}] 測試物件類別: {obj_name}")

        img_dim = 256
        run_name = base_model_name + "_" + obj_name + '_'

        # 載入模型
        print("  ⏳ 載入重建模型權重...")
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(
            torch.load(os.path.join("student_best" + ".pth"),
                       map_location='cuda:0'))
        # model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, run_name + ".pckl"),
                       map_location='cuda:0'))
        model.cuda()
        model.eval()

        print("  ⏳ 載入分割模型權重...")
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(
            torch.load(os.path.join(checkpoint_path, run_name + "_seg.pckl"),
                       map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        # 建立 dataset / dataloader
        data_dir = os.path.join(mvtec_path, obj_name, "test")
        print(f"  📂 建立 dataset: {data_dir}")
        dataset = MVTecDRAEM_Test_Visual_Dataset(
            data_dir, resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0)
        print("  ✅ Dataset size:", len(dataset))

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_gt_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_out_masks = torch.zeros((16, 1, 256, 256)).cuda()
        display_in_masks = torch.zeros((16, 1, 256, 256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(dataloader), size=(16, ))

        print("  🚀 開始遍歷 dataloader...")

        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 20 == 0:
                print(f"    🔹 Batch {i_batch}/{len(dataloader)}")

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose(
                (1, 2, 0))

            gray_rec, _ = model(gray_batch)
            # gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0].cpu().detach()
                display_gt_images[cnt_display] = gray_batch[0].cpu().detach()
                display_out_masks[cnt_display] = t_mask[0].cpu().detach()
                display_in_masks[cnt_display] = true_mask[0].cpu().detach()
                cnt_display += 1

            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

            # 建立 2x2 圖表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 顯示重建圖片
            axes[0, 0].imshow(gray_rec[0].detach().cpu().numpy().transpose(
                1, 2, 0))
            axes[0, 0].set_title('Reconstructed Image')
            axes[0, 0].axis('off')

            # 顯示原始圖片
            axes[0, 1].imshow(gray_batch[0].detach().cpu().numpy().transpose(
                1, 2, 0))
            axes[0, 1].set_title('Original Image')
            axes[0, 1].axis('off')

            # 顯示預測的異常遮罩
            axes[1, 0].imshow(out_mask_cv)
            axes[1, 0].set_title('Predicted Anomaly Heatmap')
            axes[1, 0].axis('off')

            # 顯示真實的異常遮罩
            axes[1, 1].imshow(true_mask[0, 0].detach().cpu().numpy(),
                              cmap='hot')
            axes[1, 1].set_title('Ground Truth Mask')
            axes[1, 1].axis('off')

            # 儲存整張圖
            plt.tight_layout()
            plt.savefig(f"{save_root}/comparison_{obj_name}_{i_batch}.png")
            plt.close()

            out_mask_averaged = torch.nn.functional.avg_pool2d(
                out_mask_sm[:, 1:, :, :], 21, stride=1,
                padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) *
                               img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) *
                                  img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        print(f"  ✅ {obj_name} 測試完成，共處理 {len(dataset)} 張影像")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name',
                        action='store',
                        type=str,
                        required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path',
                        action='store',
                        type=str,
                        required=True)

    args = parser.parse_args()

    obj_list = [
        'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor', 'tile',
        'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut', 'screw',
        'grid', 'wood'
    ]
    # 建立主存檔資料夾
    save_root = "./save_files"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path,
             args.base_model_name)
