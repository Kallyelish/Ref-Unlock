import cv2
import numpy as np
import os

def blend_images_from_folders(ref_color_folder, ref_map_folder, trans_color_folder, output_folder):
    # 获取文件夹中的所有图片文件
    ref_color_files = sorted(os.listdir(ref_color_folder))
    ref_map_files = sorted(os.listdir(ref_map_folder))
    trans_color_files = sorted(os.listdir(trans_color_folder))
    
    # 确保文件夹中的图片数量相同
    if not (len(ref_color_files) == len(ref_map_files) == len(trans_color_files)):
        print("三个文件夹中的图片数量不一致！")
        return

    # 逐一处理每一组图片
    for i in range(len(ref_color_files)):
        ref_color_path = os.path.join(ref_color_folder, ref_color_files[i])
        ref_map_path = os.path.join(ref_map_folder, ref_map_files[i])
        trans_color_path = os.path.join(trans_color_folder, trans_color_files[i])
        
        # 读取三张图片
        ref_color = cv2.imread(ref_color_path)
        ref_map = cv2.imread(ref_map_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        trans_color = cv2.imread(trans_color_path)

        # 确保图片大小一致
        if ref_color.shape != trans_color.shape or ref_color.shape[:2] != ref_map.shape:
            print(f"图片尺寸不匹配: {ref_color_files[i]}")
            continue

        # 归一化 ref_map 到 0-1 范围
        ref_map = ref_map / 255.0
        print(ref_map)
        # 计算加权结果
        blended_image = 1.5 * ref_color * ref_map[..., None] + trans_color * (1 - ref_map[..., None])

        # 转换为uint8类型，保存输出图片
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
        
        # 创建输出文件夹路径
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, ref_color_files[i])

        # 保存结果
        cv2.imwrite(output_path, blended_image)
        print(f"保存结果: {output_path}")

# 示例调用
ref_color_folder = '/home/fb_21110240032/gaussian-splatting-main/results/mirror/test/ours_30000/ref_color'  # 替换为参考彩色图像文件夹路径
ref_map_folder = '/home/fb_21110240032/gaussian-splatting-main/results/mirror/test/ours_30000/ref_map'      # 替换为参考映射图像文件夹路径
trans_color_folder = '/home/fb_21110240032/gaussian-splatting-main/results/mirror/test/ours_30000/trans_color'  # 替换为过渡彩色图像文件夹路径
output_folder = 'output_folder_1.5'  # 替换为输出结果文件夹路径

blend_images_from_folders(ref_color_folder, ref_map_folder, trans_color_folder, output_folder)
