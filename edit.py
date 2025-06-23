import cv2
import numpy as np

def blend_images_with_mask(ref_color_path, trans_color_path, mask_path, output_path):
    # 读取所有图片
    render = cv2.imread(render_path)
    ref_color = cv2.imread(ref_color_path)
    trans_color = cv2.imread(trans_color_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    
    # 确保所有图片存在
    if ref_color is None or trans_color is None or mask is None:
        print("Error: One or more input images could not be loaded!")
        return
    
    # 确保图片大小一致
    if ref_color.shape != trans_color.shape or ref_color.shape[:2] != mask.shape:
        print("Error: Image dimensions do not match!")
        return
    
    # 将mask二值化（确保是0和255）
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    print("Mask max value:", np.max(mask))
    # 将mask转换为float32并归一化到0-1范围
    mask_float = mask.astype(np.float32) / 255.0
    # print("Mask max value:", np.max(mask_float))
    
    # 扩展mask维度以便与彩色图像相乘
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
    print("Mask max value:", np.max(mask_3ch))
    # print(mask_3ch)
    # 计算混合结果
    # 白色区域 (mask == 1): 0.2 * ref_color + trans_color
    # 黑色区域 (mask == 0): ref_color + trans_color
    blended_image = (0.1 * ref_color + trans_color) * mask_3ch + render * (1 - mask_3ch)
    # blended_image = (2 * ref_color + trans_color) * mask_3ch
    # blended_image = (ref_color + trans_color)
    # 确保像素值在0-255范围内并转换为uint8
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    
    # 保存结果
    cv2.imwrite(output_path, blended_image)
    print(f"Result saved to: {output_path}")

# 示例调用
render_path = '/home/fb_21110240032/gaussian-splatting-main_sh/results/mirror_1/test/ours_30000/renders/00002.png'
ref_color_path = '/home/fb_21110240032/gaussian-splatting-main_sh/results/mirror_1/test/ours_30000/comp_ref_color/00002.png'      # 参考彩色图像路径
trans_color_path = '/home/fb_21110240032/gaussian-splatting-main_sh/results/mirror_1/test/ours_30000/comp_trans_color/00002.png'  # 过渡彩色图像路径
mask_path = '/home/fb_21110240032/EVF-SAM-main/results/00002_vis.png'               # 掩码图像路径（白色区域为特殊混合区域）
output_path = '/home/fb_21110240032/gaussian-splatting-main_sh/edit/mirror_0.1.png'           # 输出结果路径

blend_images_with_mask(ref_color_path, trans_color_path, mask_path, output_path)