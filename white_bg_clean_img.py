import cv2
import numpy as np
import os

def replace_transparent_pixels(reference_image_path, target_image_path, output_image_path):
    # 读取参考图和目标图，确保图像有alpha通道
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)  # 读取带alpha通道的图像
    target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)  # 读取带alpha通道的图像
    
    # 确保两张图都有alpha通道
    if reference_image.shape[2] != 4:
        print(f"错误: {reference_image_path}不是带alpha通道的图片。")
        return
    
    # 获取参考图中的透明像素位置
    alpha_channel = reference_image[:, :, 3]  # 获取alpha通道（透明度通道）
    transparent_pixels = alpha_channel == 0  # 透明像素的标记
    
    # 将目标图像中对应透明像素的位置替换为白色
    target_image[transparent_pixels] = [255, 255, 255]  # 设置为白色（R, G, B, A）
    
    # 保存结果图像
    cv2.imwrite(output_image_path, target_image)
    print(f"处理后的图像已保存为 {output_image_path}")

def process_images(reference_folder, target_folder, output_folder):
    # 获取参考文件夹和目标文件夹中的所有文件
    reference_files = sorted(os.listdir(reference_folder))
    target_files = sorted(os.listdir(target_folder))
    
    # 确保两个文件夹的文件数量相同
    if len(reference_files) != len(target_files):
        print("错误: 参考文件夹和目标文件夹中的文件数量不匹配。")
        return
    
    # 遍历每对参考图像和目标图像
    for reference_file, target_file in zip(reference_files, target_files):
        # 构建文件路径
        reference_image_path = os.path.join(reference_folder, reference_file)
        target_image_path = os.path.join(target_folder, target_file)
        
        # 构建输出文件路径
        output_image_path = os.path.join(output_folder, target_file)  # 输出图像与目标图像同名
        
        # 替换透明像素
        replace_transparent_pixels(reference_image_path, target_image_path, output_image_path)

if __name__ == "__main__":
    # 输入和输出文件夹路径
    reference_folder = '/home/fb_21110240032/refnerf/car/images/test'  # 参考图像文件夹（包含透明区域）
    target_folder = '/home/fb_21110240032/refnerf/car/clean_images/test'  # 目标图像文件夹（将透明像素替换为白色）
    output_folder = '/home/fb_21110240032/refnerf/car/clean_images/test'  # 输出文件夹
    
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 处理文件夹中的所有图像
    process_images(reference_folder, target_folder, output_folder)
