import cv2
import numpy as np
import pywt
import os

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)
    return img

def wavelet_decompose(image_path, wavelet='sym4', save_dir='wavelet_output'):
    # 读取图像（支持中文）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img = img.astype(np.float32)

    # ===================== 核心：两级分解 =====================
    # 一级分解
    coeffs1 = pywt.dwt2(img, wavelet)
    LL1, (LH1, HL1, HH1) = coeffs1
    # 二级分解（只分解低频LL1）
    coeffs2 = pywt.dwt2(LL1, wavelet)
    LL2, (LH2, HL2, HH2) = coeffs2

    # 归一化所有子带
    LL2 = normalize_to_uint8(LL2)
    LH2 = normalize_to_uint8(LH2)
    HL2 = normalize_to_uint8(HL2)
    HH2 = normalize_to_uint8(HH2)
    LH1 = normalize_to_uint8(LH1)
    HL1 = normalize_to_uint8(HL1)
    HH1 = normalize_to_uint8(HH1)

    # 创建文件夹+中文保存
    os.makedirs(save_dir, exist_ok=True)
    def imwrite(path, img):
        cv2.imencode('.png', img)[1].tofile(path)

    # ===================== 关键修正：标准二级分解画布拼接 =====================
    h2, w2 = LL2.shape  # 二级子带大小
    h1, w1 = LH1.shape   # 一级子带大小

    # 新建和原图一样大的画布
    canvas = np.zeros((h1*2, w1*2), dtype=np.uint8)

    # 左上角：放二级分解的4个子带（最核心的变化！）
    canvas[0:h2, 0:w2] = LL2
    canvas[0:h2, w2:w2*2] = LH2
    canvas[h2:h2*2, 0:w2] = HL2
    canvas[h2:h2*2, w2:w2*2] = HH2

    # 右侧：一级水平细节
    canvas[0:h1, w1:w1*2] = LH1
    # 下侧：一级垂直细节
    canvas[h1:h1*2, 0:w1] = HL1
    # 右下角：一级对角细节
    canvas[h1:h1*2, w1:w1*2] = HH1

    # 保存结果
    imwrite(os.path.join(save_dir, 'two_level_combined.png'), canvas)
    print("✅ 二级小波分解完成！查看 two_level_combined.png")

if __name__ == "__main__":
    image_path = r"C:\Users\asus\Desktop\文件\代码\e91570b5404dd171db53304c06e51ed7.png"
    # 支持 haar / sym4 / db4
    wavelet_decompose(image_path, wavelet='haar')