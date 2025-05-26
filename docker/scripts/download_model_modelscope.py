import os
import shutil
from modelscope import snapshot_download

# 指定模型名和输出目录
model_id = 'jiaohui/kokoro'
cache_dir = './models/'  # 缓存目录路径
dest_dir = "./api/src/models/v1_0/"  # 目标目录路径
model_file_name = "kokoro-v1_0.pth"  # 你要找的模型文件名

# 下载模型到指定目录（返回的是模型主目录）
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(f"模型已下载到: {model_dir}")

# 在模型目录下递归查找目标 .pth 文件
source_file_path = None
for root, dirs, files in os.walk(model_dir):
    if model_file_name in files:
        source_file_path = os.path.join(root, model_file_name)
        break

if source_file_path is None:
    print(f"未找到 {model_file_name}，请检查模型目录 {model_dir}")
else:
    # 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"目标目录 {dest_dir} 已创建")

    dest_file_path = os.path.join(dest_dir, model_file_name)
    shutil.copy2(source_file_path, dest_file_path)
    print(f"文件已复制到: {dest_file_path}")