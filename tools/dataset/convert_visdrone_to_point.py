import os
import glob

# 直接使用你刚刚找到的绝对路径
base_dir = "/home/pc/gxy/dataset/VisDrone2019/VisDrone2019"

def convert_yolo_to_point(split):
    print(f"正在处理 {split} 集...")
    # 直接读取 YOLO 格式的 labels 文件夹
    label_dir = os.path.join(base_dir, f"VisDrone2019-DET-{split}", "labels")
    out_dir = os.path.join(base_dir, f"VisDrone2019-DET-{split}", "point_labels")

    if not os.path.exists(label_dir):
        print(f"警告: 找不到标签文件夹 {label_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    count = 0
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        out_lines = []
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # YOLO 格式: class_id cx cy w h (已经是归一化的)
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    # 只保留类别和中心点
                    out_lines.append(f"{class_id} {cx:.6f} {cy:.6f}\n")

        with open(os.path.join(out_dir, file_name), 'w') as f:
            f.writelines(out_lines)
        count += 1

    print(f"{split} 集转换完成! 共处理 {count} 个文件。")
    print(f"输出目录: {out_dir}\n")

if __name__ == "__main__":
    convert_yolo_to_point("train")
    convert_yolo_to_point("val")