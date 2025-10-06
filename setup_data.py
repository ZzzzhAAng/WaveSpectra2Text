"""
数据设置工具
帮助用户配置现有的音频文件和标签
"""

import os
import pandas as pd
from pathlib import Path


def scan_audio_files(audio_dir='data/audio'):
    """扫描音频目录中的文件"""
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        print(f"音频目录不存在: {audio_dir}")
        return []

    # 支持的音频格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']

    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(audio_dir.glob(f'*{ext}')))
        audio_files.extend(list(audio_dir.glob(f'*{ext.upper()}')))

    # 按文件名排序
    audio_files.sort(key=lambda x: x.name)

    return audio_files


def create_labels_template(audio_files, output_file='data/labels.csv'):
    """根据音频文件创建标签模板"""
    if not audio_files:
        print("没有找到音频文件")
        return False

    print(f"找到 {len(audio_files)} 个音频文件:")
    for i, file in enumerate(audio_files[:10]):  # 只显示前10个
        print(f"  {i + 1}. {file.name}")
    if len(audio_files) > 10:
        print(f"  ... 还有 {len(audio_files) - 10} 个文件")

    # 创建标签数据
    labels_data = []
    chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    for i, audio_file in enumerate(audio_files):
        filename = audio_file.name

        # 尝试从文件名推断标签
        label = ""

        # 如果文件名包含数字，尝试映射到中文
        for j, num in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]):
            if num in filename:
                if j < len(chinese_numbers):
                    label = chinese_numbers[j]
                break

        # 如果没有找到匹配，使用占位符
        if not label:
            if i < len(chinese_numbers):
                label = chinese_numbers[i]
            else:
                label = "未知"

        labels_data.append({
            'filename': filename,
            'label': label
        })

    # 保存到CSV文件
    df = pd.DataFrame(labels_data)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n标签模板已创建: {output_file}")
    print("请检查并修改标签文件中的标签，确保它们与音频内容匹配")

    return True


def validate_labels_file(labels_file='data/labels.csv', audio_dir='data/audio'):
    """验证标签文件"""
    if not os.path.exists(labels_file):
        print(f"标签文件不存在: {labels_file}")
        return False

    try:
        df = pd.read_csv(labels_file)
    except Exception as e:
        print(f"读取标签文件失败: {e}")
        return False

    # 检查必要的列
    required_columns = ['filename', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"标签文件缺少列: {missing_columns}")
        return False

    # 检查音频文件是否存在
    audio_dir = Path(audio_dir)
    missing_files = []
    existing_files = []

    for _, row in df.iterrows():
        audio_path = audio_dir / row['filename']
        if audio_path.exists():
            existing_files.append(row['filename'])
        else:
            missing_files.append(row['filename'])

    print(f"标签文件验证结果:")
    print(f"  总条目数: {len(df)}")
    print(f"  存在的音频文件: {len(existing_files)}")
    print(f"  缺失的音频文件: {len(missing_files)}")

    if missing_files:
        print(f"  缺失文件列表: {missing_files}")
        return False

    # 检查标签
    valid_labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    invalid_labels = []

    for _, row in df.iterrows():
        if row['label'] not in valid_labels:
            invalid_labels.append((row['filename'], row['label']))

    if invalid_labels:
        print(f"  无效标签: {invalid_labels}")
        print(f"  有效标签应为: {valid_labels}")
        return False

    print("✓ 标签文件验证通过")
    return True


def show_audio_info(audio_files):
    """显示音频文件信息"""
    if not audio_files:
        print("没有音频文件")
        return

    try:
        import librosa

        print("音频文件信息:")
        for i, audio_file in enumerate(audio_files[:5]):  # 只检查前5个文件
            try:
                y, sr = librosa.load(str(audio_file), sr=None)
                duration = len(y) / sr
                print(f"  {audio_file.name}: {sr}Hz, {duration:.2f}秒")
            except Exception as e:
                print(f"  {audio_file.name}: 无法读取 ({e})")

        if len(audio_files) > 5:
            print(f"  ... 还有 {len(audio_files) - 5} 个文件")

    except ImportError:
        print("需要安装librosa来显示音频信息: pip install librosa")


def main():
    """主函数"""
    print("数据设置工具")
    print("=" * 50)

    # 扫描音频文件
    print("1. 扫描音频文件...")
    audio_files = scan_audio_files()

    if not audio_files:
        print("在 data/audio 目录中没有找到音频文件")
        print("请将你的48kHz音频文件放置在 data/audio 目录中")
        print("支持的格式: .wav, .mp3, .flac, .m4a, .aac")
        return

    # 显示音频信息
    print("\n2. 音频文件信息...")
    show_audio_info(audio_files)

    # 创建或验证标签文件
    labels_file = 'data/labels.csv'

    if os.path.exists(labels_file):
        print(f"\n3. 验证现有标签文件...")
        if validate_labels_file(labels_file):
            print("标签文件验证通过，可以开始训练")
        else:
            print("标签文件有问题，请修正后再试")
    else:
        print(f"\n3. 创建标签模板...")
        if create_labels_template(audio_files, labels_file):
            print("请编辑标签文件，然后重新运行此脚本进行验证")

    print("\n" + "=" * 50)
    print("设置完成!")
    print("\n下一步:")
    print("1. 检查并编辑 data/labels.csv 文件")
    print("2. 确保标签与音频内容匹配")
    print("3. 运行训练: python3 train_standard.py")


if __name__ == "__main__":
    main()