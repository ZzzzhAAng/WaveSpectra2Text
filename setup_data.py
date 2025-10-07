"""
数据设置工具
帮助用户配置现有的音频文件和标签
重构后使用common_utils统一工具
"""

import os
from pathlib import Path
from common_utils import LabelManager, AudioProcessor, default_audio_processor


# 使用统一工具重构的函数
def scan_audio_files(audio_dir='data/audio'):
    """扫描音频目录中的文件 - 使用统一工具"""
    return LabelManager.scan_audio_files(audio_dir)


def create_labels_template(audio_files_or_dir, output_file='data/labels.csv'):
    """根据音频文件创建标签模板 - 使用统一工具"""
    if isinstance(audio_files_or_dir, (str, Path)):
        # 如果传入目录路径，直接使用统一工具
        return LabelManager.create_labels_template(audio_files_or_dir, output_file)
    else:
        # 如果传入文件列表，保持向后兼容
        print("⚠️  检测到旧版本调用方式，建议直接传入音频目录路径")
        if not audio_files_or_dir:
            print("没有找到音频文件")
            return False

        # 从文件列表推断目录
        if audio_files_or_dir:
            audio_dir = audio_files_or_dir[0].parent
            return LabelManager.create_labels_template(audio_dir, output_file)

        return False


def validate_labels_file(labels_file='data/labels.csv', audio_dir='data/audio'):
    """验证标签文件 - 使用统一工具"""
    return LabelManager.validate_labels_file(labels_file, audio_dir)


def show_audio_info(audio_files):
    """显示音频文件信息 - 使用统一工具"""
    if not audio_files:
        print("没有音频文件")
        return

    print("音频文件信息:")
    for i, audio_file in enumerate(audio_files[:5]):  # 只检查前5个文件
        info = default_audio_processor.get_audio_info(audio_file)

        if 'error' in info:
            print(f"  {audio_file.name}: 无法读取 ({info['error']})")
        else:
            print(f"  {audio_file.name}: {info['sample_rate']}Hz, {info['duration']:.2f}秒")

    if len(audio_files) > 5:
        print(f"  ... 还有 {len(audio_files) - 5} 个文件")


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
    print("3. 运行训练: python3 train_scale_1.py")


if __name__ == "__main__":
    main()
