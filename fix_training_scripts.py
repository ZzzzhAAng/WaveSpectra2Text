#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复训练脚本的键名兼容性问题
将 batch['spectrograms'] 改为兼容新旧接口的版本
"""

import os
import re

def fix_training_script(filepath):
    """修复单个训练脚本"""
    print(f"🔧 修复文件: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    # 读取文件内容
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否需要修复
    if "batch['spectrograms']" not in content:
        print(f"✅ {filepath} 不需要修复")
        return True
    
    # 替换模式1: spectrograms = batch['spectrograms'].to(self.device)
    pattern1 = r"(\s+)spectrograms = batch\['spectrograms'\]\.to\(self\.device\)"
    replacement1 = r"""\1# 兼容新旧接口
\1if 'features' in batch:
\1    spectrograms = batch['features'].to(self.device)
\1else:
\1    spectrograms = batch['spectrograms'].to(self.device)"""
    
    content = re.sub(pattern1, replacement1, content)
    
    # 替换模式2: 其他可能的spectrograms访问
    pattern2 = r"batch\['spectrograms'\]"
    replacement2 = r"(batch['features'] if 'features' in batch else batch['spectrograms'])"
    
    # 只替换不在已修复代码中的部分
    lines = content.split('\n')
    fixed_lines = []
    skip_next = 0
    
    for i, line in enumerate(lines):
        if skip_next > 0:
            fixed_lines.append(line)
            skip_next -= 1
            continue
            
        if "# 兼容新旧接口" in line:
            # 跳过已修复的代码块
            fixed_lines.append(line)
            skip_next = 4  # 跳过接下来的4行
            continue
        
        # 替换其他未修复的spectrograms访问
        if "batch['spectrograms']" in line and "if 'features' in batch" not in line:
            line = line.replace("batch['spectrograms']", 
                              "(batch['features'] if 'features' in batch else batch['spectrograms'])")
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {filepath} 修复完成")
    return True

def main():
    """主函数"""
    print("🎯 修复训练脚本的接口兼容性")
    print("=" * 50)
    
    # 需要修复的训练脚本
    training_scripts = [
        'train_standard.py',
        'train_small.py',
        'train_medium.py', 
        'train_large.py'
    ]
    
    success_count = 0
    
    for script in training_scripts:
        if fix_training_script(script):
            success_count += 1
    
    print(f"\n📊 修复结果: {success_count}/{len(training_scripts)} 个文件修复成功")
    
    if success_count == len(training_scripts):
        print("🎉 所有训练脚本已修复！现在可以正常训练了")
        print("\n💡 使用方法:")
        print("python train_small.py --config config.json")
    else:
        print("❌ 部分文件修复失败，请检查错误信息")

if __name__ == "__main__":
    main()