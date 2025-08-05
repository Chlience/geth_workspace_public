import os
import argparse
import re

def modify_file(file_path, n_hidden, n_layers, save=False):
    """分析并修改文件中的n_hidden和n_layers值，返回修改前后的行"""
    changes = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        new_line = line
        
        # 检查并修改n_hidden
        if n_hidden is not None and re.match(r'^\s*self\.n_hidden\s*=\s*\d+', line):
            before = line.strip()
            new_line = re.sub(r'self\.n_hidden\s*=\s*\d+', f'self.n_hidden = {n_hidden}', line)
            after = new_line.strip()
            changes.append(('n_hidden', before, after))
        
        # 检查并修改n_layers
        elif n_layers is not None and re.match(r'^\s*self\.n_layers\s*=\s*\d+', line):
            before = line.strip()
            new_line = re.sub(r'self\.n_layers\s*=\s*\d+', f'self.n_layers = {n_layers}', line)
            after = new_line.strip()
            changes.append(('n_layers', before, after))
        
        new_lines.append(new_line)
    
    # 如果有修改且需要保存，则写入文件
    if changes and save:
        with open(file_path, 'w') as file:
            file.writelines(new_lines)
    
    return changes

def main():
    parser = argparse.ArgumentParser(description='修改指定模型和数据集的文件配置')
    parser.add_argument('--dir', type=str, required=True, help='包含Python文件的目录')
    parser.add_argument('--model', type=str, default='all', help='模型名称')
    parser.add_argument('--dataset', type=str, default='all', help='数据集名称')
    parser.add_argument('--n_hidden', type=int, default=None, help='新的n_hidden值')
    parser.add_argument('--n_layers', type=int, default=None, help='新的n_layers值')
    parser.add_argument('--save', action='store_true', help='是否实际写入修改（默认只显示不修改）')
    
    args = parser.parse_args()
    
    print(f"运行模式: {'实际修改' if args.save else '模拟运行（不修改文件）'}")
    
    # 遍历目录中的所有.py文件
    for filename in os.listdir(args.dir):
        if filename.endswith('.py'):
            # 解析文件名格式 {model}_{dataset}.py
            parts = filename[:-3].split('_')  # 去掉.py后缀后分割
            if len(parts) >= 2:
                file_model = parts[0]
                file_dataset = '_'.join(parts[1:])  # 处理dataset中可能包含下划线的情况
                
                # 检查是否匹配model和dataset
                model_match = (args.model == 'all') or (file_model == args.model)
                dataset_match = (args.dataset == 'all') or (file_dataset == args.dataset)
                
                if model_match and dataset_match:
                    # if file_model == 'sage':
                    #     print(f'跳过文件: {filename}（SAGE模型不处理）')
                    #     continue
                    
                    file_path = os.path.join(args.dir, filename)
                    print(f'\n处理文件: {filename}')
                    
                    changes = modify_file(file_path, args.n_hidden, args.n_layers, args.save)
                    
                    if changes:
                        for change_type, before, after in changes:
                            print(f'  {change_type}:')
                            print(f'    修改前: {before}')
                            print(f'    修改后: {after}')
                        status = "已实际修改" if args.save else "将修改（模拟运行）"
                        print(f'  [{status}]')
                    else:
                        print('  未找到需要修改的行')

if __name__ == '__main__':
    main()