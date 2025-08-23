"""运行模型对比实验的主脚本"""

import os
import sys
import argparse
from experiments.model_comparator import ModelComparator

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行Transformer和图模型的对比实验')
    parser.add_argument('--data_path', type=str, default='../input/smiles.csv',
                       help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='输出目录')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['transformer', 'GraphConv', 'MPNN', 'AttentiveFP'],
                       help='要比较的模型列表')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        print("请提供正确的数据文件路径")
        return
    
    print("=" * 60)
    print("分子性质预测模型对比实验")
    print("=" * 60)
    print(f"数据文件: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"比较模型: {', '.join(args.models)}")
    print("=" * 60)
    
    try:
        # 创建模型对比器
        comparator = ModelComparator(args.data_path, args.output_dir)
        
        # 运行实验
        results = comparator.run_all_experiments()
        
        print("\n" + "=" * 60)
        print("实验完成!")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
