import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import json
from torch.utils.data import DataLoader, Dataset

# 从训练脚本复制ALSTM模型定义
class ALSTMModel(nn.Module):
    """
    ALSTM模型的核心结构。
    """

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, n_instruments=0, embedding_dim=4,
                 n_classes=3):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_instruments = n_instruments
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes

        self.instrument_embedding = nn.Embedding(num_embeddings=self.n_instruments, embedding_dim=self.embedding_dim)
        self.fc_in = nn.Linear(in_features=self.d_feat + self.embedding_dim, out_features=self.hidden_size)
        self.act = nn.Tanh()
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.att_net = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size / 2)),
            nn.Dropout(self.dropout),
            nn.Tanh(),
            nn.Linear(in_features=int(self.hidden_size / 2), out_features=1, bias=False),
            nn.Softmax(dim=1)
        )
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=self.n_classes)

    def forward(self, x_continuous, x_instrument):
        instrument_embed = self.instrument_embedding(x_instrument)
        x = torch.cat([x_continuous, instrument_embed], dim=2)
        x = self.act(self.fc_in(x))
        rnn_out, _ = self.rnn(x)
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(torch.cat((rnn_out[:, -1, :], out_att), dim=1))
        return out

def load_and_convert_model(args):
    """
    加载.pkl模型并转换为.pt格式
    """
    print(f"--- 开始加载模型并转换格式 ---")
    print(f"模型目录: {args.model_dir}")
    
    try:
        # 加载最优参数
        with open(os.path.join(args.model_dir, 'best_params_results.json'), 'r') as f:
            best_params = json.load(f)
        print("✅ 成功加载最优超参数")
        
        # 加载instrument映射
        try:
            with open(os.path.join(args.model_dir, 'instrument_map.pkl'), 'rb') as f:
                instrument_map = pickle.load(f)
            print("✅ 成功加载品种映射表")
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                print("⚠️ 检测到numpy版本兼容性问题，尝试使用兼容模式加载...")
                # 尝试使用兼容模式加载
                import pickle5 as pickle_compat
                try:
                    with open(os.path.join(args.model_dir, 'instrument_map.pkl'), 'rb') as f:
                        instrument_map = pickle_compat.load(f)
                    print("✅ 使用兼容模式成功加载品种映射表")
                except ImportError:
                    print("❌ 无法导入pickle5，尝试其他解决方案...")
                    # 如果pickle5不可用，尝试重新创建映射
                    print("正在重新创建品种映射...")
                    full_data = pd.read_parquet(args.data_path)
                    train_data = full_data[full_data['datetime'] < '2024-12-01'].copy()
                    instrument_map = pd.Index(train_data['instrument'].unique())
                    print(f"✅ 重新创建品种映射，包含 {len(instrument_map)} 个品种")
            else:
                raise e
        except Exception as e:
            print(f"❌ 加载品种映射表失败: {e}")
            print("尝试重新创建品种映射...")
            full_data = pd.read_parquet(args.data_path)
            train_data = full_data[full_data['datetime'] < '2024-12-01'].copy()
            instrument_map = pd.Index(train_data['instrument'].unique())
            print(f"✅ 重新创建品种映射，包含 {len(instrument_map)} 个品种")
        
        # 计算d_feat
        full_data = pd.read_parquet(args.data_path)
        feature_cols = [col for col in full_data.columns if not (
                col.startswith('label') or col in ['datetime', 'instrument', 'instrument_idx']
        )]
        d_feat = len(feature_cols)
        print(f"✅ 计算得到特征维度 d_feat: {d_feat}")
        
        # 构建模型
        n_instruments = len(instrument_map)
        model = ALSTMModel(
            d_feat=d_feat,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            n_instruments=n_instruments,
            embedding_dim=best_params['embedding_dim'],
            n_classes=3
        )
        
        # 加载.pkl模型权重
        model_filename = f"alstm_cls_hs{best_params['hidden_size']}_nl{best_params['num_layers']}_do{best_params['dropout']}_ed{best_params['embedding_dim']}_lr{best_params['lr']}_bs{best_params['batch_size']}.pkl"
        model_path = os.path.join(args.model_dir, model_filename)
        
        try:
            checkpoint_state_dict = torch.load(model_path, map_location='cpu')
            print(f"✅ 成功加载模型权重: {model_filename}")
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                print("⚠️ 模型权重加载遇到numpy版本兼容性问题...")
                print("尝试使用兼容模式加载模型权重...")
                # 尝试使用pickle5加载
                try:
                    import pickle5 as pickle_compat
                    with open(model_path, 'rb') as f:
                        checkpoint_state_dict = pickle_compat.load(f)
                    print(f"✅ 使用兼容模式成功加载模型权重: {model_filename}")
                except ImportError:
                    print("❌ 无法使用pickle5，尝试其他解决方案...")
                    raise e
            else:
                raise e
        except Exception as e:
            print(f"❌ 加载模型权重失败: {e}")
            raise e
        
        # 在模型权重加载成功后添加load_state_dict调用
        model.load_state_dict(checkpoint_state_dict)
        
        # 保存为.pt格式
        pt_model_path = os.path.join(args.model_dir, 'alstm_model.pt')
        torch.save(model, pt_model_path)
        print(f"✅ 成功保存为.pt格式: {pt_model_path}")
        
        # 打印模型信息
        print(f"\n--- 模型信息 ---")
        print(f"特征维度 (d_feat): {d_feat}")
        print(f"隐藏层大小: {best_params['hidden_size']}")
        print(f"LSTM层数: {best_params['num_layers']}")
        print(f"Dropout率: {best_params['dropout']}")
        print(f"嵌入维度: {best_params['embedding_dim']}")
        print(f"品种数量: {n_instruments}")
        print(f"输出类别数: 3")
        
        return pt_model_path, d_feat, best_params
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="ALSTM模型格式转换工具")
    
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(base_dir, 'models/ALSTM_tuned_classification/'),
                        help='包含最优模型、参数和映射表的目录路径')
    
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(base_dir, '../data/output/final_data_standardized_with_ud.parquet'),
                        help='预处理后的数据文件路径')
    
    args = parser.parse_args()
    
    pt_model_path, d_feat, best_params = load_and_convert_model(args)
    
    if pt_model_path:
        print(f"\n🎉 转换完成！模型已保存为: {pt_model_path}")
        print("现在可以使用 alstm_draw_pic.py 来可视化模型结构了。")
    else:
        print("\n❌ 转换失败，请检查错误信息。") 