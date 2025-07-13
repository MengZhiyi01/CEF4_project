import torch
import torch.nn as nn
import netron
import os
import argparse
import numpy as np

# 从训练脚本复制ALSTM模型定义（确保与转换脚本一致）
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

def visualize_alstm_model(args):
    """
    加载ALSTM模型并生成可视化结构图
    """
    print(f"--- 开始ALSTM模型可视化 ---")
    print(f"模型路径: {args.model_path}")
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(args.model_path):
            print(f"❌ 错误: 模型文件不存在: {args.model_path}")
            print("请先运行 alstm_pkl_to_pt.py 来生成.pt格式的模型文件")
            return False
        
        # 加载模型
        model = torch.load(args.model_path, map_location='cpu')
        model.eval()  # 设置为评估模式
        print("✅ 成功加载ALSTM模型")
        
        # 打印模型信息
        print(f"\n--- 模型信息 ---")
        print(f"模型类型: {type(model)}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # 创建dummy输入
        # ALSTM模型需要两个输入：连续特征和品种索引
        batch_size = 1
        seq_len = 10  # 时间步长
        d_feat = model.d_feat
        n_instruments = model.n_instruments
        
        print(f"\n--- 创建Dummy输入 ---")
        print(f"批次大小: {batch_size}")
        print(f"序列长度: {seq_len}")
        print(f"特征维度: {d_feat}")
        print(f"品种数量: {n_instruments}")
        
        # 创建dummy输入
        dummy_x_continuous = torch.randn(batch_size, seq_len, d_feat)
        dummy_x_instrument = torch.randint(0, n_instruments, (batch_size, seq_len))
        
        print(f"连续特征输入形状: {dummy_x_continuous.shape}")
        print(f"品种索引输入形状: {dummy_x_instrument.shape}")
        
        # 测试前向传播
        with torch.no_grad():
            output = model(dummy_x_continuous, dummy_x_instrument)
            print(f"模型输出形状: {output.shape}")
        
        # 导出为ONNX格式
        onnx_path = os.path.join(os.path.dirname(args.model_path), 'alstm_model.onnx')
        
        print(f"\n--- 导出ONNX模型 ---")
        print(f"ONNX文件路径: {onnx_path}")
        
        torch.onnx.export(
            model,
            (dummy_x_continuous, dummy_x_instrument),  # 模型输入
            onnx_path,
            input_names=["x_continuous", "x_instrument"],
            output_names=["output"],
            dynamic_axes={
                "x_continuous": {0: "batch_size", 1: "seq_len"},
                "x_instrument": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size"}
            },
            opset_version=11,
            verbose=True
        )
        
        print(f"✅ 成功导出ONNX模型: {onnx_path}")
        
        # 使用Netron可视化
        print(f"\n--- 启动Netron可视化 ---")
        print("正在打开浏览器显示模型结构图...")
        print("如果浏览器没有自动打开，请手动访问显示的URL")
        
        try:
            # 尝试启动netron，不指定port参数
            netron.start(onnx_path)
            print(f"\n🎉 可视化完成！")
            print(f"ONNX模型文件: {onnx_path}")
            print("模型结构图已在浏览器中打开")
        except Exception as e:
            print(f"⚠️ Netron启动失败: {e}")
            print("尝试使用备用方法...")
            try:
                # 备用方法：直接启动，让netron自动选择端口
                netron.start(onnx_path, browse=True)
                print(f"\n🎉 可视化完成！")
                print(f"ONNX模型文件: {onnx_path}")
                print("模型结构图已在浏览器中打开")
            except Exception as e2:
                print(f"❌ 备用方法也失败: {e2}")
                print(f"请手动打开浏览器访问: file://{os.path.abspath(onnx_path)}")
                print("或者安装最新版本的netron: pip install --upgrade netron")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="ALSTM模型可视化工具")
    
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(base_dir, '../experiments/models/ALSTM_tuned_classification/alstm_model.pt'),
                        help='ALSTM模型文件路径(.pt格式)')
    
    args = parser.parse_args()
    
    success = visualize_alstm_model(args)
    
    if success:
        print(f"\n✅ 可视化流程完成！")
        print("请在浏览器中查看模型结构图")
    else:
        print(f"\n❌ 可视化失败，请检查错误信息")
        print("确保已运行 alstm_pkl_to_pt.py 生成.pt格式的模型文件") 
