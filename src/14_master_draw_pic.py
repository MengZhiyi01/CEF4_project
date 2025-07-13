import torch
import torch.nn as nn
import netron
import math
import os
import json

# ==========================================================================================
# 1. 从 master_tune_class_official.py 复制必要的模型定义
#    为了让脚本可以独立运行，这里需要包含 MASTER 模型及其所有子模块的定义。
# ==========================================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        # self.pe 的形状: (max_len, d_model)
        # 我们需要将 pe 扩展以匹配 x 的批次大小
        # pe[:x.shape[1], :] 的形状是 (seq_len, d_model)
        return x + self.pe[:x.shape[1], :].unsqueeze(0)


class SAttention(nn.Module):
    """
    空间注意力模块
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(nn.Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    """
    时间注意力模块
    """

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(nn.Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TemporalAttention(nn.Module):
    """
    时间注意力聚合模块
    """

    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)
        return output


class MASTER(nn.Module):
    """
    MASTER 模型
    """

    def __init__(self, d_feat=133, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 num_classes=3):
        super(MASTER, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output

# ==========================================================================================
# 2. 主逻辑：加载最优模型并导出为 ONNX
# ==========================================================================================

def main():
    """
    主函数，用于加载模型、导出并可视化
    """
    # ---- 2.1 定义路径和参数 ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '../experiments/models/MASTER_tuned_classification')

    # 从训练脚本推断出的固定参数
    # d_feat 在训练脚本中是动态计算的，这里我们使用其计算逻辑
    # 原始特征133 + time_sin + time_cos = 135
    d_feat = 131
    step_len = 10 # 训练时使用的时间步长
    num_classes = 3 # 分类任务的类别数

    # 加载最佳超参数
    best_params_path = os.path.join(model_dir, 'master_clf_best_params.json')
    try:
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        print(f"成功加载最佳参数: {best_params}")
    except FileNotFoundError:
        print(f"错误: 找不到最佳参数文件 {best_params_path}")
        print("请先运行 master_tune_class_official.py 以生成最佳参数文件。")
        return

    # ---- 2.2 构建模型文件名并加载模型 ----
    model_name = (
        f"master_clf_d{best_params['d_model']}"
        f"_t{best_params['t_nhead']}_s{best_params['s_nhead']}"
        f"_do{best_params['dropout']}_lr{best_params['lr']}"
    )
    model_path = os.path.join(model_dir, f"{model_name}.pt")

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确认该模型文件已存在。")
        return

    # ---- 2.3 实例化模型并加载权重 ----
    print("正在实例化 MASTER 模型...")
    model = MASTER(
        d_feat=d_feat,
        d_model=best_params['d_model'],
        t_nhead=best_params['t_nhead'],
        s_nhead=best_params['s_nhead'],
        T_dropout_rate=best_params['dropout'],
        S_dropout_rate=best_params['dropout'],
        num_classes=num_classes
    )

    print(f"正在从 {model_path} 加载模型权重...")
    # 加载状态字典
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval() # 设置为评估模式
    print("模型加载成功。")

    # ---- 2.4 创建一个符合模型输入的 dummy_input ----
    # MASTER 模型期望的输入形状为 (batch_size, sequence_length, d_feat)
    batch_size = 1
    dummy_input = torch.randn(batch_size, step_len, d_feat)
    print(f"创建的 dummy_input 形状: {dummy_input.shape}")

    # ---- 2.5 导出为 ONNX 格式 ----
    onnx_path = os.path.join(base_dir, 'master_model.onnx')
    print(f"正在导出模型到 ONNX: {onnx_path}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=11
        )
        print(f"✅ 成功导出 ONNX 模型至: {onnx_path}")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型失败: {e}")
        return

    # ---- 2.6 使用 Netron 可视化 ----
    print("正在启动 Netron 来可视化模型...")
    print("如果浏览器没有自动打开，请手动访问 http://localhost:8080")
    netron.start(onnx_path)

if __name__ == "__main__":
    main()

