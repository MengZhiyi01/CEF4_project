import torch
import torch.nn as nn
import netron

# ==== 1. 加载保存的模型 ====
# 假设模型保存在 'model.pt'，并且是通过 torch.save(model, 'model.pt') 保存的
model_path = 'master_model.pt'
model = torch.load(model_path, map_location='cpu')  # 加载整个模型结构
model.train()  # 设置为评估模式

# ==== 2. 创建 dummy 输入 ====
# 这里假设模型的输入维度为 [batch_size, 158]，你可以根据实际情况修改
dummy_input = torch.randn(1, 158)  # batch_size = 1

# ==== 3. 导出为 ONNX ====
onnx_path = 'nn_model.onnx'
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

print(f"✅ 成功导出 ONNX 模型至：{onnx_path}")

# ==== 4. 使用 Netron 可视化 ====
netron.start(onnx_path)