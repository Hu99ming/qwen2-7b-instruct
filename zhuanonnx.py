import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
  
def _load_model_tokenizer(checkpoint_path):  
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, resume_download=False)  
    model = AutoModelForCausalLM.from_pretrained(  
        checkpoint_path,  
        torch_dtype="auto",  
        resume_download=False,  
    ).eval()  
    return model, tokenizer  
  
def _convert_to_onnx(model, tokenizer, output_onnx_path):  
    # 定义一个示例输入，这里需要根据你的模型输入要求进行调整  
    # 假设模型接受一个批次的文本输入，并将其编码为张量  
    dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]  
      
    # 导出模型为 ONNX 格式  
    torch.onnx.export(  
        model,                       # 要导出的模型  
        dummy_input,                 # 示例输入  
        output_onnx_path,            # 输出的 ONNX 文件路径  
        export_params=True,          # 导出参数  
        opset_version=14,            # ONNX opset 版本，根据需要选择  
        do_constant_folding=True,    # 是否进行常量折叠优化  
        input_names=['input_ids'],   # 输入张量的名称（与模型输入对应）  
        output_names=['output'],     # 输出张量的名称（根据需要命名）  
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴设置，这里假设批次大小是动态的  
    )  
    print(f"Model exported to ONNX file: {output_onnx_path}")  
  
def main():  
    checkpoint_path = "/root/wenjian/qwen2/Qwen2-7B-Instruct"  # 模型路径  
    output_onnx_path = "qwen2_7B_instruct.onnx"  # 输出的 ONNX 文件路径  
  
    model, tokenizer = _load_model_tokenizer(checkpoint_path)  
    _convert_to_onnx(model, tokenizer, output_onnx_path)  
  
if __name__ == "__main__":  
    main()