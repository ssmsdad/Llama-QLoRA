import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_path = "Llama-3.1-8B-Instruct"
adapter_path = "fine-tuning-model/qlora-medicine-model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# 正确加载 LoRA adapter（关键一步）
ft_model = PeftModel.from_pretrained(model,adapter_path,is_trainable=True)

ft_model.print_trainable_parameters()

# https://github.com/ssmsdad/Llama-QLoRA.git