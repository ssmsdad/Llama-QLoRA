import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_path = "Llama-3.1-8B-Instruct"
adapter_path = "fine-tuning-model/qlora-medicine-model"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def get_response(model, prompt, tokenizer, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# prompt = "Instruction: 请详细说明线粒体在程序性细胞死亡过程中扮演的角色。\nInput: \nOutput:"
# prompt = "请回答以下问题：线粒体是否在lace plant的程序性细胞死亡中发挥作用？"
prompt = "Instruction: 请根据以下医学文献内容回答问题：Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\nInput: Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.\nOutput:"

# 1. 推理微调前（全精度）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto"
)
print("【微调前】")
print(get_response(base_model, prompt, tokenizer))
# 释放对象，清理显存
del base_model
torch.cuda.empty_cache()

# 2. 推理微调后（必须量化加载）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
ft_model = PeftModel.from_pretrained(base_model,adapter_path,)
print(ft_model.print_trainable_parameters())
print("\n【微调后】")
print(get_response(ft_model, prompt, tokenizer))
del ft_model, base_model
torch.cuda.empty_cache()