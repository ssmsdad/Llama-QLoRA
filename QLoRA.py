from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorWithPadding,DataCollatorForSeq2Seq
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_path = "Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 是否以4bit量化方式加载模型，极大节省显存
    bnb_4bit_compute_dtype=torch.float16,   # 计算时使用的数据类型（如float16），影响推理/训练速度和精度
    bnb_4bit_quant_type="nf4",              # 4bit量化类型，"nf4"（NormalFloat4）是效果较好的新型4bit量化方式
    bnb_4bit_use_double_quant=True,         # 是否使用双重量化（double quantization），进一步减少显存占用
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

'''
prepare_model_for_kbit_training：
1、冻结模型参数，防止误更新。
2、处理量化模型的梯度计算、输入输出等兼容性问题。
3、确保量化权重和 LoRA adapter 能正确协同训练。
'''
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,      # LoRA 的缩放因子（影响训练稳定性和效果，常用16/32/64等）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# get_peft_model：把 LoRA adapter 层“插入”到原始大模型中
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

dataset = load_dataset("json", data_files={"train": "pubmedqa_llama.jsonl"}, split="train")

def preprocess_function(examples):
    inputs = [
        f"Instruction: {ins}\nInput: {inp}\nOutput:" 
        for ins, inp in zip(examples["instruction"], examples["input"])
    ]
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True,        # 如果文本超过 max_length，会自动截断到512
        padding="max_length" 
    )
    labels = tokenizer(
        examples["output"], 
        max_length=512, 
        truncation=True, 
        padding="max_length" 
    )
    # 将 labels 中的 padding token 替换为 -100，表示忽略这些位置的损失计算，虽然DataCollatorForSeq2Seq会自动处理，但这里手动设置更保险
    labels = labels["input_ids"]
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])

'''
dataset:(batch_size=2)
{
  'input_ids': [
    [101, 2009, 2001, 1037, 2742, 102],      # 第一个样本的 prompt
    [101, 2023, 2003, 2172, 102, 0, 0]       # 第二个样本的 prompt（有 padding）
  ],
  'attention_mask': [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0]
  ],
  'labels': [
    [2009, 2001, 1037, 2742, 102, -100],     # 第一个样本的 output
    [2023, 2003, 2172, 102, -100, -100, -100]
  ]
}
'''

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

training_args = TrainingArguments(
    output_dir="fine-tuning-model/qlora-medicine-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,     # 通常2e-5到5e-5之间
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,     # 最多只保留2个最近的 checkpoint，旧的会被删掉
    bf16=True,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("fine-tuning-model/qlora-medicine-model")
tokenizer.save_pretrained("fine-tuning-model/qlora-medicine-model")
