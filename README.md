# Llama-QLoRA
使用QLoRA微调Llama-3.1-8B-Instruct

## 一、下载模型
去huggingface官网下载自己想要微调的模型（需要申请权限才可以access），我这里选择了Llama-3.1-8B-Instruct
拿到权限后clone想要的模型到本地，之后执行'git lfs pull'拉取模型权重，如果一切正常，我们会得到一个文件夹，名字为模型的名字，结构如下：
'''
├── config.json
├── generation_config.json
├── LICENSE
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── original
│   ├── consolidated.00.pth
│   ├── params.json
│   └── tokenizer.model
├── README.md
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── USE_POLICY.md
'''

## 二、数据集下载与处理
同样可以选择去huggingface官网下载自己想要的数据集，我这里选择了一个医学方面的数据集，运行'dataset_gen.py'文件即可得到'pubmedqa_llama.jsonl'微调数据集，格式为:
'{"instruction": "", "input": "", "output": ""}'

## 三、配置环境
'''
conda create -n finetuning python=3.10 -y
conda activate finetuning

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft bitsandbytes accelerate trl
'''
同样也可以
'''
conda create -n finetuning python=3.10 -y
conda activate finetuning

pip install -r requirements.txt
'''

## 四、微调模型
创建文件夹保存我们微调后的模型权重，我这里创建为'fine-tuning-model/qlora-medicine-model'，之后运行'QLoRA.py'文件来微调我们的模型，出现一下内容说明模型已经开始微调：
'''
  0%|          | 1/15314 [00:20<85:49:34, 20.18s/it]
  0%|          | 2/15314 [00:39<83:55:59, 19.73s/it]
  0%|          | 3/15314 [00:59<83:27:22, 19.62s/it]
  0%|          | 4/15314 [01:18<83:14:57, 19.58s/it]
'''
等待模型微调完毕即可

## 五、推理验证
模型微调完毕后会将微调得到的新权重保存到你指定的路径中，我这里为'fine-tuning-model/qlora-medicine-model'，如果一切正常，结构如下：
'''
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── checkpoint-375
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja
│   ├── optimizer.pt
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── trainer_state.json
│   └── training_args.bin
├── README.md
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
'''
最后，我们可以执行'compare.py'来对比模型微调前后对同一问题的回答来验证我们的模型是否微调成功
