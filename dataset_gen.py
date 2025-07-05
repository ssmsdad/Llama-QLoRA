from datasets import load_dataset
import json

def convert_pubmedqa_to_llama(data_list, save_path="pubmedqa_llama.jsonl"):
    with open(save_path, "w", encoding="utf-8") as fout:
        for item in data_list:
            question = item.get("question", "")
            # 合并所有上下文段落
            contexts = item.get("context", {}).get("contexts", [])
            context_text = "\n".join(contexts)

            long_answer = item.get("long_answer", "").strip()
            if not long_answer:
                # 如果没有长答案，使用final_decision或者默认提示
                long_answer = item.get("final_decision", "无法回答")

            instruction = f"Please answer the following question based on the content of the following medical literature: {question}"
            input_text = context_text
            output_text = long_answer

            record = {
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"转换完成，数据保存到 {save_path}")

def main():
    print("开始加载数据集...")
    dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
    print(f"加载了 {len(dataset)} 条样本")
    print("示例样本：")
    print(dataset[0])

    # 转换成 LLaMA 微调格式
    convert_pubmedqa_to_llama(dataset, save_path="pubmedqa_llama.jsonl")

if __name__ == "__main__":
    main()
