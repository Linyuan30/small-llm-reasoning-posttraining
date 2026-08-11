from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/model/Qwen3-0.6B-Base"
)

print(tokenizer.encode("<think>"))
print(tokenizer.encode("</think>"))
print(tokenizer.encode("<answer>"))
print(tokenizer.encode("</answer>"))

