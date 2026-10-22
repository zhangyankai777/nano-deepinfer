import argparse

from nanodeepinfer import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
parser.add_argument("--prompt", default="请用一句话介绍深度学习推理框架。")
args = parser.parse_args()

llm = LLM(args.model, enforce_eager=True, max_num_seqs=16)
outputs = llm.generate([args.prompt], SamplingParams(temperature=0.6, max_tokens=128))
print(outputs[0]["text"])
