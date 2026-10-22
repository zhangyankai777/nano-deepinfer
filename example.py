import argparse

from nanodeepinfer import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
parser.add_argument("--prompt", default="请用一句话介绍深度学习推理框架。")
parser.add_argument("--profile", default="deepinfer", choices=["deepinfer", "rtx3060", "3060", "sm86"],
                    help="deepinfer: 原版数据中心卡默认配置；rtx3060: RTX 3060 (SM86+) 12GB 级单卡配置")
parser.add_argument("--max-tokens", type=int, default=128)
args = parser.parse_args()

llm = LLM(args.model, profile=args.profile, enforce_eager=True)
outputs = llm.generate([args.prompt], SamplingParams(temperature=0.6, max_tokens=args.max_tokens))
print(outputs[0]["text"])
