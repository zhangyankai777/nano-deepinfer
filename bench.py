import argparse
import time

from nanodeepinfer import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
parser.add_argument("--num-seqs", type=int, default=16)
parser.add_argument("--max-tokens", type=int, default=128)
args = parser.parse_args()

llm = LLM(args.model, enforce_eager=True)
prompts = [f"Question {i}: explain inference optimization in one paragraph." for i in range(args.num_seqs)]
t0 = time.time()
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_tokens), use_tqdm=False)
total = sum(len(o["token_ids"]) for o in outputs)
elapsed = time.time() - t0
print(f"{total} tokens, {elapsed:.2f}s, {total / elapsed:.1f} tok/s")
