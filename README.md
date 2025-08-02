# LLM Math Evaluation Harness
## 🚀 Getting Started

### ⚙️ Environment Setup
```
cd math-evaluation-harness
pip install uv # if not install uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
git submodule update --init --recursive
cd octotools
uv pip install -e .
# set .env 
```

### ⚖️ Evaluation
Here we evaluate `Qwen-2.5-Math-1.5B/7B-Verl-Tool` using the following script. More examples can be found in [./scripts](./scripts).

```bash
# Qwen-2.5-Math-1.5B/7B-Verl-Tool
bash scripts/run_eval_math_greedy_deepmath.sh 
```

### How to add new dataset?
Use `add_new_dataset.py`. You should check that `question` and `answer` fileds are inclued in processed data.  


### How to add new prompt?
+ **Step 1**: Add your prompt in `./utils.py`.

+ **Step 2**: Add your model's stop words in `./math_eval.py`.  

As an example, you can search for `torl` to see how it has been integrated and modified.


## Cheatsheet

| Model | Prompt |
|:-----:|:------:|
|   Qwen-2.5-Math    |  `qwen25-math-cot`      |
|    Qwen-2.5   |  `qwen25-cot`      |
|   [ToRL](https://github.com/Zhuofeng-Li/ToRL)    |   `torl`    |
|   [Verl-Tool-Math](https://huggingface.co/VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step)    |   `torl_deepmath_qwen`     |
|   [Search-R1](https://github.com/PeterGriffinJin/Search-R1)    |  `search-r1`      |


