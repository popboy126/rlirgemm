# Notes

## Download models

The following commands download the models used by the MLC LLM integration.

```shell
uv add huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

mkdir -p DeepSeek-R1-0528-Qwen3-8 gemma-3-270m Ministral-3-3B-Instruct-2512 Llama-3.2-3B-Instruct
uvx hf download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local-dir ./DeepSeek-R1-0528-Qwen3-8
uvx hf download google/gemma-3-270m --local-dir ./gemma-3-270m && \
uvx hf download mistralai/Ministral-3-3B-Instruct-2512 --local-dir ./Ministral-3-3B-Instruct-2512 && \
uvx hf download meta-llama/Llama-3.2-3B-Instruct  --local-dir ./Llama-3.2-3B-Instruct

```
