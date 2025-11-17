# ShareGPT to Qwen Pre-Tokenized Dataset Converter

Convert ShareGPT format datasets (like FineTome-100k) to pre-tokenized format compatible with Axolotl training using Qwen tokenizer.

## Quick Start

```bash
python sharegpt_to_qwen_tokenized.py \
  --dataset mlabonne/FineTome-100k \
  --output finetome_qwen_tokenized.jsonl \
  --model Qwen/Qwen2.5-0.5B \
  --max-length 3000 \
  --num-samples 1000 \
  --num-examples 5 \
  --show-complete
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | HuggingFace dataset name or local path | `mlabonne/FineTome-100k` |
| `--output` | Output JSONL file path | `finetome_qwen_tokenized.jsonl` |
| `--model` | Qwen model name for tokenizer | `Qwen/Qwen2-0.5B` |
| `--max-length` | Maximum sequence length | `2048` |
| `--num-samples` | Number of samples to convert (None = all) | `None` |
| `--num-examples` | Number of examples to display after conversion | `2` |
| `--split` | Dataset split to use | `train` |
| `--add-generation-prompt` | Add generation prompt (for inference datasets) | `False` |
| `--no-color` | Disable colored output for sample display | `False` |
| `--show-complete` | Show complete decoded text (not truncated) | `False` |
| `--no-verbose` | Disable per-example truncation warnings | `False` |

## Output Format

The converter generates JSONL files with pre-tokenized examples:

```json
{
  "input_ids": [151644, 8948, 198, ...],
  "attention_mask": [1, 1, 1, ...],
  "labels": [-100, -100, -100, ..., 151644]
}
```

- **input_ids**: Token IDs from the Qwen tokenizer
- **attention_mask**: All 1s (all tokens are valid)
- **labels**: Token IDs with `-100` for masked tokens (user/system messages), actual token IDs for assistant responses

## Features

- ✅ **Proper Label Masking**: Only trains on assistant responses
- ✅ **Colored Output**: Visual distinction between trained (green) and masked (red) tokens
- ✅ **Truncation Warnings**: Shows when sequences exceed max length
- ✅ **Multiple Display Options**: Control verbosity and output detail
- ✅ **Generation Prompt Support**: For inference datasets

## Examples

### Convert full dataset
```bash
python sharegpt_to_qwen_tokenized.py \
  --dataset mlabonne/FineTome-100k \
  --output finetome_full.jsonl \
  --model Qwen/Qwen2.5-1.5B
```

### Convert with longer context
```bash
python sharegpt_to_qwen_tokenized.py \
  --dataset mlabonne/FineTome-100k \
  --output finetome_long.jsonl \
  --max-length 4096 \
  --num-samples 5000
```

### Quiet mode (no per-example warnings)
```bash
python sharegpt_to_qwen_tokenized.py \
  --dataset mlabonne/FineTome-100k \
  --output finetome_quiet.jsonl \
  --no-verbose \
  --no-color \
  --num-examples 0
```

## Use with Axolotl

After conversion, use the generated JSONL file with Axolotl:

```yaml
# config.yml
datasets:
  - path: finetome_qwen_tokenized.jsonl
    type:  # Leave empty for pre-tokenized

model_type: AutoModelForCausalLM
base_model: Qwen/Qwen2.5-0.5B

sequence_len: 3000
sample_packing: false

# LoRA configuration
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

Then train:
```bash
accelerate launch -m axolotl.cli.train config.yml
```

## Notes

- Qwen models don't use BOS tokens, only EOS tokens
- User and system messages are masked with `-100` (not trained)
- Assistant responses are trained (including the final EOS token)
- Use `--verbose` mode to see individual truncation warnings
- Summary statistics always show total truncation count
