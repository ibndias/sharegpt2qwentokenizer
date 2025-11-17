"""
ShareGPT to Qwen Pre-Tokenized Dataset Converter

This script converts ShareGPT format datasets (like FineTome-100k) to 
pre-tokenized format compatible with Axolotl training using Qwen tokenizer.

Output format:
- input_ids: List of token IDs
- attention_mask: List of 1s (same length as input_ids)
- labels: List of token IDs with -100 for tokens that should be ignored during training

Author: Generated for fine-tuning Qwen models
Date: 2025-11-17
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


class ShareGPTToQwenConverter:
    """Converts ShareGPT format to Qwen pre-tokenized format."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B"):
        """
        Initialize converter with Qwen tokenizer.
        
        Args:
            model_name: HuggingFace model name for tokenizer (default: Qwen/Qwen2-0.5B)
        """
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Ensure we have the special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Qwen models don't have a separate BOS token
        # They only use EOS token (which serves as both beginning and end)
        self.bos_token_id = self.tokenizer.bos_token_id  # Will be None for Qwen
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer loaded successfully")
        print(f"  BOS token ID: {self.bos_token_id if self.bos_token_id is not None else 'None (Qwen uses EOS only)'}")
        print(f"  EOS token ID: {self.eos_token_id}")
        print(f"  PAD token ID: {self.tokenizer.pad_token_id}")
        print(f"  Vocab size: {len(self.tokenizer)}")
    
    def format_sharegpt_conversation(self, conversations: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        """
        Format ShareGPT conversation using tokenizer's chat template.
        
        Args:
            conversations: List of dicts with 'from' and 'value' keys
            add_generation_prompt: Whether to add generation prompt at the end
            
        Returns:
            Formatted conversation string
        """
        # Convert ShareGPT format to standard chat format
        chat_messages = []
        
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            
            # Map ShareGPT roles to standard roles
            if role in ["human", "user"]:
                chat_messages.append({"role": "user", "content": content})
            elif role in ["gpt", "assistant"]:
                chat_messages.append({"role": "assistant", "content": content})
            elif role == "system":
                chat_messages.append({"role": "system", "content": content})
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
                return formatted_text
            except Exception as e:
                print(f"Warning: Could not use chat template: {e}")
                # Fall back to manual formatting
        
        # Manual formatting (fallback)
        formatted_text = ""
        for msg in chat_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        
        # Add generation prompt if requested (for manual formatting)
        if add_generation_prompt and formatted_text:
            formatted_text += "<|im_start|>assistant\n"
        
        return formatted_text
    
    def create_tokenized_example(
        self, 
        conversations: List[Dict[str, str]], 
        max_length: int = 2048,
        add_generation_prompt: bool = False,
        verbose: bool = True
    ) -> Dict[str, List[int]]:
        """
        Convert a ShareGPT conversation to pre-tokenized format.
        
        Args:
            conversations: List of conversation turns
            max_length: Maximum sequence length
            add_generation_prompt: Whether to add generation prompt (for inference)
            verbose: Whether to show per-example truncation warnings
            
        Returns:
            Dict with input_ids, attention_mask, and labels
        """
        # Format the conversation
        formatted_text = self.format_sharegpt_conversation(conversations, add_generation_prompt=add_generation_prompt)
        
        # Tokenize the full conversation
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=False,  # We'll add EOS manually (Qwen has no BOS)
            truncation=True,
            max_length=max_length - 1,  # Reserve space for EOS only
            return_attention_mask=False
        )
        
        input_ids = encoding["input_ids"]
        
        # Check if sequence was truncated
        original_length = len(self.tokenizer.encode(formatted_text, add_special_tokens=False))
        if original_length >= max_length - 1 and verbose:
            print(f"⚠️  Warning: Sequence truncated from {original_length} to {len(input_ids)} tokens (max_length={max_length})")
        
        # Add EOS at the end (Qwen doesn't use BOS token)
        input_ids = input_ids + [self.eos_token_id]
        
        # Create attention mask (all 1s for valid tokens)
        attention_mask = [1] * len(input_ids)
        
        # Create labels for training
        # We want to train on assistant responses only
        labels = self.create_labels(conversations, input_ids, formatted_text)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def create_labels(
        self, 
        conversations: List[Dict[str, str]], 
        input_ids: List[int],
        formatted_text: str
    ) -> List[int]:
        """
        Create labels where only assistant responses are trained on.
        User messages and system messages are masked with -100.
        
        Args:
            conversations: Original conversation structure
            input_ids: Tokenized input IDs
            formatted_text: Formatted conversation text
            
        Returns:
            List of labels with -100 for ignored tokens
        """
        labels = input_ids.copy()
        
        # Find positions of assistant responses to unmask them
        current_pos = 0
        
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            
            if role in ["human", "user"]:
                # Mask user messages
                turn_text = f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "system":
                # Mask system messages
                turn_text = f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role in ["gpt", "assistant"]:
                # Keep assistant messages for training
                turn_text = f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                continue
            
            # Tokenize this turn
            turn_encoding = self.tokenizer(
                turn_text,
                add_special_tokens=False,
                return_attention_mask=False
            )
            turn_length = len(turn_encoding["input_ids"])
            
            # Mask non-assistant turns
            if role not in ["gpt", "assistant"]:
                end_pos = min(current_pos + turn_length, len(labels))
                for i in range(current_pos, end_pos):
                    labels[i] = -100
            
            current_pos += turn_length
        
        # Always train on EOS token at the end
        # labels[-1] should remain as eos_token_id (not -100)
        
        return labels
    
    def convert_dataset(
        self,
        dataset_name: str,
        output_path: str,
        max_length: int = 2048,
        num_samples: int = None,
        split: str = "train",
        show_colored: bool = True,
        show_complete: bool = False,
        num_examples_to_show: int = 2,
        add_generation_prompt: bool = False,
        verbose: bool = True
    ):
        """
        Convert a ShareGPT format dataset to pre-tokenized JSONL format.
        
        Args:
            dataset_name: HuggingFace dataset name or local path
            output_path: Output JSONL file path
            max_length: Maximum sequence length
            num_samples: Number of samples to convert (None for all)
            split: Dataset split to use
            show_colored: Whether to show colored output in samples
            show_complete: Whether to show complete text (not truncated)
            num_examples_to_show: Number of examples to display after conversion
            add_generation_prompt: Whether to add generation prompt (for inference)
            verbose: Whether to show per-example truncation warnings
        """
        print(f"\n{'='*60}")
        print(f"Converting dataset: {dataset_name}")
        print(f"Output path: {output_path}")
        print(f"Max length: {max_length}")
        print(f"Split: {split}")
        print(f"{'='*60}\n")
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            print(f"Processing {num_samples} samples...")
        else:
            print(f"Processing all {len(dataset)} samples...")
        
        # Convert and write to JSONL
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        truncated_count = 0
        max_sequence_length = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Converting"):
                try:
                    # Get conversations from the example
                    # FineTome-100k has 'conversations' field
                    if 'conversations' in example:
                        conversations = example['conversations']
                    elif 'messages' in example:
                        conversations = example['messages']
                    else:
                        print(f"Warning: No conversation field found in example")
                        skipped_count += 1
                        continue
                    
                    # Convert to tokenized format
                    tokenized_example = self.create_tokenized_example(
                        conversations,
                        max_length=max_length,
                        add_generation_prompt=add_generation_prompt,
                        verbose=verbose
                    )
                    
                    # Track statistics
                    sequence_length = len(tokenized_example['input_ids'])
                    max_sequence_length = max(max_sequence_length, sequence_length)
                    
                    # Check if this sequence reached max length (likely truncated)
                    if sequence_length >= max_length:
                        truncated_count += 1
                    
                    # Write to JSONL
                    f.write(json.dumps(tokenized_example) + '\n')
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing example: {e}")
                    skipped_count += 1
                    continue
        
        print(f"\n{'='*60}")
        print(f"✓ Conversion complete!")
        print(f"  Processed: {processed_count} examples")
        print(f"  Skipped: {skipped_count} examples")
        print(f"  Max sequence length: {max_sequence_length} tokens")
        if truncated_count > 0:
            print(f"  ⚠️  Truncated: {truncated_count} examples ({100*truncated_count/processed_count:.1f}%)")
            print(f"      Consider increasing --max-length if truncation is too high")
        print(f"  Output: {output_file}")
        print(f"{'='*60}\n")
        
        # Show sample
        self.show_sample(output_file, num_samples=num_examples_to_show, show_colored=show_colored, show_complete=show_complete)
    
    def show_sample(self, jsonl_path: str, num_samples: int = 2, show_colored: bool = True, show_complete: bool = False):
        """
        Display sample tokenized examples from the output file.
        
        Args:
            jsonl_path: Path to JSONL file
            num_samples: Number of samples to display
            show_colored: Whether to show colored decoded text (green=trained, red=masked)
            show_complete: Whether to show complete text (not truncated)
        """
        # ANSI color codes
        GREEN = '\033[92m'  # Trained tokens
        RED = '\033[91m'    # Masked tokens
        RESET = '\033[0m'   # Reset color
        
        print(f"\n{'='*60}")
        print(f"Sample outputs from {jsonl_path}:")
        print(f"{'='*60}\n")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                example = json.loads(line)
                print(f"Example {i+1}:")
                print(f"  Length: {len(example['input_ids'])} tokens")
                print(f"  Input IDs (first 20): {example['input_ids'][:20]}")
                print(f"  Attention Mask (first 20): {example['attention_mask'][:20]}")
                print(f"  Labels (first 20): {example['labels'][:20]}")
                
                # Count training tokens (non -100)
                train_tokens = sum(1 for label in example['labels'] if label != -100)
                print(f"  Training tokens: {train_tokens}/{len(example['labels'])} "
                      f"({100*train_tokens/len(example['labels']):.1f}%)")
                
                # Decode with color coding
                if show_colored:
                    print(f"\n  {GREEN}Green = Trained{RESET} | {RED}Red = Masked{RESET}")
                    if show_complete:
                        print(f"  Complete decoded text with labels:\n")
                    else:
                        print(f"  Decoded text with labels (first 200 tokens):\n")
                    
                    max_tokens = None if show_complete else 200
                    self._print_colored_tokens(example['input_ids'], example['labels'], max_tokens=max_tokens)
                else:
                    # Decode a portion to show text (old method)
                    valid_ids = [tid for tid in example['input_ids'][:50] if tid is not None and tid != -100]
                    if valid_ids:
                        decoded = self.tokenizer.decode(valid_ids)
                        print(f"  Decoded (first 50 tokens): {decoded[:200]}...")
                    else:
                        print(f"  Decoded: (no valid tokens to decode)")
                print()
    
    def _print_colored_tokens(self, input_ids: List[int], labels: List[int], max_tokens: int = None):
        """
        Print tokens with color coding based on labels.
        
        Args:
            input_ids: List of token IDs
            labels: List of labels (-100 for masked, token_id for trained)
            max_tokens: Maximum number of tokens to display (None = all tokens)
        """
        # ANSI color codes
        GREEN = '\033[92m'  # Trained tokens
        RED = '\033[91m'    # Masked tokens
        RESET = '\033[0m'   # Reset color
        
        output_parts = []
        current_color = None
        
        # Process tokens up to max_tokens (or all if None)
        if max_tokens is None:
            display_tokens = len(input_ids)
        else:
            display_tokens = min(len(input_ids), max_tokens)
        
        for idx in range(display_tokens):
            token_id = input_ids[idx]
            label = labels[idx]
            
            # Skip None tokens
            if token_id is None:
                continue
            
            # Decode individual token
            try:
                token_text = self.tokenizer.decode([token_id])
            except:
                token_text = f"[{token_id}]"
            
            # Determine color based on label
            is_trained = (label != -100)
            new_color = GREEN if is_trained else RED
            
            # Add color change if needed
            if new_color != current_color:
                if current_color is not None:
                    output_parts.append(RESET)
                output_parts.append(new_color)
                current_color = new_color
            
            output_parts.append(token_text)
        
        # Reset color at the end
        if current_color is not None:
            output_parts.append(RESET)
        
        # Print the result
        full_text = ''.join(output_parts)
        
        # Wrap text at reasonable width
        import textwrap
        wrapped = textwrap.fill(full_text, width=80, subsequent_indent='    ')
        print(f"    {wrapped}")
        
        if display_tokens < len(input_ids):
            print(f"\n    ... ({len(input_ids) - display_tokens} more tokens)")



def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT format dataset to Qwen pre-tokenized format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mlabonne/FineTome-100k",
        help="HuggingFace dataset name or local path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finetome_qwen_tokenized.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Qwen model name for tokenizer (e.g., Qwen/Qwen2-0.5B, Qwen/Qwen2-1.5B)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to convert (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Add generation prompt at the end (use for inference datasets, not training)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output for sample display"
    )
    parser.add_argument(
        "--show-complete",
        action="store_true",
        help="Show complete decoded text (not truncated to 200 tokens)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=2,
        help="Number of examples to display after conversion (default: 2)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable per-example truncation warnings during conversion"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = ShareGPTToQwenConverter(model_name=args.model)
    
    # Convert dataset
    converter.convert_dataset(
        dataset_name=args.dataset,
        output_path=args.output,
        max_length=args.max_length,
        num_samples=args.num_samples,
        split=args.split,
        show_colored=not args.no_color,
        show_complete=args.show_complete,
        num_examples_to_show=args.num_examples,
        add_generation_prompt=args.add_generation_prompt,
        verbose=not args.no_verbose
    )

if __name__ == "__main__":
    main()
