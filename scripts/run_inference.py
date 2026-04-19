#!/usr/bin/env python3
"""
CLI script for running Llama model inference.

This script provides a command-line interface to run text generation with
the Llama language model.

Examples:
    # Run with default model and prompt
    python run_inference.py
    
    # Run with a custom prompt
    python run_inference.py --prompt "Once upon a time"
    
    # Run with custom max tokens
    python run_inference.py --prompt "The answer is" --max-tokens 50
    
    # Run with CPU device
    python run_inference.py --device cpu --max-tokens 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from therapml.phase2.part1.inference import InferenceApp
from therapml.phase2.part1.inference import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_PROMPT,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a Llama language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the saved model (.pth file). Default: {DEFAULT_MODEL_PATH}",
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Path to the saved tokenizer (.json file). Default: {DEFAULT_TOKENIZER_PATH}",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=f"Input prompt for generation. Default: '{DEFAULT_PROMPT}'",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate. Default: 100",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run inference on. Default: auto-detect (CUDA if available)",
    )
    
    args = parser.parse_args()
    
    try:
        print("Initializing inference app...")
        app = InferenceApp(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
        )
        
        print(f"✓ Model loaded from: {args.model_path}")
        print(f"✓ Tokenizer loaded from: {args.tokenizer_path}")
        print(f"✓ Device: {app.inference.device}")
        print(f"✓ Vocabulary size: {app.inference.vocab_size}")
        print()
        
        # Run inference
        print("Running inference...")
        result = app.run(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
        )
        
        print("-" * 80)
        print(f"Input Prompt:\n{result['prompt']}\n")
        print(f"Generated Text:\n{result['generated_text']}\n")
        print(f"Tokens Generated: {result['num_tokens_generated']}")
        print("-" * 80)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
