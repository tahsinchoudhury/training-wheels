"""
Inference module for running generation with Llama language models.

Usage:
    from therapml.phase2.part1.inference import InferenceApp
    
    app = InferenceApp()
    result = app.run(prompt="Once upon a time", max_new_tokens=100)
    print(result["generated_text"])
"""

import argparse
from pathlib import Path
import sys
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from therapml.phase2.part1.models.llama import Llama
    from therapml.phase2.part1.tokenizer.bpe import BPETokenizer
else:
    from .models.llama import Llama
    from .tokenizer.bpe import BPETokenizer


# Default paths (relative to project root)
DEFAULT_MODEL_PATH = "models/llama_phase2_part1.pth"
DEFAULT_TOKENIZER_PATH = "data/tinystories_bpe.json"
DEFAULT_PROMPT = "The quick brown fox"


class LLMInference:
    """
    Base class for language model inference.
    
    Handles model loading, tokenization, and text generation.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str | None = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the saved model (.pth file)
            tokenizer_path: Path to the saved tokenizer (.json file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        
        Raises:
            FileNotFoundError: If model or tokenizer paths don't exist
            RuntimeError: If model loading fails
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        
        # Validate paths exist
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {self.tokenizer_path}")
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        self.model = Llama.load(self.model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BPETokenizer.load(self.tokenizer_path)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input text to continue from
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            Generated text (decoded output)
        """
        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(prompt)
        
        # Convert to tensor with batch dimension
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        
        # Generate tokens
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                device=self.device,
                max_new_tokens=max_new_tokens,
            )
        
        # Decode the output (only the generated tokens after the prompt)
        generated_ids = output_ids[0, len(prompt_ids):].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text


class LlamaInference(LLMInference):
    """
    Concrete implementation of LLMInference for Llama models.
    
    Inherits all functionality from LLMInference with Llama-specific optimizations.
    """
    pass


class InferenceApp:
    """
    High-level application for running inference with user-friendly interface.
    
    Provides:
    - Model loading with sensible defaults
    - Input validation and error handling
    - User-friendly generation API
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        default_prompt: str = DEFAULT_PROMPT,
        device: str | None = None,
    ):
        """
        Initialize the inference app.
        
        Args:
            model_path: Path to the saved model (.pth file)
            tokenizer_path: Path to the saved tokenizer (.json file)
            default_prompt: Default prompt to use if none is provided in run()
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        
        Raises:
            FileNotFoundError: If model or tokenizer paths don't exist
            RuntimeError: If model loading fails
        """
        self.default_prompt = default_prompt
        self.inference = LlamaInference(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )
    
    def run(
        self,
        prompt: str | None = None,
        max_new_tokens: int = 100,
    ) -> dict:
        """
        Run inference with the given prompt.
        
        Args:
            prompt: Input text to continue from. If None, uses default_prompt
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            Dictionary containing:
            - "prompt": The input prompt used
            - "generated_text": The generated continuation
            - "num_tokens_generated": Number of tokens generated
        """
        if prompt is None:
            prompt = self.default_prompt
        
        # Validate input
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        
        # Generate text
        generated_text = self.inference.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        
        # Count tokens in generated text
        generated_ids = self.inference.tokenizer.encode(generated_text)
        num_tokens = len(generated_ids)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens_generated": num_tokens,
        }


def main() -> int:
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
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
