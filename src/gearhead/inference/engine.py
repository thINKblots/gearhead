"""
Diagnostic inference engine for Gearhead.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from ..model.gearhead_model import GearheadConfig, GearheadModel


class DiagnosticEngine:
    """
    Inference engine for equipment diagnostics.

    Loads a trained Gearhead model and provides high-level
    diagnostic assistance methods.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer,
        device: Optional[str] = None,
    ):
        """
        Initialize diagnostic engine.

        Args:
            model_path: Path to model checkpoint directory
            tokenizer: GearheadTokenizer instance
            device: Device to run inference on (cuda/cpu)
        """
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str) -> GearheadModel:
        """Load model from checkpoint."""
        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        config = torch.load(config_path, map_location="cpu")

        # Load model weights
        weights_path = model_path / "model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        model = GearheadModel(config)
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

        print(f"Model loaded from {model_path}")
        print(f"Parameters: {model.num_parameters():,}")

        return model

    def diagnose(
        self,
        equipment: str,
        symptom: str,
        error_codes: Optional[List[str]] = None,
        max_length: int = 500,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Dict[str, str]:
        """
        Generate diagnostic assistance for a given scenario.

        Args:
            equipment: Equipment type/model
            symptom: Observed symptom or problem
            error_codes: Optional list of error codes
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary with diagnostic information
        """
        # Format input prompt
        prompt_parts = [
            f"{self.tokenizer.EQUIPMENT_TOKEN} {equipment}",
            f"{self.tokenizer.SYMPTOM_TOKEN} {symptom}",
        ]

        if error_codes:
            error_str = ", ".join(error_codes)
            prompt_parts.append(f"{self.tokenizer.ERROR_CODE_TOKEN} {error_str}")

        # Add markers for expected outputs
        prompt_parts.append(self.tokenizer.CAUSE_TOKEN)

        prompt = "\n".join(prompt_parts)

        # Generate response
        response = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Parse response
        parsed = self._parse_diagnostic_response(response)

        return {
            "equipment": equipment,
            "symptom": symptom,
            "error_codes": error_codes or [],
            "probable_cause": parsed.get("cause", ""),
            "solution": parsed.get("solution", ""),
            "full_response": response,
        }

    def generate(
        self,
        prompt: str,
        max_length: int = 500,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
            stop_tokens: Optional list of stop tokens

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=False,
        )

        # Remove the prompt from output
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def _parse_diagnostic_response(self, response: str) -> Dict[str, str]:
        """
        Parse structured diagnostic response.

        Args:
            response: Generated response text

        Returns:
            Dictionary with parsed fields
        """
        result = {}

        # Extract cause
        if self.tokenizer.CAUSE_TOKEN in response:
            cause_start = response.find(self.tokenizer.CAUSE_TOKEN)
            cause_start += len(self.tokenizer.CAUSE_TOKEN)

            # Find end (next token or end of string)
            cause_end = len(response)
            if self.tokenizer.SOLUTION_TOKEN in response[cause_start:]:
                cause_end = response.find(self.tokenizer.SOLUTION_TOKEN, cause_start)

            result["cause"] = response[cause_start:cause_end].strip()

        # Extract solution
        if self.tokenizer.SOLUTION_TOKEN in response:
            solution_start = response.find(self.tokenizer.SOLUTION_TOKEN)
            solution_start += len(self.tokenizer.SOLUTION_TOKEN)

            # Find end (EOS token or end of string)
            solution_end = len(response)
            if self.tokenizer.EOS_TOKEN in response[solution_start:]:
                solution_end = response.find(self.tokenizer.EOS_TOKEN, solution_start)

            result["solution"] = response[solution_start:solution_end].strip()

        return result

    def batch_diagnose(
        self,
        scenarios: List[Dict[str, Union[str, List[str]]]],
        max_length: int = 500,
        temperature: float = 0.7,
    ) -> List[Dict[str, str]]:
        """
        Run batch diagnostic inference.

        Args:
            scenarios: List of scenario dicts with 'equipment', 'symptom', 'error_codes'
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            List of diagnostic results
        """
        results = []

        for scenario in scenarios:
            result = self.diagnose(
                equipment=scenario.get("equipment", ""),
                symptom=scenario.get("symptom", ""),
                error_codes=scenario.get("error_codes"),
                max_length=max_length,
                temperature=temperature,
            )
            results.append(result)

        return results

    def explain_error_code(
        self,
        error_code: str,
        equipment: Optional[str] = None,
        max_length: int = 300,
    ) -> str:
        """
        Get explanation for an error code.

        Args:
            error_code: Error code to explain
            equipment: Optional equipment context
            max_length: Maximum generation length

        Returns:
            Error code explanation
        """
        prompt_parts = []

        if equipment:
            prompt_parts.append(f"{self.tokenizer.EQUIPMENT_TOKEN} {equipment}")

        prompt_parts.append(f"{self.tokenizer.ERROR_CODE_TOKEN} {error_code}")
        prompt_parts.append(self.tokenizer.CAUSE_TOKEN)

        prompt = "\n".join(prompt_parts)

        explanation = self.generate(
            prompt,
            max_length=max_length,
            temperature=0.5,  # Lower temperature for more factual responses
        )

        return explanation
