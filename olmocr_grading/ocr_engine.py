"""
Model loading and inference wrapper around allenai/olmOCR-2-7B-1025.

olmOCR-2-7B-1025 is a Qwen2.5-VL-7B-Instruct fine-tune, so it is loaded
with the Qwen2.5-VL model class. Per the official model card, the
processor is loaded from the base "Qwen/Qwen2.5-VL-7B-Instruct" repo
rather than from the olmOCR checkpoint (which doesn't ship its own
processor files).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from .config import OCRConfig
from .prompts import get_prompt

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^\s*---.*?---\s*\n?", re.DOTALL)


def _is_oom_error(e: Exception) -> bool:
    """Detect an out-of-memory RuntimeError across torch versions/devices
    (covers CUDA, MPS, and CPU allocator OOM messages)."""
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg or "mps backend out of memory" in msg


@dataclass
class PageResult:
    page: int
    text: str
    confidence: Optional[float] = None
    raw_metadata: Optional[str] = None  # stock-prompt YAML front matter, if used


class OlmOCREngine:
    """Thin wrapper: load once, call transcribe() per page image."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype, self.device)
        self.model = None
        self.processor = None

    # ---------- setup ----------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
        if dtype != "auto":
            return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            # bfloat16 support on MPS varies by torch/macOS version; float16 is safer.
            return torch.float16
        return torch.float32  # CPU

    def load(self) -> None:
        """Load model + processor. Raises a clear error on common failure modes."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(
            "Loading model '%s' (device=%s, dtype=%s)...",
            self.config.model_id,
            self.device,
            self.dtype,
        )
        try:
            model_kwargs = {"torch_dtype": self.dtype}
            if self.config.attn_implementation:
                model_kwargs["attn_implementation"] = self.config.attn_implementation

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_id, **model_kwargs
            ).eval()
            self.model.to(self.device)
        except OSError as e:
            raise RuntimeError(
                f"Could not load model '{self.config.model_id}'. Check the model id and your "
                f"internet connection / Hugging Face auth (huggingface-cli login) if it's gated. "
                f"Original error: {e}"
            ) from e
        except RuntimeError as e:
            if _is_oom_error(e):
                raise RuntimeError(
                    "Out of GPU memory while loading the model. Try the FP8 checkpoint "
                    "'allenai/olmOCR-2-7B-1025-FP8' instead, or run on CPU (--device cpu, slow)."
                ) from e
            raise

        try:
            self.processor = AutoProcessor.from_pretrained(self.config.processor_id)
        except OSError as e:
            raise RuntimeError(
                f"Could not load processor '{self.config.processor_id}'. Original error: {e}"
            ) from e

        logger.info("Model + processor loaded.")

    # ---------- inference ----------

    def transcribe(self, image: Image.Image, page_number: int = 1) -> PageResult:
        """Run OCR on a single page image and return a PageResult."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Call .load() before .transcribe().")

        prompt_text = get_prompt(self.config.use_stock_prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": image},
                ],
            }
        ]

        try:
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            )
            if self.config.do_sample:
                gen_kwargs["temperature"] = self.config.temperature

            confidence = None
            with torch.inference_mode():
                if self.config.compute_confidence:
                    output = self.model.generate(
                        **inputs,
                        **gen_kwargs,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    sequences = output.sequences
                    confidence = self._compute_confidence(sequences, output.scores, inputs["input_ids"].shape[1])
                else:
                    sequences = self.model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = sequences[:, prompt_len:]
            raw_text = self.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        except RuntimeError as e:
            if _is_oom_error(e):
                raise RuntimeError(
                    f"Out of memory generating page {page_number}. Try a smaller "
                    f"--target-longest-dim, fewer --max-new-tokens, or the FP8 model checkpoint."
                ) from e
            raise

        text, metadata = self._strip_front_matter(raw_text) if self.config.use_stock_prompt else (raw_text, None)

        return PageResult(page=page_number, text=text, confidence=confidence, raw_metadata=metadata)

    # ---------- helpers ----------

    @staticmethod
    def _strip_front_matter(raw_text: str) -> tuple[str, Optional[str]]:
        """The stock olmOCR prompt prefixes output with a '---\\n...\\n---' YAML
        block (primary_language, is_table, is_diagram, ...). Split it off so
        the saved output is just the transcription."""
        match = _FRONT_MATTER_RE.match(raw_text)
        if not match:
            return raw_text, None
        metadata = match.group(0).strip()
        body = raw_text[match.end():].strip()
        return body, metadata

    @staticmethod
    def _compute_confidence(sequences, scores, prompt_len: int) -> Optional[float]:
        """Heuristic mean per-token generation confidence (exp of average
        log-prob of the chosen tokens). This is NOT a calibrated
        probability that the transcription is correct -- just a rough
        signal for flagging low-confidence pages for human review."""
        try:
            import torch.nn.functional as F

            log_probs = []
            for step_scores, token_id in zip(scores, sequences[0][prompt_len:]):
                step_log_probs = F.log_softmax(step_scores[0], dim=-1)
                log_probs.append(step_log_probs[token_id].item())
            if not log_probs:
                return None
            avg_log_prob = sum(log_probs) / len(log_probs)
            return float(torch.exp(torch.tensor(avg_log_prob)))
        except Exception:
            logger.debug("Confidence computation failed; skipping.", exc_info=True)
            return None
