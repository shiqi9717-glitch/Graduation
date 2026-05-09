"""Transformers-based local open-model probe runner."""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalProbeConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    max_length: int = 2048
    top_k: int = 12
    hidden_state_layers: tuple[int, ...] = (-1, -2, -3, -4)


class LocalProbeRunner:
    """Lazy-loaded local causal LM runner for white-box probing."""

    def __init__(self, config: LocalProbeConfig) -> None:
        self.config = config
        self._torch = None
        self._transformers = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._active_dtype = None
        self._warned_forced_float32 = False

    def _lazy_imports(self) -> None:
        if self._torch is None:
            import torch

            self._torch = torch
        if self._transformers is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._transformers = {
                "AutoModelForCausalLM": AutoModelForCausalLM,
                "AutoTokenizer": AutoTokenizer,
            }

    def _repair_custom_model_buffers(self) -> None:
        """Materialize custom-code buffers that may remain on the meta device."""
        assert self._model is not None
        assert self._device is not None
        for module in self._model.modules():
            has_rotary_cache = all(hasattr(module, name) for name in ("cos_cached", "sin_cached", "max_seq_len_cached"))
            if not has_rotary_cache:
                continue
            cos_cached = getattr(module, "cos_cached", None)
            sin_cached = getattr(module, "sin_cached", None)
            inv_freq = getattr(module, "inv_freq", None)
            needs_rebuild = bool(
                getattr(cos_cached, "is_meta", False)
                or getattr(sin_cached, "is_meta", False)
                or getattr(inv_freq, "is_meta", False)
            )
            if not needs_rebuild:
                continue
            if inv_freq is not None and hasattr(inv_freq, "shape") and len(inv_freq.shape) == 1 and int(inv_freq.shape[0]) > 0:
                dim = int(inv_freq.shape[0]) * 2
            elif cos_cached is not None and hasattr(cos_cached, "shape") and len(cos_cached.shape) >= 4 and int(cos_cached.shape[-1]) > 0:
                dim = int(cos_cached.shape[-1])
            else:
                continue
            seq_len = int(getattr(module, "max_seq_len_cached", 2048) or 2048)
            base = float(getattr(module, "base", 10000.0) or 10000.0)
            inv_freq = 1.0 / (
                base ** (self._torch.arange(0, dim, 2, device=self._device, dtype=self._torch.float32) / dim)
            )
            t = self._torch.arange(seq_len, device=self._device, dtype=self._torch.float32)
            freqs = self._torch.outer(t, inv_freq)
            emb = self._torch.cat((freqs, freqs), dim=-1)
            module.inv_freq = inv_freq
            module.cos_cached = emb.cos()[None, None, :, :].to(self._torch.float32)
            module.sin_cached = emb.sin()[None, None, :, :].to(self._torch.float32)

    def _resolve_device(self) -> str:
        self._lazy_imports()
        if self.config.device != "auto":
            requested_device = str(self.config.device)
            if requested_device == "mps" and not self._torch.backends.mps.is_available():
                if not self._warned_forced_float32:
                    warnings.warn(
                        "Requested MPS device is unavailable in this session; falling back to CPU.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_forced_float32 = True
                return "cpu"
            return requested_device
        if self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self):
        self._lazy_imports()
        if self._should_force_float32_on_mps():
            return self._torch.float32
        if self.config.dtype != "auto":
            return getattr(self._torch, str(self.config.dtype))
        if self._resolve_device() == "mps":
            return self._torch.float32 if self._prefer_float32_on_mps() else self._torch.float16
        return self._torch.float32

    def _estimated_model_params_b(self) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", str(self.config.model_name))
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _prefer_float32_on_mps(self) -> bool:
        params_b = self._estimated_model_params_b()
        return params_b is not None and params_b >= 7.0

    def _should_force_float32_on_mps(self) -> bool:
        requested_dtype = str(self.config.dtype).strip().lower()
        return (
            self._resolve_device() == "mps"
            and self._prefer_float32_on_mps()
            and requested_dtype in {"auto", "float16"}
        )

    def _resolve_pretrained_source(self) -> str:
        model_name = str(self.config.model_name)
        cache_root = os.path.expanduser("~/.cache/huggingface/hub")
        cache_dir = os.path.join(cache_root, "models--" + model_name.replace("/", "--"))
        refs_main = os.path.join(cache_dir, "refs", "main")
        if os.path.exists(refs_main):
            revision = Path(refs_main).read_text(encoding="utf-8").strip()
            snapshot_dir = os.path.join(cache_dir, "snapshots", revision)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir
        return model_name

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._lazy_imports()
        tokenizer_cls = self._transformers["AutoTokenizer"]
        model_cls = self._transformers["AutoModelForCausalLM"]
        pretrained_source = self._resolve_pretrained_source()
        tokenizer_attempts = [
            {"trust_remote_code": True, "use_fast": True},
            {"trust_remote_code": True, "use_fast": False},
            {"trust_remote_code": True, "use_fast": True, "local_files_only": True},
            {"trust_remote_code": True, "use_fast": False, "local_files_only": True},
        ]
        last_tokenizer_error: Exception | None = None
        for kwargs in tokenizer_attempts:
            try:
                self._tokenizer = tokenizer_cls.from_pretrained(pretrained_source, **kwargs)
                last_tokenizer_error = None
                break
            except Exception as exc:
                last_tokenizer_error = exc
        if self._tokenizer is None:
            assert last_tokenizer_error is not None
            raise last_tokenizer_error
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._device = self._resolve_device()
        if self._should_force_float32_on_mps() and not self._warned_forced_float32:
            warnings.warn(
                (
                    f"Forcing float32 for {self.config.model_name} on MPS because "
                    "7B+ Qwen runs were numerically unstable under float16."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_forced_float32 = True
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if self._device == "cpu":
            model_kwargs["torch_dtype"] = self._resolve_dtype()
        else:
            model_kwargs["torch_dtype"] = self._resolve_dtype()
        try:
            self._model = model_cls.from_pretrained(pretrained_source, **model_kwargs)
        except Exception:
            self._model = model_cls.from_pretrained(
                pretrained_source,
                local_files_only=True,
                **model_kwargs,
            )
        self._model.to(self._device)
        self._repair_custom_model_buffers()
        self._active_dtype = self._resolve_dtype()
        self._model.eval()

    def _ensure_float32_stability(self) -> bool:
        self.load()
        assert self._model is not None
        if self._device != "mps" or self._active_dtype == self._torch.float32:
            return False
        self._model.to(dtype=self._torch.float32)
        self._active_dtype = self._torch.float32
        self._model.eval()
        return True

    def _tensor_has_nonfinite(self, value) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return any(self._tensor_has_nonfinite(item) for item in value)
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            return not bool(self._torch.isfinite(value).all().item())
        return False

    def _outputs_have_nonfinite(self, outputs) -> bool:
        return any(
            self._tensor_has_nonfinite(value)
            for value in (
                getattr(outputs, "logits", None),
                getattr(outputs, "hidden_states", None),
                getattr(outputs, "attentions", None),
            )
        )

    def _forward(self, **model_kwargs):
        self.load()
        assert self._model is not None
        # Some custom-code checkpoints (for example InternLM2) assume cache
        # helper APIs from a narrower transformers version range. We do
        # correctness-first probing and never rely on KV cache, so force it off
        # unless a caller explicitly overrides it.
        model_kwargs.setdefault("use_cache", False)
        model_kwargs.setdefault("return_dict", True)
        with self._torch.no_grad():
            outputs = self._model(**model_kwargs)
        if self._outputs_have_nonfinite(outputs) and self._ensure_float32_stability():
            with self._torch.no_grad():
                outputs = self._model(**model_kwargs)
        return outputs

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def _resolve_model_parts(self):
        assert self._model is not None
        core_model = getattr(self._model, "model", None) or getattr(self._model, "transformer", None)
        if core_model is None:
            raise ValueError("Unsupported model structure: missing core decoder module.")
        decoder_layers = getattr(core_model, "layers", None) or getattr(core_model, "h", None)
        if decoder_layers is None:
            raise ValueError("Unsupported model structure: missing decoder layers.")
        final_norm = getattr(core_model, "norm", None) or getattr(core_model, "ln_f", None)
        lm_head = (
            getattr(self._model, "lm_head", None)
            or getattr(self._model, "embed_out", None)
            or getattr(self._model, "output", None)
        )
        if lm_head is None and hasattr(self._model, "get_output_embeddings"):
            lm_head = self._model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("Unsupported model structure: missing output embedding head.")
        return core_model, decoder_layers, final_norm, lm_head

    def _layer_self_attention(self, layer_index: int):
        _, decoder_layers, _, _ = self._resolve_model_parts()
        layer = decoder_layers[int(layer_index)]
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            raise ValueError(f"Unsupported layer structure at layer {layer_index}: missing self attention module.")
        return attn

    def _layer_mlp(self, layer_index: int):
        _, decoder_layers, _, _ = self._resolve_model_parts()
        layer = decoder_layers[int(layer_index)]
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is None:
            raise ValueError(f"Unsupported layer structure at layer {layer_index}: missing MLP/feed-forward module.")
        return mlp

    def attention_head_layout(self, layer_index: int) -> Dict[str, int]:
        attn = self._layer_self_attention(int(layer_index))
        model_config = getattr(self._model, "config", None)
        num_heads = int(
            getattr(attn, "num_heads", 0)
            or getattr(attn, "num_attention_heads", 0)
            or getattr(model_config, "num_attention_heads", 0)
            or getattr(model_config, "num_heads", 0)
        )
        hidden_size = int(getattr(attn, "hidden_size", 0) or getattr(attn, "embed_dim", 0) or getattr(attn, "hidden_size_per_partition", 0) or 0)
        head_dim = int(getattr(attn, "head_dim", 0) or (hidden_size // max(num_heads, 1)))
        return {
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size or (num_heads * head_dim),
        }

    def _candidate_option_token_ids(self, option: str) -> List[int]:
        assert self._tokenizer is not None
        candidates = []
        for text in (option, f" {option}", f"\n{option}"):
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                candidates.append(token_ids[0])
        if not candidates:
            raise ValueError(f"Could not resolve single-token ids for option {option!r}")
        return sorted(set(candidates))

    def _option_token_id_map(self) -> Dict[str, List[int]]:
        return {option: self._candidate_option_token_ids(option) for option in ("A", "B", "C", "D")}

    def _top_token_logits(self, logits, top_k: int) -> List[Dict[str, Any]]:
        assert self._tokenizer is not None
        values, indices = self._torch.topk(logits, k=min(int(top_k), int(logits.shape[-1])))
        rows = []
        for token_id, value in zip(indices.tolist(), values.tolist()):
            rows.append(
                {
                    "token_id": int(token_id),
                    "token_text": self._tokenizer.decode([int(token_id)]),
                    "logit": float(value),
                }
            )
        return rows

    def _extract_hidden_arrays(self, hidden_states, selected_layers: Iterable[int]) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        for layer_idx in selected_layers:
            layer_tensor = hidden_states[layer_idx][0].detach().float().cpu().numpy()
            arrays[f"layer_{layer_idx}_full_sequence"] = layer_tensor
            arrays[f"layer_{layer_idx}_final_token"] = layer_tensor[-1]
            arrays[f"layer_{layer_idx}_pooled_mean"] = layer_tensor.mean(axis=0)
        return arrays

    def _hidden_state_summary(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in arrays.items():
            summary[key] = {
                "shape": list(value.shape),
                "norm": float(np.linalg.norm(value)),
                "mean": float(value.mean()) if value.size else 0.0,
                "std": float(value.std()) if value.size else 0.0,
            }
        return summary

    def _find_subsequence(self, sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, int] | None:
        if not subsequence:
            return None
        needle = list(subsequence)
        haystack = list(sequence)
        limit = len(haystack) - len(needle) + 1
        for start in range(max(limit, 0)):
            if haystack[start : start + len(needle)] == needle:
                return start, start + len(needle)
        return None

    def _extract_token_positions(self, input_ids: List[int], token_ids: Iterable[int]) -> List[int]:
        token_id_set = {int(token_id) for token_id in token_ids}
        return [idx for idx, token_id in enumerate(input_ids) if int(token_id) in token_id_set]

    def _extract_span_positions(self, span: tuple[int, int] | None) -> List[int]:
        if span is None:
            return []
        start, end = span
        return list(range(int(start), int(end)))

    def _wrong_option_question_positions(
        self,
        *,
        input_ids: List[int],
        question_text: str,
        wrong_option: str,
        option_token_map: Dict[str, List[int]],
    ) -> List[int]:
        assert self._tokenizer is not None
        question_ids = self._tokenizer.encode(str(question_text or ""), add_special_tokens=False)
        question_span = self._find_subsequence(input_ids, question_ids)
        question_positions = set(self._extract_span_positions(question_span))
        wrong_positions = self._extract_token_positions(input_ids, option_token_map.get(str(wrong_option or "").strip().upper(), []))
        return [pos for pos in wrong_positions if pos in question_positions]

    def _summarize_attention_to_positions(self, attention_matrix: np.ndarray, positions: List[int]) -> Dict[str, float]:
        if not positions:
            return {"mean": 0.0, "max_head_mean": 0.0}
        selected = attention_matrix[:, positions]
        per_head = selected.mean(axis=1)
        return {
            "mean": float(selected.mean()),
            "max_head_mean": float(per_head.max()),
        }

    def _answer_logits_from_last_logits(self, last_logits, option_token_map: Dict[str, List[int]]) -> Dict[str, float]:
        answer_logits: Dict[str, float] = {}
        for option, token_ids in option_token_map.items():
            answer_logits[option] = float(max(last_logits[token_id].item() for token_id in token_ids))
        return answer_logits

    def _project_hidden_state_to_option_logits(
        self,
        hidden_state,
        *,
        final_norm,
        lm_head,
        option_token_map: Dict[str, List[int]],
        apply_final_norm: bool,
    ) -> Dict[str, float]:
        if hidden_state.ndim == 2:
            hidden_state = hidden_state.unsqueeze(0)
        hidden_tensor = hidden_state
        if apply_final_norm and final_norm is not None:
            hidden_tensor = final_norm(hidden_tensor)
        logits = lm_head(hidden_tensor[:, -1:, :])[0, -1, :].detach().float().cpu()
        return self._answer_logits_from_last_logits(logits, option_token_map)

    def _rank_options(self, answer_logits: Dict[str, float]) -> List[str]:
        return [item[0] for item in sorted(answer_logits.items(), key=lambda item: item[1], reverse=True)]

    def _collect_layer_logit_lens(
        self,
        *,
        hidden_states,
        final_norm,
        lm_head,
        option_token_map: Dict[str, List[int]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        total_layers = max(len(hidden_states) - 1, 0)
        for layer_index in range(total_layers):
            hidden_state = hidden_states[layer_index + 1]
            apply_final_norm = layer_index < total_layers - 1
            answer_logits = self._project_hidden_state_to_option_logits(
                hidden_state,
                final_norm=final_norm,
                lm_head=lm_head,
                option_token_map=option_token_map,
                apply_final_norm=apply_final_norm,
            )
            rows.append(
                {
                    "layer_index": int(layer_index),
                    "answer_logits": answer_logits,
                    "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
                    "ranked_options": self._rank_options(answer_logits),
                }
            )
        return rows

    def _collect_layer_hidden_arrays(self, hidden_states) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        total_layers = max(len(hidden_states) - 1, 0)
        for layer_index in range(total_layers):
            layer_tensor = hidden_states[layer_index + 1][0].detach().float().cpu().numpy()
            arrays[f"layer_{layer_index}_final_token"] = layer_tensor[-1]
            arrays[f"layer_{layer_index}_pooled_mean"] = layer_tensor.mean(axis=0)
        return arrays

    def _collect_selected_layer_hidden_arrays(self, hidden_states, selected_layers: Iterable[int]) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        total_layers = max(len(hidden_states) - 1, 0)
        for layer_index in selected_layers:
            layer_index = int(layer_index)
            if layer_index < 0 or layer_index >= total_layers:
                raise ValueError(f"Layer index {layer_index} out of range for model with {total_layers} decoder layers.")
            layer_tensor = hidden_states[layer_index + 1][0].detach().float().cpu().numpy()
            arrays[f"layer_{layer_index}_final_token"] = layer_tensor[-1]
            arrays[f"layer_{layer_index}_pooled_mean"] = layer_tensor.mean(axis=0)
        return arrays

    def _collect_attention_summary(
        self,
        *,
        attentions,
        input_ids: List[int],
        prompt_prefix: str,
        question_text: str,
        correct_option: str,
        wrong_option: str,
        option_token_map: Dict[str, List[int]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        if not attentions:
            return [], {}
        assert self._tokenizer is not None
        prefix_ids = self._tokenizer.encode(str(prompt_prefix or ""), add_special_tokens=False)
        question_ids = self._tokenizer.encode(str(question_text or ""), add_special_tokens=False)
        prefix_span = self._find_subsequence(input_ids, prefix_ids)
        question_span = self._find_subsequence(input_ids, question_ids)
        prefix_positions = self._extract_span_positions(prefix_span)
        question_positions = self._extract_span_positions(question_span)
        wrong_option_positions = self._extract_token_positions(input_ids, option_token_map.get(str(wrong_option or ""), []))
        correct_option_positions = self._extract_token_positions(input_ids, option_token_map.get(str(correct_option or ""), []))
        wrong_option_prefix_positions = [pos for pos in wrong_option_positions if pos in set(prefix_positions)]

        rows: List[Dict[str, Any]] = []
        arrays: Dict[str, np.ndarray] = {}
        for layer_index, attention_tensor in enumerate(attentions):
            final_to_all = attention_tensor[0, :, -1, :].detach().float().cpu().numpy()
            arrays[f"layer_{layer_index}_final_token_attention"] = final_to_all
            prefix_summary = self._summarize_attention_to_positions(final_to_all, prefix_positions)
            question_summary = self._summarize_attention_to_positions(final_to_all, question_positions)
            wrong_summary = self._summarize_attention_to_positions(final_to_all, wrong_option_positions)
            correct_summary = self._summarize_attention_to_positions(final_to_all, correct_option_positions)
            wrong_prefix_summary = self._summarize_attention_to_positions(final_to_all, wrong_option_prefix_positions)
            rows.append(
                {
                    "layer_index": int(layer_index),
                    "prefix_attention_mean": prefix_summary["mean"],
                    "prefix_attention_max_head_mean": prefix_summary["max_head_mean"],
                    "question_attention_mean": question_summary["mean"],
                    "wrong_option_attention_mean": wrong_summary["mean"],
                    "correct_option_attention_mean": correct_summary["mean"],
                    "wrong_option_prefix_attention_mean": wrong_prefix_summary["mean"],
                }
            )
        return rows, arrays

    def _encode_prompt(self, prompt: str):
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config.max_length),
        )
        return {key: value.to(self._device) for key, value in encoded.items()}

    def _capture_attention_pre_o_proj(
        self,
        *,
        prompt: str,
        target_layers: Iterable[int],
    ) -> Dict[int, np.ndarray]:
        self.load()
        encoded = self._encode_prompt(prompt)
        captured: Dict[int, np.ndarray] = {}
        handles = []

        def make_hook(layer_index: int):
            def _hook(_module, inputs):
                hidden = inputs[0].detach().float().cpu().numpy()
                captured[int(layer_index)] = hidden[0]
                return None

            return _hook

        for layer_index in target_layers:
            attn = self._layer_self_attention(int(layer_index))
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise ValueError(f"Layer {layer_index} missing o_proj; cannot capture head outputs.")
            handles.append(o_proj.register_forward_pre_hook(make_hook(int(layer_index))))

        try:
            self._forward(**encoded, output_hidden_states=False, output_attentions=False)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def capture_layer_token_residuals(
        self,
        *,
        prompt: str,
        target_layers: Iterable[int],
        token_positions: Iterable[int],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        self.load()
        encoded = self._encode_prompt(prompt)
        wanted_layers = {int(layer) for layer in target_layers}
        wanted_positions = [int(pos) for pos in token_positions]
        captured: Dict[int, Dict[int, np.ndarray]] = {}
        outputs = self._forward(**encoded, output_hidden_states=True, output_attentions=False)
        total_layers = max(len(outputs.hidden_states) - 1, 0)
        for layer_index in wanted_layers:
            if layer_index < 0 or layer_index >= total_layers:
                continue
            layer_tensor = outputs.hidden_states[layer_index + 1][0].detach().float().cpu().numpy()
            captured[layer_index] = {}
            for position in wanted_positions:
                if 0 <= position < layer_tensor.shape[0]:
                    captured[layer_index][position] = layer_tensor[position]
        return captured

    def capture_attention_head_outputs(
        self,
        *,
        prompt: str,
        target_layers: Iterable[int],
    ) -> Dict[int, np.ndarray]:
        return self._capture_attention_pre_o_proj(prompt=prompt, target_layers=target_layers)

    def capture_attention_output(
        self,
        *,
        prompt: str,
        target_layers: Iterable[int],
    ) -> Dict[int, np.ndarray]:
        raw = self._capture_attention_pre_o_proj(prompt=prompt, target_layers=target_layers)
        captured: Dict[int, np.ndarray] = {}
        for layer_index, value in raw.items():
            tensor = np.asarray(value, dtype=np.float32)
            if tensor.ndim == 1:
                captured[int(layer_index)] = tensor
            elif tensor.ndim == 2:
                captured[int(layer_index)] = tensor[-1]
            else:
                raise ValueError(
                    f"Layer {layer_index}: expected attention capture with ndim 1/2, got shape {tensor.shape}."
                )
        return captured

    def capture_mlp_output(
        self,
        *,
        prompt: str,
        target_layers: Iterable[int],
    ) -> Dict[int, np.ndarray]:
        self.load()
        encoded = self._encode_prompt(prompt)
        captured: Dict[int, np.ndarray] = {}
        handles = []

        def make_hook(layer_index: int):
            def _hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                tensor = hidden.detach().float().cpu().numpy()
                captured[int(layer_index)] = tensor[0, -1]
                return None

            return _hook

        for layer_index in target_layers:
            mlp = self._layer_mlp(int(layer_index))
            handles.append(mlp.register_forward_hook(make_hook(int(layer_index))))

        try:
            self._forward(**encoded, output_hidden_states=False, output_attentions=False)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def locate_wrong_option_question_positions(
        self,
        *,
        prompt: str,
        question_text: str,
        wrong_option: str,
    ) -> List[int]:
        self.load()
        assert self._tokenizer is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()
        return self._wrong_option_question_positions(
            input_ids=encoded["input_ids"][0].detach().cpu().tolist(),
            question_text=question_text,
            wrong_option=wrong_option,
            option_token_map=option_token_map,
        )

    def _extract_base_result(
        self,
        *,
        sample_id: str,
        scenario: str,
        question_text: str,
        prompt_prefix: str,
        ground_truth: str,
        wrong_option: str,
        answer_logits: Dict[str, float],
        last_logits,
    ) -> Dict[str, Any]:
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        predicted_answer = max(answer_logits.items(), key=lambda item: item[1])[0]
        return {
            "sample_id": str(sample_id),
            "scenario": str(scenario),
            "model_name": self.model_name,
            "question_text": str(question_text),
            "prompt_prefix": str(prompt_prefix),
            "ground_truth": correct_option,
            "wrong_option": wrong_option_norm,
            "predicted_answer": predicted_answer,
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def analyze_prompt(
        self,
        *,
        prompt: str,
        sample_id: str,
        scenario: str,
        question_text: str,
        prompt_prefix: str,
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()
        outputs = self._forward(**encoded, output_hidden_states=True)
        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        hidden_arrays = self._extract_hidden_arrays(outputs.hidden_states, self.config.hidden_state_layers)
        result = self._extract_base_result(
            sample_id=sample_id,
            scenario=scenario,
            question_text=question_text,
            prompt_prefix=prompt_prefix,
            ground_truth=ground_truth,
            wrong_option=wrong_option,
            answer_logits=answer_logits,
            last_logits=last_logits,
        )
        result["hidden_state_summary"] = self._hidden_state_summary(hidden_arrays)
        result["_hidden_state_arrays"] = hidden_arrays
        return result

    def analyze_prompt_detailed(
        self,
        *,
        prompt: str,
        sample_id: str,
        scenario: str,
        question_text: str,
        prompt_prefix: str,
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        assert self._device is not None
        encoded = self._encode_prompt(prompt)
        _, _, final_norm, lm_head = self._resolve_model_parts()
        option_token_map = self._option_token_id_map()
        outputs = self._forward(**encoded, output_hidden_states=True, output_attentions=True)
        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        layer_hidden_arrays = self._collect_layer_hidden_arrays(outputs.hidden_states)
        layer_logit_lens = self._collect_layer_logit_lens(
            hidden_states=outputs.hidden_states,
            final_norm=final_norm,
            lm_head=lm_head,
            option_token_map=option_token_map,
        )
        attention_summary, attention_arrays = self._collect_attention_summary(
            attentions=outputs.attentions,
            input_ids=encoded["input_ids"][0].detach().cpu().tolist(),
            prompt_prefix=prompt_prefix,
            question_text=question_text,
            correct_option=correct_option,
            wrong_option=wrong_option_norm,
            option_token_map=option_token_map,
        )
        result = self._extract_base_result(
            sample_id=sample_id,
            scenario=scenario,
            question_text=question_text,
            prompt_prefix=prompt_prefix,
            ground_truth=ground_truth,
            wrong_option=wrong_option,
            answer_logits=answer_logits,
            last_logits=last_logits,
        )
        result["hidden_state_summary"] = self._hidden_state_summary(layer_hidden_arrays)
        result["layer_logit_lens"] = layer_logit_lens
        result["attention_summary"] = attention_summary
        result["_hidden_state_arrays"] = layer_hidden_arrays
        result["_attention_arrays"] = attention_arrays
        return result

    def analyze_prompt_selected_layers(
        self,
        *,
        prompt: str,
        sample_id: str,
        scenario: str,
        question_text: str,
        prompt_prefix: str,
        ground_truth: str,
        wrong_option: str,
        selected_layers: Iterable[int],
    ) -> Dict[str, Any]:
        """Analyze a prompt while collecting only selected decoder-layer hidden states."""
        self.load()
        assert self._model is not None
        assert self._device is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()
        outputs = self._forward(**encoded, output_hidden_states=True, output_attentions=False)
        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        hidden_arrays = self._collect_selected_layer_hidden_arrays(outputs.hidden_states, selected_layers)
        result = self._extract_base_result(
            sample_id=sample_id,
            scenario=scenario,
            question_text=question_text,
            prompt_prefix=prompt_prefix,
            ground_truth=ground_truth,
            wrong_option=wrong_option,
            answer_logits=answer_logits,
            last_logits=last_logits,
        )
        result["hidden_state_summary"] = self._hidden_state_summary(hidden_arrays)
        result["_hidden_state_arrays"] = hidden_arrays
        return result

    def patch_final_token_residual(
        self,
        *,
        prompt: str,
        patch_layer_index: int,
        patched_final_token,
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        _, decoder_layers, _, _ = self._resolve_model_parts()
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        patch_tensor = patched_final_token
        if isinstance(patch_tensor, np.ndarray):
            patch_tensor = self._torch.from_numpy(patch_tensor)
        patch_tensor = patch_tensor.to(device=self._device, dtype=self._resolve_dtype())

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                patched = hidden_states.clone()
                patched[:, -1, :] = patch_tensor.to(dtype=hidden_states.dtype)
                return (patched, *output[1:])
            patched = output.clone()
            patched[:, -1, :] = patch_tensor.to(dtype=patched.dtype)
            return patched

        handle = decoder_layers[int(patch_layer_index)].register_forward_hook(_hook)
        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            handle.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patch_layer_index": int(patch_layer_index),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def patch_final_token_residuals_multi(
        self,
        *,
        prompt: str,
        layer_patch_map: Dict[int, Any],
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        _, decoder_layers, _, _ = self._resolve_model_parts()
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        prepared: Dict[int, Any] = {}
        for layer_index, patch_tensor in layer_patch_map.items():
            tensor = patch_tensor
            if isinstance(tensor, np.ndarray):
                tensor = self._torch.from_numpy(tensor)
            prepared[int(layer_index)] = tensor.to(device=self._device, dtype=self._resolve_dtype())

        handles = []
        for layer_index, patch_tensor in prepared.items():
            layer = decoder_layers[int(layer_index)]

            def make_hook(tensor):
                def _hook(_module, _inputs, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        patched = hidden_states.clone()
                        patched[:, -1, :] = tensor.to(dtype=patched.dtype)
                        return (patched, *output[1:])
                    patched = output.clone()
                    patched[:, -1, :] = tensor.to(dtype=patched.dtype)
                    return patched

                return _hook

            handles.append(layer.register_forward_hook(make_hook(patch_tensor)))

        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            for handle in handles:
                handle.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patched_layers": sorted(int(layer_index) for layer_index in prepared),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def generate_with_final_token_residual_patches(
        self,
        *,
        prompt: str,
        layer_patch_map: Dict[int, Any],
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        do_sample: bool = False,
        stop_token_ids: Sequence[int] | None = None,
        use_cache: bool = False,
        patch_generation_steps: bool = True,
        patch_step_count: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        if use_cache:
            raise ValueError("use_cache=True is not supported in this correctness-first patched generation implementation.")
        _, decoder_layers, _, _ = self._resolve_model_parts()
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config.max_length),
        )
        current_inputs = {key: value.to(self._device) for key, value in encoded.items()}

        prepared: Dict[int, Any] = {}
        for layer_index, patch_tensor in layer_patch_map.items():
            tensor = patch_tensor
            if isinstance(tensor, np.ndarray):
                tensor = self._torch.from_numpy(tensor)
            prepared[int(layer_index)] = tensor.to(device=self._device, dtype=self._resolve_dtype())

        handles = []
        for layer_index, patch_tensor in prepared.items():
            layer = decoder_layers[int(layer_index)]

            def make_hook(tensor):
                def _hook(_module, _inputs, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        patched = hidden_states.clone()
                        patched[:, -1, :] = tensor.to(dtype=patched.dtype)
                        return (patched, *output[1:])
                    patched = output.clone()
                    patched[:, -1, :] = tensor.to(dtype=patched.dtype)
                    return patched

                return _hook

            handles.append(layer.register_forward_hook(make_hook(patch_tensor)))

        generated_token_ids: List[int] = []
        per_step_top_logits: List[List[Dict[str, Any]]] = []
        finish_reason = "max_new_tokens"
        eos_token_id = self._tokenizer.eos_token_id
        stop_ids = {int(token_id) for token_id in (stop_token_ids or [])}
        steps_remaining = int(patch_step_count) if patch_step_count is not None else None
        try:
            for _step in range(int(max_new_tokens)):
                outputs = self._forward(
                    **current_inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                last_logits = outputs.logits[:, -1, :].detach()
                if float(repetition_penalty) > 1.0:
                    penalty = float(repetition_penalty)
                    penalized_ids = set(current_inputs["input_ids"][0].detach().cpu().tolist())
                    for token_id in penalized_ids:
                        value = last_logits[0, int(token_id)]
                        if float(value.item()) < 0.0:
                            last_logits[0, int(token_id)] = value * penalty
                        else:
                            last_logits[0, int(token_id)] = value / penalty
                logits_for_sampling = last_logits
                if do_sample:
                    safe_temperature = max(float(temperature), 1e-5)
                    logits_for_sampling = logits_for_sampling / safe_temperature
                    probs = self._torch.softmax(logits_for_sampling, dim=-1)
                    next_token = self._torch.multinomial(probs, num_samples=1)
                else:
                    next_token = self._torch.argmax(logits_for_sampling, dim=-1, keepdim=True)
                next_token_id = int(next_token[0, 0].item())
                generated_token_ids.append(next_token_id)
                per_step_top_logits.append(self._top_token_logits(last_logits[0].detach().float().cpu(), self.config.top_k))

                current_inputs["input_ids"] = self._torch.cat([current_inputs["input_ids"], next_token], dim=1)
                if "attention_mask" in current_inputs:
                    next_mask = self._torch.ones_like(next_token, device=self._device)
                    current_inputs["attention_mask"] = self._torch.cat([current_inputs["attention_mask"], next_mask], dim=1)

                if eos_token_id is not None and next_token_id == int(eos_token_id):
                    finish_reason = "eos_token"
                    break
                if next_token_id in stop_ids:
                    finish_reason = "stop_token"
                    break
                if steps_remaining is not None:
                    steps_remaining -= 1
                    if steps_remaining <= 0 and handles:
                        for handle in handles:
                            handle.remove()
                        handles = []
                elif not patch_generation_steps and handles:
                    for handle in handles:
                        handle.remove()
                    handles = []
        finally:
            for handle in handles:
                handle.remove()

        generated_text = self._tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        return {
            "generated_text": generated_text,
            "generated_token_ids": list(generated_token_ids),
            "num_new_tokens": len(generated_token_ids),
            "finish_reason": finish_reason,
            "patched_layers": sorted(int(layer_index) for layer_index in prepared),
            "per_step_top_logits": per_step_top_logits,
        }

    def patch_residual_positions(
        self,
        *,
        prompt: str,
        patch_layer_index: int,
        patch_positions: Dict[int, Any],
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        _, decoder_layers, _, _ = self._resolve_model_parts()
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        prepared: Dict[int, Any] = {}
        for position, patch_tensor in patch_positions.items():
            if isinstance(patch_tensor, np.ndarray):
                patch_tensor = self._torch.from_numpy(patch_tensor)
            prepared[int(position)] = patch_tensor.to(device=self._device, dtype=self._resolve_dtype())

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                patched = hidden_states.clone()
                for position, tensor in prepared.items():
                    patched[:, int(position), :] = tensor.to(dtype=patched.dtype)
                return (patched, *output[1:])
            patched = output.clone()
            for position, tensor in prepared.items():
                patched[:, int(position), :] = tensor.to(dtype=patched.dtype)
            return patched

        handle = decoder_layers[int(patch_layer_index)].register_forward_hook(_hook)
        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            handle.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patch_layer_index": int(patch_layer_index),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def patch_residual_positions_multi_layer(
        self,
        *,
        prompt: str,
        layer_position_patch_map: Dict[int, Dict[int, Any]],
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        _, decoder_layers, _, _ = self._resolve_model_parts()
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        prepared: Dict[int, Dict[int, Any]] = {}
        for layer_index, patch_positions in layer_position_patch_map.items():
            prepared[int(layer_index)] = {}
            for position, patch_tensor in patch_positions.items():
                tensor = patch_tensor
                if isinstance(tensor, np.ndarray):
                    tensor = self._torch.from_numpy(tensor)
                prepared[int(layer_index)][int(position)] = tensor.to(device=self._device, dtype=self._resolve_dtype())

        handles = []
        for layer_index, patch_positions in prepared.items():
            layer = decoder_layers[int(layer_index)]

            def make_hook(position_map):
                def _hook(_module, _inputs, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        patched = hidden_states.clone()
                        for position, tensor in position_map.items():
                            patched[:, int(position), :] = tensor.to(dtype=patched.dtype)
                        return (patched, *output[1:])
                    patched = output.clone()
                    for position, tensor in position_map.items():
                        patched[:, int(position), :] = tensor.to(dtype=patched.dtype)
                    return patched

                return _hook

            handles.append(layer.register_forward_hook(make_hook(patch_positions)))

        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            for handle in handles:
                handle.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patched_layers": sorted(int(layer_index) for layer_index in prepared),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def patch_attention_head_output(
        self,
        *,
        prompt: str,
        patch_layer_index: int,
        head_index: int,
        patched_head_vector,
        patch_token_position: int,
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()
        attn = self._layer_self_attention(int(patch_layer_index))
        o_proj = getattr(attn, "o_proj", None)
        layout = self.attention_head_layout(int(patch_layer_index))
        num_heads = int(layout["num_heads"])
        head_dim = int(layout["head_dim"])
        if o_proj is None or num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"Layer {patch_layer_index} does not expose attention head metadata.")

        head_tensor = patched_head_vector
        if isinstance(head_tensor, np.ndarray):
            head_tensor = self._torch.from_numpy(head_tensor)
        head_tensor = head_tensor.to(device=self._device, dtype=self._resolve_dtype())
        start = int(head_index) * head_dim
        end = start + head_dim

        def _hook(_module, inputs):
            hidden = inputs[0].clone()
            hidden[:, int(patch_token_position), start:end] = head_tensor.to(dtype=hidden.dtype)
            return (hidden,)

        handle = o_proj.register_forward_pre_hook(_hook)
        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            handle.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patch_layer_index": int(patch_layer_index),
            "head_index": int(head_index),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def patch_attention_head_outputs_multi(
        self,
        *,
        prompt: str,
        patch_specs: Iterable[Dict[str, Any]],
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        hooks = []
        grouped_specs: Dict[int, List[Dict[str, Any]]] = {}
        for spec in patch_specs:
            grouped_specs.setdefault(int(spec["patch_layer_index"]), []).append(dict(spec))

        for layer_index, layer_specs in grouped_specs.items():
            attn = self._layer_self_attention(int(layer_index))
            o_proj = getattr(attn, "o_proj", None)
            layout = self.attention_head_layout(int(layer_index))
            num_heads = int(layout["num_heads"])
            head_dim = int(layout["head_dim"])
            if o_proj is None or num_heads <= 0 or head_dim <= 0:
                raise ValueError(f"Layer {layer_index} does not expose attention head metadata.")

            prepared_specs = []
            for spec in layer_specs:
                head_index = int(spec["head_index"])
                patch_token_position = int(spec.get("patch_token_position", -1))
                head_tensor = spec["patched_head_vector"]
                if isinstance(head_tensor, np.ndarray):
                    head_tensor = self._torch.from_numpy(head_tensor)
                prepared_specs.append(
                    {
                        "head_index": head_index,
                        "patch_token_position": patch_token_position,
                        "patched_head_vector": head_tensor.to(device=self._device, dtype=self._resolve_dtype()),
                    }
                )

            def make_hook(specs_for_layer):
                def _hook(_module, inputs):
                    hidden = inputs[0].clone()
                    for spec in specs_for_layer:
                        start = int(spec["head_index"]) * head_dim
                        end = start + head_dim
                        patch_pos = int(spec["patch_token_position"])
                        hidden[:, patch_pos, start:end] = spec["patched_head_vector"].to(dtype=hidden.dtype)
                    return (hidden,)

                return _hook

            hooks.append(o_proj.register_forward_pre_hook(make_hook(prepared_specs)))

        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            for hook in hooks:
                hook.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def patch_attention_output_multi(
        self,
        *,
        prompt: str,
        layer_patch_map: Dict[int, Any],
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        prepared: Dict[int, Any] = {}
        for layer_index, patch_tensor in layer_patch_map.items():
            tensor = patch_tensor
            if isinstance(tensor, np.ndarray):
                tensor = self._torch.from_numpy(tensor)
            prepared[int(layer_index)] = tensor.to(device=self._device, dtype=self._resolve_dtype())

        hooks = []
        for layer_index, patch_tensor in prepared.items():
            attn = self._layer_self_attention(int(layer_index))
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise ValueError(f"Layer {layer_index} missing o_proj; cannot patch attention output.")

            def make_hook(tensor):
                def _hook(_module, inputs):
                    hidden = inputs[0].clone()
                    hidden[:, -1, :] = tensor.to(dtype=hidden.dtype)
                    return (hidden,)

                return _hook

            hooks.append(o_proj.register_forward_pre_hook(make_hook(patch_tensor)))

        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            for hook in hooks:
                hook.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patched_layers": sorted(int(layer_index) for layer_index in prepared),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }

    def patch_mlp_output_multi(
        self,
        *,
        prompt: str,
        layer_patch_map: Dict[int, Any],
        ground_truth: str,
        wrong_option: str,
    ) -> Dict[str, Any]:
        self.load()
        assert self._model is not None
        encoded = self._encode_prompt(prompt)
        option_token_map = self._option_token_id_map()

        prepared: Dict[int, Any] = {}
        for layer_index, patch_tensor in layer_patch_map.items():
            tensor = patch_tensor
            if isinstance(tensor, np.ndarray):
                tensor = self._torch.from_numpy(tensor)
            prepared[int(layer_index)] = tensor.to(device=self._device, dtype=self._resolve_dtype())

        hooks = []
        for layer_index, patch_tensor in prepared.items():
            mlp = self._layer_mlp(int(layer_index))

            def make_hook(tensor):
                def _hook(_module, _inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        patched = hidden.clone()
                        patched[:, -1, :] = tensor.to(dtype=patched.dtype)
                        return (patched, *output[1:])
                    patched = output.clone()
                    patched[:, -1, :] = tensor.to(dtype=patched.dtype)
                    return patched

                return _hook

            hooks.append(mlp.register_forward_hook(make_hook(patch_tensor)))

        try:
            outputs = self._forward(**encoded, output_hidden_states=False)
        finally:
            for hook in hooks:
                hook.remove()

        last_logits = outputs.logits[0, -1, :].detach().float().cpu()
        answer_logits = self._answer_logits_from_last_logits(last_logits, option_token_map)
        correct_option = str(ground_truth or "").strip().upper()
        wrong_option_norm = str(wrong_option or "").strip().upper()
        return {
            "patched_layers": sorted(int(layer_index) for layer_index in prepared),
            "predicted_answer": max(answer_logits.items(), key=lambda item: item[1])[0],
            "answer_logits": answer_logits,
            "correct_option_logit": float(answer_logits.get(correct_option, float("nan"))),
            "wrong_option_logit": float(answer_logits.get(wrong_option_norm, float("nan"))),
            "correct_wrong_margin": float(answer_logits.get(correct_option, 0.0) - answer_logits.get(wrong_option_norm, 0.0)),
            "top_token_logits": self._top_token_logits(last_logits, self.config.top_k),
        }
