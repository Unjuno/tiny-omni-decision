"""Local raw-text runtime for the fixed BANKING77 MiniLM-L6 classifier."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .classification_json import classification_json

CLASS_COUNT = 77
FEATURE_SIZE = 384
MAX_SEQUENCE_LENGTH = 128
CLASS_NAMES = (
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
)
EXPECTED_ENCODER_PARAMETERS = 22_713_216
EXPECTED_MODEL_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
EXPECTED_CHECKPOINT_SHA256 = "5846768a9b598eb4821e63b630086fb01bc866dd006261124c88dd6a01199577"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_ids() -> list[str]:
    """Return label IDs in the order pinned by the publisher's dataset card."""
    return [f"banking77:{name}" for name in CLASS_NAMES]


class Banking77MiniLML6Decision:
    """Frozen MiniLM-L6 plus a fitted 77-way linear readout.

    This is a closed-taxonomy banking intent classifier. Supplied options must
    enumerate every class exactly once; it does not score arbitrary text options.
    """

    def __init__(self, encoder: nn.Module, tokenizer: Any, readout: nn.Module,
                 train_mean: torch.Tensor, train_std: torch.Tensor) -> None:
        self.encoder = encoder.cpu().eval()
        self.tokenizer = tokenizer
        self.readout = readout.cpu().eval()
        self.train_mean = train_mean.cpu().float()
        self.train_std = train_std.cpu().float()

    @classmethod
    def from_local_files(
        cls, model_dir: Path, checkpoint_path: Path
    ) -> Banking77MiniLML6Decision:
        model_dir = Path(model_dir)
        checkpoint_path = Path(checkpoint_path)
        weight_path = model_dir / "model.safetensors"
        if _sha256(weight_path) != EXPECTED_MODEL_SHA256:
            raise ValueError("MiniLM-L6 encoder weight hash does not match pinned revision")
        if _sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("BANKING77 readout hash does not match selected checkpoint")

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).cpu().eval()
        encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
        if encoder_parameters != EXPECTED_ENCODER_PARAMETERS:
            raise ValueError(f"Unexpected encoder parameter count: {encoder_parameters}")
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

        # The checkpoint is a locally generated tensor dictionary, not a model artifact.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("class_count") != CLASS_COUNT or checkpoint.get("best_epoch") != 5:
            raise ValueError("Unexpected selected readout metadata")
        readout = nn.Linear(FEATURE_SIZE, CLASS_COUNT)
        readout.load_state_dict(checkpoint["state_dict"], strict=True)
        train_mean = checkpoint["train_mean"].float()
        train_std = checkpoint["train_std"].float()
        if train_mean.shape != (FEATURE_SIZE,) or train_std.shape != (FEATURE_SIZE,):
            raise ValueError("Readout feature standardization has an unexpected shape")
        if not torch.isfinite(train_mean).all() or not torch.isfinite(train_std).all():
            raise ValueError("Readout feature statistics are non-finite")
        if (train_std <= 0).any():
            raise ValueError("Readout feature standard deviations must be positive")
        return cls(encoder, tokenizer, readout, train_mean, train_std)

    @torch.inference_mode()
    def probabilities(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        features = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        standardized = (features.float() - self.train_mean) / self.train_std
        logits = self.readout(standardized)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        if probabilities.shape != (CLASS_COUNT,) or not torch.isfinite(probabilities).all():
            raise RuntimeError("Classifier produced invalid probabilities")
        return probabilities.tolist()

    def decide(self, text: str, options: list[dict[str, str]]) -> dict[str, Any]:
        return classification_json(self.probabilities(text), class_ids(), options)
