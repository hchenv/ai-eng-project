import re
from typing import Any, Dict

from guardrails import Guard, OnFailAction
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from langsmith import traceable


PRODUCT_ID = re.compile(r"\bB0[A-Z0-9]{8}\b")
REPLACEMENT = "[REDACTED..]"


@register_validator(name="redact_product_id", data_type="string")
class RedactProductID(Validator):
    def _validate(
        self, value: Any, metadata: Dict[str, Any] | None = None
    ) -> ValidationResult:
        matches = PRODUCT_ID.findall(value)

        if matches:
            return FailResult(
                error_message=f"found {len(matches)} product ids: {matches}",
                fix_value=PRODUCT_ID.sub(REPLACEMENT, value),
            )

        return PassResult()


@traceable(name="redact_product_id")
def redact_product_id(state) -> dict:
    guard = Guard().use(RedactProductID(on_fail=OnFailAction.FIX))
    outcome = guard.validate(state.answer)

    return {"answer": outcome.validated_output}
