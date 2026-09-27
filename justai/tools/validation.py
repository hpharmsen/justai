import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

MAX_ERROR_LINES = 20

STRUCTURED_FEEDBACK = (
    'The previous response failed validation.\n\nValidation errors:\n{errors}\n\n'
    'Return a corrected response that satisfies the required schema.\n'
    'Do not explain the correction.'
)
TOOL_FEEDBACK = (
    'Tool call `{name}` could not be executed because its arguments failed validation.\n\n'
    'Validation errors:\n{errors}\n\nCorrect the tool arguments and call the tool again.'
)


class ValidationRetryError(ValueError):
    """Validation still failed after the allowed repair attempts."""

    def __init__(self, message: str, attempts: int, last_error: ValidationError, raw: Any):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.raw = raw


def format_validation_errors(err: ValidationError) -> str:
    """One line per error as '- path: msg', without input values, capped at MAX_ERROR_LINES."""
    errors = err.errors(include_url=False, include_input=False, include_context=False)
    lines = [f'- {".".join(str(p) for p in e["loc"]) or "(root)"}: {e["msg"]}' for e in errors[:MAX_ERROR_LINES]]
    if len(errors) > MAX_ERROR_LINES:
        lines.append(f'- ... and {len(errors) - MAX_ERROR_LINES} more')
    return '\n'.join(lines)


class RepairBudget:
    """Validates raw output and counts consecutive failures against a retry budget."""

    def __init__(
        self, validate: Callable[[Any], Any], max_retries: int, label: str, template: str = STRUCTURED_FEEDBACK
    ):
        assert max_retries >= 0, 'max_retries must be non-negative'
        self.validate, self.max_retries, self.label, self.template = validate, max_retries, label, template
        self.failures = 0

    def check(self, raw: Any) -> tuple[Any, str | None]:
        """Return (value, None) when valid, (None, feedback) when a retry is left, else raise."""
        try:
            value = self.validate(raw)
        except ValidationError as e:
            self.failures += 1
            if self.failures > self.max_retries:
                raise ValidationRetryError(
                    f'{self.label} failed validation after {self.failures} attempts', self.failures, e, raw
                ) from e
            logger.debug(f'{self.label} validation failed, retry {self.failures}/{self.max_retries}')
            return None, self.template.format(errors=format_validation_errors(e), name=self.label)
        self.failures = 0
        return value, None
