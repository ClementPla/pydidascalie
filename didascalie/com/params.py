"""Arguments a function exposes in Didascalie, declared in its decorator.

    @register_seg("unet", threshold=Float(0.5, min=0, max=1, label="Threshold"))
    def unet(image, threshold):
        ...

Didascalie builds a form from them. Values are checked here as well: this is
the side that calls the function.
"""
import math
from typing import Any, Optional, Sequence


class Param:
    """Base of the argument types. `label` defaults to the parameter's name."""

    type = ""

    def __init__(self, default: Any, label: Optional[str] = None, hint: Optional[str] = None):
        self.label = label
        self.hint = hint
        self.default = self.check(default)

    def check(self, value: Any) -> Any:
        """`value` as the function receives it. Raises ValueError if invalid."""
        raise NotImplementedError

    def _default_hint(self) -> str:
        return ""

    def _fields(self) -> dict:
        return {}

    def describe(self, name: str) -> dict:
        return {
            "name": name,
            "type": self.type,
            "label": self.label or name.replace("_", " ").capitalize(),
            "hint": self.hint or self._default_hint(),
            "default": self.default,
            **self._fields(),
        }


class _Number(Param):
    noun = ""

    def __init__(self, default, *, min=None, max=None, step=None, label=None, hint=None):
        if min is not None and max is not None and min > max:
            raise ValueError(f"min ({min}) is greater than max ({max})")
        if step is not None and step <= 0:
            raise ValueError("step must be positive")
        self.min = min
        self.max = max
        self.step = step
        super().__init__(default, label, hint)

    def _convert(self, value):
        raise NotImplementedError

    def check(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"expected {self.noun}, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"expected a finite number, got {value!r}")
        value = self._convert(value)
        if self.min is not None and value < self.min:
            raise ValueError(f"{value} is below the minimum, {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"{value} is above the maximum, {self.max}")
        return value

    def _default_hint(self) -> str:
        kind = self.noun.split(" ", 1)[1].capitalize()
        if self.min is not None and self.max is not None:
            return f"{kind} from {self.min} to {self.max}"
        if self.min is not None:
            return f"{kind}, at least {self.min}"
        if self.max is not None:
            return f"{kind}, at most {self.max}"
        return kind

    def _fields(self) -> dict:
        bounds = {"min": self.min, "max": self.max, "step": self.step}
        return {k: float(v) for k, v in bounds.items() if v is not None}


class Int(_Number):
    """A whole number."""

    type = "int"
    noun = "an integer"

    def _convert(self, value):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"expected {self.noun}, got {value!r}")
        return int(value)


class Float(_Number):
    """A decimal number."""

    type = "float"
    noun = "a number"

    def _convert(self, value):
        return float(value)


class Boolean(Param):
    """An on/off switch."""

    type = "bool"

    def __init__(self, default: bool = False, *, label=None, hint=None):
        super().__init__(default, label, hint)

    def check(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"expected true or false, got {value!r}")
        return value


class Enum(Param):
    """One choice among strings. `default` is the first choice unless given."""

    type = "enum"

    def __init__(self, choices: Sequence[str], *, default=None, label=None, hint=None):
        if isinstance(choices, str):
            raise TypeError("choices must be a list of strings, not one string")
        self.choices = list(choices)
        if not self.choices or not all(isinstance(c, str) for c in self.choices):
            raise ValueError("choices must be a non-empty list of strings")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("choices must be unique")
        super().__init__(self.choices[0] if default is None else default, label, hint)

    def check(self, value):
        if value not in self.choices:
            raise ValueError(f"{value!r} is not one of {self.choices}")
        return value

    def _fields(self) -> dict:
        return {"choices": self.choices}


def resolve(params: dict, sent: Optional[dict]) -> dict:
    """The keyword arguments for a call: what was sent, else the defaults."""
    sent = sent or {}
    unknown = sorted(set(sent) - set(params))
    if unknown:
        raise ValueError(f"Unknown arguments: {', '.join(unknown)}")
    values = {}
    for name, param in params.items():
        if name not in sent:
            values[name] = param.default
            continue
        try:
            values[name] = param.check(sent[name])
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None
    return values
