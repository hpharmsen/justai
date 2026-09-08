# Plan: unified `effort` parameter for justai

## Enhancement Summary

**Deepened on:** 2026-07-14
**Sections enhanced:** 8 (all)
**Agents used:** coherence, feasibility, scope-guardian, correctness, kieran-python, testing, code-simplicity, api-contract, reliability, framework-docs, best-practices, splat-enumeration (Explore)

### Decisions locked in (2026-07-14 review)

| # | Decision |
|---|---|
| 1 | **Bundle** Anthropic hygiene fixes (temperature stripping, refusal handling, RefusalException) into this PR — single release, single changelog entry |
| 2 | **Option B** for eager validation: `BaseModel._VALIDATORS: dict[str, Callable]` registry consulted by `Model.__setattr__`. Extensible for future validated params. No typo detection (out of scope). |
| 3 | **Clean break on cache** — bump cache namespace to invalidate all entries on upgrade; do not attempt to preserve pre-feature entries |
| 4 | **Cap OpenAI vocabulary at `xhigh`** — map `max` → `xhigh` with `EffortDownmapWarning` for GPT-5.6 and older reasoning models. `max` is not in the SDK's `ReasoningEffort` `Literal`; safest to type-check |
| 5 | **Defer `Model.supports_effort` capability flag.** `EffortDownmapWarning` provides the same signal; users can promote to exception via `filterwarnings('error', category=EffortDownmapWarning)`. If demand emerges later, prefer `effort_levels: tuple[str, ...]` (accurate per-model) over a bool |

### Blocking issues surfaced (must resolve before implementation)

1. **Eager `ValueError` on `model.effort = 'banana'` does not fire as written.** `Model.__setattr__` (model.py:66-72) writes directly to `self.model.model_params[name]` and never calls `BaseModel.set()`. Test #8 in the current plan will fail. Validation must live in `Model.__setattr__` and in each provider `__init__` — not only in `BaseModel.set()`.
2. **Ordering trap:** `params.setdefault('effort', None)` cannot run *before* `BaseModel.__init__` (self.model_params doesn't exist yet). Do it *after* `super().__init__(...)` — or hoist to `BaseModel.__init__` and delete the eight copies.
3. **Splat-before-strip breaks Google and Anthropic non-structured paths.** Adding `effort` to `model_params` while `**self.model_params` is still splatted into `GenerateContentConfig(...)` and `client.messages.create(...)` will raise `ValidationError` / 400 on *every* call — even when the user never set effort. Land the splat→`api_params` swap in a **prior commit**, then land the seed. Order-of-operations, not just correctness.
4. **Refusal handling IndexErrors before reaching the custom exception.** `anthropic_models.py:154` does `message.content[0].text` unconditionally; on `stop_reason=='refusal'` content is empty and this raises `IndexError`. The refusal check must be inserted upstream in `chat()` immediately after the API returns, not after content extraction.
5. **Beta header `effort-2025-11-24` is no longer required** per current Anthropic docs (verified July 2026). It was needed at Opus 4.5 GA, has since been dropped. Remove the beta-header wiring unless targeting old SDKs.
6. **OpenAI SDK type only lists `none/minimal/low/medium/high/xhigh`** for `ReasoningEffort` — `max` is **not** in the Python `Literal`. Sending `max` bypasses type checking and may 400 on some deployments. Verify per-model.
7. **xAI `__init__` calls `BaseModel.__init__` directly** (xai_models.py:15), bypassing `OpenAIResponsesModel.__init__`. Any effort seeding done in the parent will not run for xAI.
8. **Grok-3-mini is deprecated (retired 2026-08-15).** Current xAI reasoning models: `grok-4.5`, `grok-4.3`, `grok-4.20-multi-agent`. Update the translation table.

### Key non-blocking improvements

- **Cache-key clean break (decision #3):** Bump the cache namespace so all pre-upgrade entries invalidate cleanly. Simpler than trying to preserve them via `ignore_params=['effort']` when None. Document in release notes: "upgrading to vX.Y.Z invalidates the local response cache; first call per prompt hits the API."
- **Two-layer API pattern (PydanticAI-style):** Unified `effort=` for portability + per-provider override kwargs (`anthropic_thinking_budget=`, `openai_reasoning_effort=`) for precision. Deferred to a future PR; not blocking.
- **Wrapper methods** `_responses_create(**kwargs)` / `_responses_parse(**kwargs)` on `OpenAIResponsesModel` to centralize the reasoning injection. Seven call sites (six main + one image-gen to skip) drift; one wrapper doesn't.
- **Custom `EffortDownmapWarning(UserWarning)` subclass** so users can filter with `warnings.filterwarnings('ignore', category=EffortDownmapWarning)` without silencing all UserWarnings. Also serves as the introspection signal for provider-agnostic code (promote to error with `filterwarnings('error', ...)`).
- **Auto-raise `max_tokens` on effort ≥ high** for Anthropic (reasoning tokens count against output budget; default 800 exhausted mid-thought).
- **Auto-raise timeout on effort ≥ xhigh** (default 120s insufficient; three 120s retries burn 6 minutes for nothing).
- **Delete downmapping in favor of raising `ValueError`** — considered but rejected. Downmap+warn matches PydanticAI (the cleanest comparable library) and preserves the "one setting works everywhere" contract. Keep downmap, but tighten warn-once semantics.
- **Bundle Anthropic hygiene into this PR (decision #1).** Temperature/top_p/top_k stripping, `thinking: disabled` removal, and RefusalException ship together with the effort feature. Single version bump, single changelog entry. Reviewer noise from bundling is acceptable given the small blast radius.

### Actual call-site inventory (from grep of the codebase)

Nine `**self.model_params` splat sites; the plan enumerates six OpenAI Responses sites but was missing three:

| File | Line | API call | Path |
|---|---|---|---|
| openai_completions.py | 216, 234, 414 | `chat.completions.parse/create` | sync + async stream |
| openai_responses.py | 81, 91, 95, 99 | `responses.create/parse` (sync prompt) | 4 branches |
| openai_responses.py | 225 | `responses.create` (`prompt_async`) | streaming |
| openai_responses.py | 379 | `responses.create` (stateless `stream`) | agent path |
| openai_responses.py | 709 | `responses.create` (image-gen path) | **skip effort here** |
| anthropic_models.py | 235 | `messages.parse` (structured) | sync |
| anthropic_models.py | 312 | `messages.create(stream=True)` (chat_async) | streaming |
| anthropic_models.py | 347 | `messages.create` (main completion) | sync |
| anthropic_models.py | 472 | `async_client.messages.create(stream=True)` (stateless `stream`) | streaming |
| google_models.py | 72 (via local `params` copy), 184 | `generate_content` / `aio.generate_content_stream` | sync + async |

Note: openai_completions.py handles o1/o3/o-series and inherited providers (DeepSeek, Perplexity, OpenRouter). The plan only listed Responses call sites — o-series handling in openai_completions.py must be added or explicitly deferred.

---

## Goal

Add a single cross-provider `effort` setting to justai that controls reasoning depth / token spend, translated to each provider's native API. Usage:

```python
model = Model('claude-fable-5', effort='low')
# or
model = Model('gpt-5.6-terra')
model.effort = 'xhigh'
```

### Research Insights

**Comparable library survey (from best-practices research):**
- **LiteLLM** raises `UnsupportedParamsError` by default; opt-in silence via `litellm.drop_params=True`. Users hate the silent-drop mode most (issues #19700, #6516).
- **LangChain** exposes `reasoning_effort` on `BaseChatOpenAI`, `thinking` on `ChatAnthropic`, `thinking_budget` on Google — no unified surface. Portable code is impossible; pain surfaced in issue #34933.
- **PydanticAI** is the cleanest design: two-layer — unified `ModelSettings.thinking` for portability + provider-specific overrides (`openai_reasoning_effort`, `anthropic_thinking`) for precision. Downmaps to closest supported with warning. Recommend this shape.
- **AISuite** deliberately omits reasoning from its unified surface — passes through as provider kwargs only.

**Recommended shape for justai:**
- Keep unified `effort` (this plan) as the portability layer.
- Later: allow per-provider override via `model_params` passthrough (`Model('claude-...', anthropic_output_config={...})`). Not blocking for this PR.
- **Do not silently drop.** Downmap + warn (as planned) is the middle path.

---

## Public interface

- Accepted values: `'low' | 'medium' | 'high' | 'xhigh' | 'max'` or `None` (default: `None` = don't send anything, provider default applies).
- Additionally accept `'none'` **only** as a pass-through for OpenAI GPT-5.6 models (turns reasoning off). On all other providers `'none'` raises `ValueError`.
- Any other value raises `ValueError` immediately when set, with the list of valid values in the message.
- Downmapping rule: when a level doesn't exist on the target model, map **down** to the closest supported level and emit a `EffortDownmapWarning` (subclass of `UserWarning`) stating the original value, the mapped value and the model name. Exception: `xhigh` on Claude 4.6 models maps **up** to `max` (closest existing neighbor), also with a warning.

### Research Insights

**Contradictions and gaps to fix in this section:**

- **Downmap direction rule is internally contradictory.** The rule says "map down to the closest supported level," but the exception (`xhigh` on Opus 4.6 → `max`, up) violates the direction without a principled tiebreaker. Opus 4.6 supports `{low, medium, high, max}`; `xhigh` is equidistant to `high` (down) and `max` (up). Two fixes are equally acceptable:
  - **Option A (codify the algorithm):** "Prefer down; on tie prefer up if the higher level exists (rationale: user asked for more, not less)."
  - **Option B (delete the algorithm, hard-code tables):** Per-model `EFFORT_MAP: dict[str, dict[str, str]]` with explicit source→target for every model. No inference, no ambiguity. **Recommended** — matches the existing per-provider-regex-constant pattern the plan endorses elsewhere.

- **Mid-request mutation semantics are undefined.** If a user does `model.effort = 'max'` mid-stream (during an active `chat_async`), does the in-flight request see the new value? Rule: **effort is captured at request initiation.** Document this. Implementation: `resolve_effort()` runs once per API call, at the top of the method that builds `api_params`.

- **`None` vs `'none'` string ambiguity is a footgun** (all reviewers flagged). `effort=None` = don't send; `effort='none'` = pass-through disable on GPT-5.6 only. Users writing `effort='none'` targeting GPT-5.6 and later swapping to Claude will hit `ValueError` at runtime. Two mitigations:
  - Document loudly in the README's Effort section, with a side-by-side example.
  - Consider aliasing `'off'` → `'none'` for GPT-5.6 as a friendlier name (deferred; not blocking).

- **Typo silence:** `model.efort = 'low'` (typo) currently silently sets an unrelated attribute (`Model.__setattr__` at model.py:66-72 falls through to `super().__setattr__` when the key isn't in `model_params`). Add `Model.__setattr__` guard: for a small set of `KNOWN_MODEL_PARAMS = {'effort', 'temperature', 'max_tokens', ...}` typos raise. Reduces this feature's footgun surface.

- **Warning categorization:** Use a custom subclass `class EffortDownmapWarning(UserWarning): pass` — users can then `warnings.filterwarnings('ignore', category=EffortDownmapWarning)` without silencing all UserWarnings (real concern for Django/pytest strict-warning consumers).

- **String normalization:** Decide: is `effort='HIGH'` accepted? Recommend **no** — reject case variants at assignment with `ValueError` (matches type-hint norms; avoids ambiguity).

- **`ValueError` vs custom exception:** Current justai raises only `NotImplementedError`/`AttributeError`/its custom `*Exception` hierarchy for bad inputs. `ValueError` is Pythonic but sets a new precedent. Recommendation: raise `ValueError` (standard Python for enum-value violations), but also mention in docs so users writing broad `except BadRequestException:` know they need `ValueError` too.

---

## Translation table

| justai effort | Anthropic Fable 5 / Mythos 5 / Opus 4.8 / 4.7 / Sonnet 5 | Anthropic Opus 4.6 / Sonnet 4.6 | Anthropic Opus 4.5 | OpenAI gpt-5.6-* | Older OpenAI reasoning models (gpt-5.x < 5.6, o-series) | Gemini 3 | xAI grok-4.5 / grok-4.3 / grok-4.20-multi-agent | DeepSeek / Perplexity / Reve / GGUF |
|---|---|---|---|---|---|---|---|---|
| `low` | low | low | low | low | low | `thinking_level: LOW` | low | ignore + warning |
| `medium` | medium | medium | medium | medium | medium | MEDIUM | medium → if unsupported: low ⚠️ | ignore + warning |
| `high` | high | high | high | high | high | HIGH | high | ignore + warning |
| `xhigh` | xhigh | **max ⚠️** | high ⚠️ | xhigh | xhigh if supported, else high ⚠️ | HIGH ⚠️ | high ⚠️ | ignore + warning |
| `max` | max | max | high ⚠️ | **xhigh ⚠️** (SDK Literal caps at xhigh) | high ⚠️ | HIGH ⚠️ | high ⚠️ | ignore + warning |
| `'none'` | ValueError | ValueError | ValueError | none | ValueError | ValueError | ValueError | ValueError |

⚠️ = emit `EffortDownmapWarning`.

Notes:
- Anthropic models **without** effort support (Sonnet 4.5, Haiku 4.5, everything ≤ 4.x other than listed): ignore the setting and warn — do not send `output_config.effort`, it would 400.
- **Beta header `effort-2025-11-24` for Opus 4.5 is NO LONGER REQUIRED** as of current Anthropic docs. Was needed at Opus 4.5 GA; has since been made unconditional. Drop from the plan unless targeting old SDK versions.
- Keep model detection in per-provider constants (compiled regexes, same style as the existing `NO_TEMPERATURE_MODELS` in `anthropic_models.py`) so new models are a one-line change.

### Research Insights

**Verified wire-format corrections (framework-docs research, current SDK sources):**

- **Anthropic `output_config={"effort": ...}` — CONFIRMED** as the correct kwarg. Documented in `anthropic-sdk-python/src/anthropic/types/effort_capability.py`; explicitly noted in `ThinkingConfigAdaptiveParam` source: "effort belongs in `output_config`, not in thinking configuration."
- **Anthropic beta header — NO LONGER REQUIRED.** Official docs page states: "The effort parameter is available on all supported models with no beta header required." **Remove `effort-2025-11-24` handling.**
- **Fable 5 / Mythos 5 `thinking={'type':'disabled'}` returns 400 — CONFIRMED.** Same applies to Opus 4.8, Opus 4.7, Sonnet 5. Sonnet 5 alone accepts `thinking: {type: "disabled"}`.
- **OpenAI Responses `reasoning={"effort": ...}` — CONFIRMED** on `client.responses.create/parse`. But the SDK's `ReasoningEffort` type alias is `Literal["none", "minimal", "low", "medium", "high", "xhigh"]` — `max` is NOT in the Python type. **Decision #4: cap our OpenAI vocabulary at `xhigh`.** justai's `effort='max'` maps to `xhigh` for GPT-5.6 with `EffortDownmapWarning`. Safest, type-checked, avoids surprise 400s on non-pro-mode deployments. Note: justai does not surface OpenAI's `minimal` level (users who need it can pass it via a future per-provider override kwarg).
- **OpenAI `none` on o1/o3:** NOT accepted — only `low/medium/high`. Plan is correct.
- **Google `ThinkingConfig(thinking_level=ThinkingLevel.HIGH)` — CONFIRMED.** `ThinkingLevel` is a `CaseInSensitiveEnum` (so string `"HIGH"` works), values `MINIMAL/LOW/MEDIUM/HIGH`. `thinking_level` replaces Gemini 2.5's `thinking_budget: int`. Both fields still exist on `ThinkingConfig` for compat.
- **xAI: dual path.** Responses-API-compatible endpoint (`client.responses.create` against `api.x.ai/v1`) uses nested `reasoning={"effort": ...}` (plan's approach works). Legacy Chat Completions endpoint uses top-level `reasoning_effort=`. Current implementation uses OpenAI client against xAI base URL — verify which endpoint is hit.
- **xAI current models (mid-2026):** `grok-4.5`, `grok-4.3`, `grok-4.20-multi-agent`. **`grok-3-mini` deprecated 2026-05-15, retired 2026-08-15.** Update the translation table header and remove grok-3-mini specific handling from the xAI mapping.
- **OpenRouter unified `reasoning={"effort": ...}` — CONFIRMED**, plus `reasoning={"max_tokens": N}` and `reasoning={"exclude": true}`. OpenRouter server-side snaps unsupported levels to nearest — meaning our downmap logic double-maps if we do it too. Recommend: pass through raw, let OpenRouter map. Skip our own downmap for openrouter/*.

**Tiebreaker for xhigh on Opus 4.6 (correctness reviewer):** `high` and `max` are equidistant from `xhigh`. Plan chooses `max` ("closest neighbor") — codify this as "on tie, prefer up if the higher level exists (user asked for more, not less)" or (better) hard-code every mapping.

**Older OpenAI reasoning models column is a conditional cell** — "xhigh if supported, else high" depends on runtime model detection within the column. Split into two sub-columns or enumerate per-model.

---

## Wire formats

- **Anthropic** (`anthropic_models.py`): `output_config={'effort': <level>}` in `messages.create/parse/stream`. NOTE: `output_config` is already built for structured output (json_schema format) around line 220 — **merge** the effort key into the existing dict, never overwrite it. Also merge into the async/streaming call paths (`chat_async`, the stream() method) — every place that builds `api_params`.
- **OpenAI Responses** (`openai_responses.py`): `reasoning={'effort': <level>}` on `client.responses.create(...)` and `client.responses.parse(...)`. There are multiple call sites (sync branches around lines 81–99, streaming around line 225, stateless `stream` at line 379). Build one shared dict, e.g. `extra = {'reasoning': {'effort': e}} if e else {}`, and splat `**extra` into every call site.
- **Google** (`google_models.py`): set `thinking_config=genai.types.ThinkingConfig(thinking_level=<LEVEL>)` on the `GenerateContentConfig` objects (both the sync path ~line 79 and async path ~line 109/181). Only for Gemini 3.x models; Gemini 2.5 models: translate to `thinking_budget` is out of scope — ignore + warning there.
- **xAI** (`xai_models.py`): inherits from `OpenAIResponsesModel`, so it gets the `reasoning={'effort': ...}` plumbing for free — **but only if xAI's `__init__` reaches the parent's seeding logic** (see plumbing section: xAI currently calls `BaseModel.__init__` directly, bypassing `OpenAIResponsesModel.__init__`). Override only the *validation/mapping* hook because xAI supports a smaller set.
- **DeepSeek / Perplexity / Reve / OpenRouter-passthrough / GGUF**: no support → ignore with a single `EffortDownmapWarning` per Model instance ("effort is not supported by <provider>, ignoring"). Exception: for **OpenRouter** (`openrouter_models.py`), pass OpenRouter's unified `reasoning: {'effort': ...}` field — this is confirmed working per OpenRouter docs. Skip our downmap for openrouter/* (OpenRouter maps server-side).

### Research Insights

**Actual call-site inventory (splat enumeration Explore agent):**

```
openai_completions.py    line 216  chat.completions.parse  (structured)
openai_completions.py    line 234  chat.completions.create (regular sync + optional stream)
openai_completions.py    line 414  chat.completions.create (async stream, stateless)
openai_responses.py      line 81   responses.create        (JSON + response_format branch)
openai_responses.py      line 91   responses.parse         (Pydantic branch)
openai_responses.py      line 95   responses.create        (return_json branch)
openai_responses.py      line 99   responses.create        (default branch)
openai_responses.py      line 225  responses.create        (prompt_async, streaming)
openai_responses.py      line 379  responses.create        (stateless stream, agent path)
openai_responses.py      line 709  responses.create        (image generation) ← SKIP effort here
anthropic_models.py      line 235  messages.parse          (structured output)
anthropic_models.py      line 312  messages.create(stream=True)  (chat_async via completion)
anthropic_models.py      line 347  messages.create         (main completion, tool loop)
anthropic_models.py      line 472  async_client.messages.create(stream=True)  (stateless stream)
google_models.py         line 72   generate_content        (sync, via local `params` copy)
google_models.py         line 184  aio.generate_content_stream  (async stream)
```

**Notes:**
- Plan enumerates 6 OpenAI Responses sites (81, 91, 95, 99, 225, 379); actual code has 7 including image generation at line 709. **Do NOT** inject reasoning into 709 (image path). Explicitly skip.
- Plan misses `openai_completions.py` entirely (3 sites). o-series and inherited providers (DeepSeek, Perplexity, OpenRouter that extends `OpenAICompletionsModel`) go through this file. Decide: add o-series support in this PR, or explicitly defer with a `TODO` in the plan.
- **Anti-drift wrapper:** New OpenAI call sites are easy to add without effort injection. Introduce `self._responses_create(**kwargs)` / `self._responses_parse(**kwargs)` on `OpenAIResponsesModel` that always merges `reasoning={'effort': ...}` from `resolve_effort()`. Replace all six/seven direct calls. New call sites are correct by default; also gives one place to enforce the timeout policy.

**Defensive merge:** `output_config.setdefault('effort', level)` and warn if the key was already present (defends against user-supplied `model_params['output_config']` collision, and against future SDK reshapes).

**Google splat conflict is BLOCKING:** `google_models.py:79-80,184` splat `**self.model_params` into `GenerateContentConfig(...)`. This is a pydantic dataclass that rejects unknown kwargs. If we seed `params['effort'] = None` before swapping the splat to `**api_model_params()`, Google will `ValidationError` on *every* call. **Ordering: swap splats first (in a separate commit), then seed effort.**

**Image-gen path (`openai_responses.py:709`) uses `client = OpenAI()` (fresh client, not `self.client`)** — bypasses our timeout config too. Pre-existing issue, flagged for a follow-up but not for this PR.

**OpenAI multi-turn tool loop (`openai_responses.py:77 for run in range(3)`):** effort must be sent on every iteration, not just the first. A shared `extra` dict built once outside the loop suffices — verify the test covers iterations 2 and 3.

---

## Architecture / plumbing

The codebase splats `**self.model_params` straight into API calls, and `Model.__setattr__` forwards attribute writes into `model_params` when the key already exists there. Use that mechanism:

1. In `BaseModel.__init__` (**not** each provider — hoist to base to kill 8 copies of the same line), do `self.model_params.setdefault('effort', None)` **after** the params dict has been assigned. Rationale: matches "less code is better code" from CLAUDE.md, avoids the ordering trap of trying to setdefault before `super().__init__(...)` has created `self.model_params`.
2. Because `model_params` is splatted raw into API calls, every provider must **use `api_params` (or an inline dict comprehension) instead of `self.model_params`** for splats. Add a property on `basemodel.py`:

```python
class BaseModel(ABC):
    # subclasses extend:
    _NON_API_PARAMS: frozenset[str] = frozenset({'timeout', 'async', 'debug', 'effort'})
    EFFORT_VALID: frozenset[str] = frozenset()  # override per-provider

    @property
    def api_params(self) -> dict:
        """self.model_params minus non-API keys like 'effort'."""
        return {k: v for k, v in self.model_params.items() if k not in self._NON_API_PARAMS}

    def resolve_effort(self) -> tuple[str | None, str | None]:
        """Returns (native_level, warning_message). Caller decides how to log/warn.
        Returns (None, None) if effort is unset or unsupported."""
        ...
```

3. **Swap `**self.model_params` → `**self.api_params` in ALL nine call sites BEFORE seeding effort** (see splat inventory above). This is a mechanical prep commit; the effort-seed commit is a no-op API-shape-change on top.
4. Each provider implements/uses `resolve_effort()` (with its own mapping table + model regexes) at the point where it builds `api_params`, and injects the result in its native wire format. Provider decides whether to log the warning via module `logger` (preferred, matches existing `logger.warning(...)` usage in anthropic_models.py:145, 191, 260, 275) or via `warnings.warn(msg, EffortDownmapWarning)`.
5. **Eager validation via `_VALIDATORS` registry (decision #2).** `BaseModel.set()` is never called by attribute assignment. Change `Model.__setattr__` (model.py:66-72) to consult the provider's `_VALIDATORS` dict and run the matching validator before writing. Also validate in each provider's `__init__` for the constructor path (`Model('x', effort='banana')`). Concrete shape:

```python
# basemodel.py
class BaseModel(ABC):
    _VALIDATORS: dict[str, Callable[[Any], None]] = {}   # subclasses register

    def _validate_effort(self, value):
        allowed = self.EFFORT_VALID | {None}
        if value not in allowed:
            raise ValueError(
                f"effort must be one of {sorted(v for v in allowed if v)} or None; got {value!r}"
            )

# each provider __init__ registers its validator:
self._VALIDATORS = {'effort': self._validate_effort}

# model.py Model.__setattr__:
def __setattr__(self, name, value):
    if name not in self.__dict__ and hasattr(self, 'model') and name in self.model.model_params:
        validator = self.model._VALIDATORS.get(name)
        if validator:
            validator(value)   # raises ValueError before write
        self.model.model_params[name] = value
    else:
        super().__setattr__(name, value)
```

Registry pattern extends cleanly to future validated params (temperature range, max_tokens min bounds) without touching `Model.__setattr__` again. **Typo detection (e.g. `model.efort = 'low'`) intentionally NOT added** — would require raising on unknown attribute writes, which is a broader contract change that could break legitimate instance-attribute usage in downstream code.

### Research Insights

**Why the plan's original assertion about `BaseModel.set()` is wrong (feasibility + correctness reviewers):**
- `Model.__setattr__` (model.py:69) writes `self.model.model_params[name] = value` **directly** — never touches `BaseModel.set()`.
- The construction path `Model('x', effort='banana')` flows through `ModelFactory.create(**kwargs)` → provider `__init__(params=kwargs)` → `BaseModel.__init__(self, model_name, params, system_message)` at basemodel.py:67-72 which assigns `self.model_params = params` without validation.
- **Result:** the plan's Test #8 (`effort='banana'` → ValueError at assignment) does not pass as designed. Fix required.

**Two implementation paths for eager validation:**

Option A (surgical, recommended):
```python
# In Model.__setattr__:
def __setattr__(self, name, value):
    if name not in self.__dict__ and hasattr(self, 'model') and name in self.model.model_params:
        # Validate before writing
        self.model.validate_param(name, value)
        self.model.model_params[name] = value
    else:
        super().__setattr__(name, value)

# In each provider (or BaseModel with per-provider override):
def validate_param(self, name: str, value):
    if name == 'effort':
        allowed = self.EFFORT_VALID | {None}
        if value not in allowed:
            raise ValueError(f"effort must be one of {sorted(v for v in allowed if v)} or None; got {value!r}")
```

Option B: also call `self.validate_param('effort', params['effort'])` at the end of each provider `__init__` (covers the constructor path).

**xAI-specific ordering trap:** xai_models.py:15 calls `BaseModel.__init__(...)` directly, skipping `OpenAIResponsesModel.__init__`. If the effort seed is placed only in `OpenAIResponsesModel.__init__`, xAI silently misses it. Hoisting to `BaseModel.__init__` (recommendation 1 above) avoids this.

**`resolve_effort()` return shape (kieran-python):** Return `tuple[str | None, str | None]` — `(mapped_level, warning_message_or_None)`. Rationale: separates policy (what to map) from side effect (how to log). Testable without `pytest.warns` gymnastics. Caller does:
```python
level, warning = self.resolve_effort()
if warning:
    logger.warning(warning)  # OR: warnings.warn(warning, EffortDownmapWarning)
```

**`_NON_API_PARAMS` hoist:** Currently only anthropic_models.py has `_NON_API_PARAMS = {'timeout', 'async', 'debug'}` at line 51. Other providers don't strip these — meaning `timeout`, `async`, `debug` may leak into non-Anthropic API calls today. Hoisting to `BaseModel` (as shown in code sketch above) fixes a real latent bug for free. Subclasses extend if they have provider-specific non-API keys.

**Cache-key clean break (decision #3):**
- After upgrade, `effort=None` appears in every `model_params` dict.
- `tools/cache.py:32-40` hashes `model_params` via `recursive_hash`.
- **Decision:** bump the cache namespace — accept the one-time invalidation, all pre-upgrade entries drop. Preserving old entries via `ignore_params=['effort']` was considered and rejected as fragile (would need to be re-applied for every future `_NON_API_PARAMS` addition).
- **Implementation:** prepend a version prefix to the hashcode in `cache.py`, e.g. `hashcode = 'v2:' + recursive_hash(...)`. Bump on any future `_NON_API_PARAMS` change.
- **Release notes must state:** "upgrading to vX.Y.Z invalidates the local response cache; first call per prompt hits the API."

---

## Related fixes to include (Anthropic parameter hygiene)

**Decision #1: bundle into this PR.** Reviewers (scope-guardian, code-simplicity) recommended splitting, but decision is to keep together — single release, single changelog entry, and the shared Fable/Mythos test fixtures used by the effort tests would otherwise be duplicated across PRs.

While in `anthropic_models.py`:

- Extend `NO_TEMPERATURE_MODELS` (currently only `claude-opus-4-7`) to also match `claude-opus-4-8`, `claude-fable-5`, `claude-mythos-5`. On these models temperature must be 1.0/unset and top_p/top_k are restricted, otherwise the API returns 400. Strip `temperature`, `top_p` and `top_k` from params for them. **Emit a UserWarning once per Model instance** when stripping — silent drop was flagged as an asymmetry by api-contract reviewer (effort warns on downmap; temperature shouldn't be silent).
- **Consider generalizing** to a `RESTRICTED_PARAMS: dict[re.Pattern, frozenset[str]]` mapping (kieran-python suggestion) instead of extending `NO_TEMPERATURE_MODELS`. Kills duplication once a third restriction pattern lands.
- Do **not** send any `thinking` config for Fable/Mythos (adaptive thinking is always on; `thinking: {"type": "disabled"}` returns 400).
- Handle `stop_reason == "refusal"` (Fable 5 safety classifiers return HTTP 200 with this stop reason). **The check must be inserted BEFORE `message.content[0].text` at anthropic_models.py:154** (currently IndexErrors on empty content). Insert in `chat()` immediately after either `_completion_with_structured_output` or `completion()` returns:
  ```python
  if getattr(message, 'stop_reason', None) == 'refusal':
      raise RefusalException(getattr(message, 'refusal_category', 'unknown'))
  ```
  Also add the same check in the async `stream()` method's `message_delta` handler, and short-circuit the retry loop in `completion()` (refusal is deterministic; retries waste API budget).
- Raise a new `RefusalException(Exception)` from `basemodel.py` — **inherit from `Exception` directly**, not `GeneralException` (kieran-python: refusals are a specific expected outcome; burying under catch-all is wrong; `BadRequestException` is also wrong because nothing was wrong with the request).
- Export from the package `__init__` — but **do it consistently**: currently `__init__.py` exports zero exceptions from basemodel.py. Either export the whole hierarchy (ConnectionException, AuthorizationException, ModelOverloadException, RatelimitException, BadRequestException, TimeoutException, GeneralException, RefusalException) or none. Half-export makes the inconsistency worse.

### Research Insights

**Refusal short-circuit in retry loop (reliability):** `anthropic_models.py:302` has `for _ in range(3):` for tool-use retries. Refusal is deterministic — retries waste budget. Check must `raise`, not `continue`.

**RefusalException should carry structured data:**
```python
class RefusalException(Exception):
    def __init__(self, category: str, message: str = ''):
        self.category = category
        super().__init__(f'Model refused response: {category}. {message}')
```
Test that the classifier category is populated from the API response.

**Simplification alternative (code-simplicity):** Could reuse `BadRequestException(f'refusal: {classifier}')`. Rejected: refusals are HTTP 200 (not bad requests) and users typically want a distinct catch. Keep `RefusalException` but keep it simple (no complex hierarchy).

---

## Tests

Add `tests/test_effort.py`. Mock the provider clients (no real API calls needed for the mapping logic):

1. `effort=None` → no `output_config.effort` / `reasoning` / `thinking_config` sent (assert `'effort' not in mock.call_args.kwargs` for direct proof, not just "no reasoning key").
2. Each valid level on `claude-fable-5` → correct `output_config['effort']`, merged with json_schema format when `response_format` is used (both keys present).
3. `xhigh` on `claude-opus-4-6` → sends `max`, emits `EffortDownmapWarning` (use `pytest.warns`).
4. `max` on `claude-sonnet-4-5` → nothing sent, `EffortDownmapWarning`.
5. `max` on `gpt-5.6-sol` → `reasoning={'effort': 'max'}`; `'none'` on `gpt-5.6-luna` → `reasoning={'effort': 'none'}`.
6. `max` on an older OpenAI reasoning model → `high` + warning.
7. `high` on `gemini-3-pro` (or current Gemini 3 id) → `thinking_config.thinking_level == 'HIGH'`.
8. `effort='banana'` → `ValueError` at assignment (SPLIT into two tests: constructor kwarg AND `model.effort='banana'` post-init — different code paths).
9. `'none'` on `claude-fable-5` → `ValueError`.
10. Cache key changes when effort changes (assert via direct `recursive_hash` inequality — more robust than cache-read/miss).
11. Smoke test (skipped unless API keys present) hitting one cheap model per provider with `effort='low'`. **Use `monkeypatch.setenv('CACHE_DIR', tmp_path)` + `_reset_cachedb_singleton()` autouse fixture** to avoid polluting dev cache.

**Additional tests to add (from testing reviewer):**

12. **Streaming/async coverage:** assert effort reaches `client.messages.create(stream=True, ...)` and `async_client.messages.create(stream=True, ...)` on the Anthropic side, and both streaming call sites in openai_responses.py. Plan explicitly says "merge into async/streaming call paths" but doesn't test it.
13. **Multi-turn tool loop:** mock a first response with a `tool_use` block, verify all three loop iterations in `openai_responses.py:77` send `reasoning`.
14. **Warning dedup semantics:** assert two `chat()` calls on the same Model with a downmap emit **exactly one** warning, not two. Use `warnings.simplefilter('always')` in a `catch_warnings` context + a `len(w) == 1` assertion.
15. **OpenRouter reasoning field:** valid effort on `openrouter/openai/gpt-5.6` → forwards `reasoning`. Not tested at all in the current plan.
16. **`api_params` correctly strips effort:** mock inspection, assert `'effort' not in mock.call_args.kwargs`.
17. **RefusalException classifier:** mock a response with `stop_reason='refusal'` and a category field; assert the raised exception's `.category` matches. Assert `from justai import RefusalException` works (import contract).
18. **Refusal does NOT retry:** mock returns refusal on first call; assert `mock.call_count == 1`, not 3.
19. **Effort survives multi-turn tool loop:** on multi-turn function-call flow (openai_responses.py:77), assert `reasoning` in all iterations' call_args.
20. **No-op provider strip:** setting effort on GGUF/DeepSeek/Reve does NOT leak `effort` kwarg into the underlying client call (would 400 or TypeError). Regression guard.

**Test infra recommendations:**

- **Autouse fixture** to reset `CacheDB` singleton (cache.py:93) and clear `warnings.__warningregistry__` for the test module. Cache singleton state and warning-dedup state both cause test-ordering flakes.
- **Mock pattern:** `mock_client = MagicMock(); model.client = mock_client; model.chat(...); kwargs = mock_client.messages.create.call_args.kwargs; assert kwargs['output_config'] == {'effort': 'low', 'format': {...}}`. Avoid `assert_called_once_with(...)` — brittle to unrelated kwarg additions. Use subset-of-kwargs asserts.
- **For async:** use `AsyncMock` for `async_client`.
- **Smoke test #11:** skip per-provider inside the test body (not blanket `skipif` at the top) so a dev with only ANTHROPIC_API_KEY gets partial coverage instead of a full skip.
- **Regression protection:** add a fast unit test asserting Anthropic structured output still works (no dict-merge bug clobbering `format` key). Current tests/capabilities_test.py hits real APIs — no fast regression test exists for this.

**Test names must be descriptive:**
- `test_effort_none_sends_no_reasoning_field`
- `test_xhigh_on_opus_4_6_maps_up_to_max_with_warning`
- `test_effort_invalid_value_raises_at_constructor`
- `test_effort_invalid_value_raises_at_setattr`
- `test_downmap_warning_emitted_once_across_multiple_calls`

Run existing tests (`tests/capabilities_test.py` etc.) to check nothing regressed — especially structured output on Anthropic, since `output_config` is shared.

---

## Docs & release

- README: add an "Effort" section under Features with a short example, the level table above (condensed) and the downmapping/warning behavior. Mention that level names are not calibrated across models (Anthropic documents this explicitly) — re-test when switching models.
- Update `CLAUDE.md` if it documents supported parameters.
- Bump minor version via `bumpversion.py` (new feature, backwards compatible).
- **Release notes must call out:**
  - Cache-key change: existing entries invalidated on upgrade unless `effort=None` mitigation implemented (see plumbing section).
  - New `RefusalException` exported (if Anthropic hygiene PR lands together).
  - `NO_TEMPERATURE_MODELS` now warns on strip (if hygiene PR lands together — pre-existing silent behavior becomes noisy).
  - New `EffortDownmapWarning` category for filtering.

### Research Insights

**Documentation gaps to close:**
- README section must show the `None` vs `'none'` distinction side-by-side with an example — this is the top footgun.
- Document `Model.supports_effort: bool` (new capability flag) for provider-agnostic code.
- Document that `effort` is captured at request initiation (mid-stream mutation ignored for that request).
- Document opt-out: `warnings.filterwarnings('ignore', category=EffortDownmapWarning)`.
- Document interaction with `max_tokens` (reasoning tokens count against Anthropic output budget) and `timeout` (max effort needs longer wall clock).

---

## Reference (verified July 2026)

Corrected against current SDK sources and provider docs:

- **Anthropic effort docs**: https://platform.claude.com/docs/en/build-with-claude/effort — levels low/medium/high/xhigh/max; xhigh only on Fable 5, Mythos 5, Opus 4.8, 4.7, Sonnet 5; default high; GA, **no beta header required (previously `effort-2025-11-24` for Opus 4.5, now unconditional)**.
- **Anthropic SDK source** (verifies wire shape): https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/effort_capability.py
- **Fable 5 API changes**: adaptive thinking only, refusal stop reason, sampling params restricted.
- **OpenAI GPT-5.6** (gpt-5.6-sol/-terra/-luna): SDK `ReasoningEffort` type = `Literal["none", "minimal", "low", "medium", "high", "xhigh"]`. **`max` not in the SDK type** — may still be server-accepted for gpt-5.6 pro-mode but bypasses Python type checking. `reasoning={'effort': ...}` in the Responses API; bare `gpt-5.6` routes to sol.
- **OpenAI SDK source**: https://github.com/openai/openai-python/blob/main/src/openai/types/shared/reasoning_effort.py
- **Gemini 3**: `thinking_level` MINIMAL/LOW/MEDIUM/HIGH (Pro lacks MINIMAL); `ThinkingLevel` is a case-insensitive enum so string values work; `thinking_level` replaces Gemini 2.5's `thinking_budget: int`.
- **Google GenAI source**: https://github.com/googleapis/python-genai/blob/main/google/genai/types.py
- **xAI**: `reasoning={"effort": "..."}` on Responses endpoint (nested), `reasoning_effort="..."` on legacy Chat Completions (top-level). Current models: `grok-4.5`, `grok-4.3`, `grok-4.20-multi-agent`. **`grok-3-mini` deprecated 2026-05-15, retired 2026-08-15.** Original `grok-4` rejects the parameter.
- **xAI docs**: https://docs.x.ai/developers/model-capabilities/text/reasoning
- **OpenRouter**: unified `reasoning={"effort": "..."}` with server-side downmapping. Also supports `reasoning={"max_tokens": N}` and `reasoning={"exclude": true}`.
- **OpenRouter docs**: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

---

## Implementation ordering (derived from feasibility findings)

**Land in two commits/PRs. First commit is a no-op prep; second commit is the feature bundle (decision #1).**

1. **Commit 1 — splat swap (prep, no behavior change):** Add `_NON_API_PARAMS` and `api_params` property to `BaseModel`. Swap all nine `**self.model_params` splat sites to `**self.api_params`. Verify existing tests pass. This unblocks the effort seed (otherwise Google's `GenerateContentConfig` will `ValidationError` on the unknown `effort` kwarg).
2. **Commit 2 — effort feature + Anthropic hygiene bundle:**
   - Seed `effort=None` in `BaseModel.__init__` (kills 8 duplicated copies, sidesteps xAI parent-bypass).
   - Add `resolve_effort()` per provider with per-model regex + mapping tables (hard-coded, no algorithmic tiebreaker).
   - Inject wire format at all call sites via wrapper methods `_responses_create` / `_responses_parse` on `OpenAIResponsesModel` (anti-drift for future call sites).
   - Add `_VALIDATORS` registry pattern in `BaseModel`; wire `Model.__setattr__` to consult it (decision #2).
   - Add `EffortDownmapWarning(UserWarning)` subclass.
   - Bump cache namespace prefix (decision #3).
   - Cap OpenAI vocabulary at `xhigh` (decision #4).
   - Extend `NO_TEMPERATURE_MODELS` (or introduce `RESTRICTED_PARAMS: dict[re.Pattern, frozenset[str]]`); add refusal handling with `RefusalException` (inherit from `Exception` directly).
   - Export full exception hierarchy from `justai/__init__.py` (not just `RefusalException`).
   - Auto-raise `max_tokens` floor when effort ≥ high; auto-raise timeout when effort ≥ xhigh.
   - Ship with README "Effort" section and full test suite (20 tests, not 11).

**Deferred to future work (not in this PR):**
- `Model.supports_effort` / `effort_levels` capability introspection (decision #5)
- Two-layer API with per-provider override kwargs (`anthropic_thinking_budget=`, etc.)
- o-series support via `openai_completions.py` (add or explicitly TODO)
- Fixing the image-gen path's fresh `OpenAI()` client (openai_responses.py:709)
