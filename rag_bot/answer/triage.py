"""Pre-retrieval triage router (architecture 13).

Why this runs *before* retrieval, and not after: a performance question ("which of
these has the best 1-year return?") retrieves extremely well, because returns are
the most prominent content on a fund page. Routed after retrieval, the generator
receives exactly the numbers it must not quote and the constraint degrades to a
prompt request. Routed before retrieval, the numbers are never in the prompt.

P4 measured why this layer is load-bearing rather than a nicety: the relevance
score cannot separate an answerable question from a withheld-fact question,
because the withheld facts are topically adjacent to what is indexed. An AUM
question scored 0.784 against the minimum-SIP chunk. Score alone would answer it
from the wrong fact. See docs/data-findings.md 7.3.

Two layers, deterministic first:

| Layer | Mechanism                | Latency | Catches                        |
|-------|--------------------------|---------|--------------------------------|
| 1     | RULES_D / RULES_C        | ~0 ms   | the brief's own examples       |
| 2     | one cheap LLM call       | ~300 ms | paraphrases layer 1 misses     |

Layer 1 fires first and its hit is logged as layer="rules", so a refusal costs
zero LLM calls and cannot fail on a network hiccup. Layer 2 exists so layer 1's
blind spots are a tuning problem rather than a product boundary.
"""
from __future__ import annotations

import re

from rag_bot.answer.schemes import resolve_scheme
from rag_bot.providers.base import LLMProvider
from rag_bot.types import Outcome, Source, TriageLayer, TriageResult

# ---------------------------------------------------------------------------
# Banned bare keywords
# ---------------------------------------------------------------------------
# These words are BANNED as standalone patterns. Each is a real in-scope fact
# question wearing a performance word's clothes:
#
#   risk     "riskometer level" is a class-A fact, and "sit on the risk scale"
#   return   "returns period" is an exit-load synonym
#   better   comparative fee questions say "which is better"
#   perform  "against which index does it perform?" is a benchmark question
#   growth   growth figures are performance, but "growth" also appears in prose
#   invest   "a good investment for me" is advice-shaped, but "investment" is in
#            "how much money is invested in the fund" (a corpus question)
#   good     "good time to buy" is advice, but "good" is far too common
#
# Each may appear ONLY inside a longer, specific phrase. This is enforced by
# test_no_rule_is_a_bare_banned_keyword and, more importantly, by the class-A
# false-positive gate.
BANNED_BARE_KEYWORDS: frozenset[str] = frozenset(
    {"risk", "better", "return", "perform", "growth", "invest", "good"}
)

# Year forms, digit and word, because users write both. "five-year return" is a
# performance question and "5 year return" is the same question.
_YEAR = r"(?:1|one|2|two|3|three|4|four|5|five)"

# ---------------------------------------------------------------------------
# Class D -- performance
# ---------------------------------------------------------------------------
# (rule_name, pattern). Names are the machine-readable `reason` on the result, so
# they are the thing a demo log shows; keep them stable.
RULES_D: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("d_best_performing", re.compile(r"\bbest[\s-]+perform(?:ing|er|ers)\b", re.I)),
    # "should I buy the fund with the best returns" has no "perform" in it, and
    # the spec calls that a performance question. Without this it routes to C and
    # the user gets a no-advice refusal that never mentions the request was for a
    # figure.
    ("d_best_return", re.compile(r"\bbest[\s-]+returns?\b", re.I)),
    (f"d_year_return", re.compile(rf"\b{_YEAR}[\s-]?year[\s-]+returns?\b", re.I)),
    ("d_since_inception", re.compile(r"\bsince\s+inception\b", re.I)),
    ("d_cagr", re.compile(r"\bcagr\b", re.I)),
    ("d_returns_of", re.compile(r"\breturns?\s+of\b", re.I)),
    # "how much did X return" needs the gap, because "how much" alone is a
    # class-A phrasing ("How much of my investment goes to fees?").
    ("d_how_much_did_return", re.compile(r"\bhow\s+much\s+did\b[^?]{0,40}\breturn", re.I)),
    ("d_nav_of", re.compile(r"\bnav\s+of\b", re.I)),
    ("d_current_nav", re.compile(r"\bcurrent\s+nav\b", re.I)),
    # Bare "NAV" and "AUM": "HDFC flexi cap NAV" is the most natural way to ask for
    # a figure and carries no "of", so the two rules above miss it and the question
    # falls through to class A. Across the 4,567 characters the indexer actually
    # sees, "nav" and "aum" occur zero times, so a bare rule here cannot make a
    # class-A fact unreachable. Placed after the two specific rules so those keep
    # the more precise reason.
    ("d_nav_bare", re.compile(r"\bnavs?\b", re.I)),
    ("d_aum", re.compile(r"\baum\b", re.I)),
    # "yield" is also absent from the corpus and is a figure wherever it turns up
    # in this domain. "growth" is deliberately NOT added: the corpus holds "Direct
    # Growth" three times, so a bare rule there would refuse ordinary questions
    # about the Growth plan.
    ("d_yield", re.compile(r"\byields?\b", re.I)),
    # Bare "performance" as a noun. The verb "perform" stays excluded, per
    # d_which_performed below. Residual risk: "what does the riskometer say about
    # performance?" is arguably class A and would now be refused; no such phrasing
    # exists in the corpus.
    ("d_performance", re.compile(r"\bperformance\b", re.I)),
    # "which performed" / "which one performed best", but NOT bare "perform" --
    # "against which index does the fund perform?" is a benchmark question.
    ("d_which_performed", re.compile(r"\bwhich\s+(?:one\s+)?perform(?:ed|s)\b", re.I)),
    ("d_top_performer", re.compile(r"\btop\s+performers?\b", re.I)),
    ("d_ranking", re.compile(r"\branking\b", re.I)),
    ("d_ranked", re.compile(r"\branked\b", re.I)),
    ("d_percentile", re.compile(r"\bpercentiles?\b", re.I)),
    ("d_peer_group_rank", re.compile(r"\bpeer[\s-]*(?:group\s+)?rank(?:ed|ing)?\b", re.I)),
    # "How does the small cap fund rank against its peers?" -- the spec's
    # "ranking"/"ranked"/"peer group rank" do not cover a bare comparative "rank",
    # so the preposition is required. Never bare "rank".
    ("d_rank_against_peers", re.compile(r"\brank\w*\b[^?]{0,30}\bpeers?\b", re.I)),
    ("d_growth_of", re.compile(r"\bgrowth\s+of\b", re.I)),
    ("d_portfolio_return", re.compile(r"\bportfolio\s+returns?\b", re.I)),
    # "Show me the returns I would have made with a 5000 rupee SIP for 5 years."
    ("d_returns_would_have_made", re.compile(r"\breturns?\s+(?:i|you)\s+would\s+have\s+made\b", re.I)),
    ("d_projected_return", re.compile(r"\bproject(?:ed|ion)\s+returns?\b", re.I)),
)

# ---------------------------------------------------------------------------
# Class C -- advice
# ---------------------------------------------------------------------------
RULES_C: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("c_should_i", re.compile(r"\bshould\s+(?:i|we)\b", re.I)),
    ("c_which_better_for_me", re.compile(r"\bwhich\s+is\s+better\s+for\s+me\b", re.I)),
    # "Is small cap better for me?" is the PRD's own example and the spec's
    # "which is better for me" does not cover it. "for me" is the required tail.
    ("c_better_for_me", re.compile(r"\bbetter\s+for\s+me\b", re.I)),
    ("c_is_it_safe_to", re.compile(r"\bis\s+it\s+safe\s+to\b", re.I)),
    ("c_recommend", re.compile(r"\brecommends?\b", re.I)),
    ("c_good_time_to", re.compile(r"\bgood\s+time\s+to\b", re.I)),
    ("c_worth_buying", re.compile(r"\bworth\s+buying\b", re.I)),
    ("c_suitable_for_me", re.compile(r"\bsuitable\s+for\s+me\b", re.I)),
    ("c_best_scheme_for", re.compile(r"\bbest\s+(?:scheme|fund|option|choice)\s+for\b", re.I)),
    ("c_portfolio_allocation", re.compile(r"\bportfolio\s+allocation\b", re.I)),
    ("c_is_now_a_good_time", re.compile(r"\bis\s+(?:now\s+)?a\s+good\s+time\b", re.I)),
    ("c_can_i_exit", re.compile(r"\bcan\s+i\s+exit\b", re.I)),
    # "Is the ELSS tax saver a good investment for me?" -- the spec has no rule
    # for it, and it is the shape of question class C exists for. "investment"
    # is a banned bare keyword, so the phrase is required, never "investment".
    ("c_good_investment", re.compile(r"\bgood\s+investment\b", re.I)),
    ("c_advice_seeking", re.compile(r"\bwhich\s+of\s+these\b[^?]{0,40}\bshould\b", re.I)),
)

_TRIAGE_LABELS = ["A_or_B", "C_advice", "D_performance"]
# Keyed uppercase, because the reply is normalised to uppercase before lookup: a
# model that answers "c_advice" or "C_ADVICE" has still answered, and re-prompting
# over a capital letter is not a useful failure mode.
_LABEL_TO_OUTCOME = {
    "A_OR_B": Outcome.A_ANSWERED,
    "C_ADVICE": Outcome.C_ADVICE_REFUSED,
    "D_PERFORMANCE": Outcome.D_PERFORMANCE_REFUSED,
}

_SYSTEM = (
    "You classify mutual-fund questions. Reply with exactly one label and nothing "
    "else.\n"
    "A_or_B: asks for a fact from a fund fact sheet (fees, exit load, minimum SIP, "
    "lock-in, riskometer, benchmark index).\n"
    "C_advice: asks for a recommendation or a view on whether to buy, hold or switch.\n"
    "D_performance: asks for returns, NAV, rankings, percentiles or growth figures."
)


def _first_match(question: str, rules: tuple[tuple[str, re.Pattern[str]], ...]) -> str | None:
    for name, pattern in rules:
        if pattern.search(question):
            return name
    return None


def classify_rules(question: str) -> TriageResult | None:
    """Layer 1. Return a TriageResult with outcome C or D, or None if unsure.

    D is checked BEFORE C, because the same sentence can be both: "should I buy
    the fund with the best returns" is a performance question first, and routing
    it to C would produce a no-advice refusal that never mentions that the request
    was for a figure.
    """
    if not question or not question.strip():
        return None

    name = _first_match(question, RULES_D)
    if name is not None:
        return TriageResult(outcome=Outcome.D_PERFORMANCE_REFUSED, reason=name,
                            layer="rules")
    name = _first_match(question, RULES_C)
    if name is not None:
        return TriageResult(outcome=Outcome.C_ADVICE_REFUSED, reason=name,
                            layer="rules")
    return None


def classify_llm(question: str, provider: LLMProvider) -> TriageResult | None:
    """Layer 2. One cheap call, three labels. Never raises.

    Returns None when the model is unusable -- a network error, an empty reply, or
    a label it invented. The caller then treats the question as class A, which is
    the safe direction: the retriever and the relevance gate still stand between
    the question and an answer.
    """
    if not question or not question.strip() or provider is None:
        return None
    try:
        raw = provider.classify(_SYSTEM, question, labels=_TRIAGE_LABELS)
    except Exception:
        return None
    if not raw or not isinstance(raw, str):
        return None

    label = raw.strip().strip("\"'` .").upper()
    outcome = _LABEL_TO_OUTCOME.get(label)
    if outcome is None:
        return None
    return TriageResult(outcome=outcome, reason=f"llm:{label}", layer="llm")


def classify(
    question: str,
    *,
    sources: list[Source] | None = None,
    provider: LLMProvider | None = None,
    use_llm: bool = True,
) -> TriageResult:
    """Layer 1, then layer 2, then assume A. Always resolves the scheme.

    `sources` defaults to empty so the spec's own verification snippet
    (`classify(r['question'])`) runs; with no registry there is simply no scheme
    to resolve, which is the same as an unresolvable scheme.
    """
    result = classify_rules(question)
    layer: TriageLayer | None = "rules"

    if result is None and use_llm and provider is not None:
        result = classify_llm(question, provider)
        layer = "llm" if result is not None else None

    if result is None:
        # Unsure means class A. The retriever plus the relevance gate are the
        # real safety net; triage refusing to guess here would be a worse
        # failure than answering from the indexed pages.
        outcome, reason, layer = Outcome.A_ANSWERED, "no_rule_matched", None
    else:
        outcome, reason = result.outcome, result.reason

    # Resolved on EVERY path, including A. Class A is where a wrong scheme_id does
    # the most damage -- it is the path that produces a figure -- and §14.4 scopes
    # retrieval with this value. An early return that skipped resolution would
    # silently leave every answer unscoped.
    scheme_id, ambiguous = resolve_scheme(question, sources or [])
    return TriageResult(
        outcome=outcome, reason=reason, layer=layer,
        scheme_id=scheme_id, scheme_ambiguous=ambiguous,
    )
