"""Pre-retrieval triage router tests.

The release gate in this file is `test_no_class_a_question_is_refused`. Zero
false positives is not a quality metric here, it is a ship/no-ship condition: a
false positive means the assistant refuses a question the brief requires it to
answer, on one of the six in-scope topics, for one of the five schemes.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_bot.answer.refusals import (
    DEFAULT_EDUCATIONAL_URL,
    refusal_b,
    refusal_c,
    refusal_d,
)
from rag_bot.answer.schemes import resolve_scheme
from rag_bot.answer.triage import (
    BANNED_BARE_KEYWORDS,
    RULES_C,
    RULES_D,
    classify,
    classify_llm,
    classify_rules,
)
from rag_bot.config import load
from rag_bot.sources.fetch import load_sources
from rag_bot.types import Outcome, Source

EVAL_FILE = Path(__file__).resolve().parent.parent / "rag_bot" / "eval" / "eval_set.yaml"

# The four prepared demo questions from the PRD's class-C/class-D table. All four
# must be layer-1 hits: the refusal demo costs zero LLM calls and cannot fail on a
# network hiccup (architecture 13.2).
DEMO_QUESTIONS = [
    ("Should I buy the HDFC Large Cap Fund?", Outcome.C_ADVICE_REFUSED),
    ("Is small cap better for me?", Outcome.C_ADVICE_REFUSED),
    ("Which of these has the best 1-year return?", Outcome.D_PERFORMANCE_REFUSED),
    ("How does the small cap fund rank against its peers?", Outcome.D_PERFORMANCE_REFUSED),
]


class CountingLLM:
    """Records calls so 'zero LLM calls' is asserted, not assumed."""

    def __init__(self, reply: str = "A_or_B") -> None:
        self.name = "counting"
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        return self.reply

    def classify(self, system: str, user: str, *, labels: list[str]) -> str:
        self.calls += 1
        return self.reply


class ExplodingLLM:
    name = "exploding"

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        raise RuntimeError("provider is down")

    def classify(self, system: str, user: str, *, labels: list[str]) -> str:
        raise RuntimeError("provider is down")


@pytest.fixture(scope="module")
def sources() -> list[Source]:
    cfg = load()
    return load_sources(cfg.sources_file)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return yaml.safe_load(EVAL_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# THE RELEASE GATE
# ---------------------------------------------------------------------------

def test_no_class_a_question_is_refused(rows, sources):
    """Zero class-A false positives, over all 33 class-A rows.

    The spec: "Any false positive here is a release blocker, because it means the
    assistant refuses a question it is required to answer." If this ever fails,
    lengthen the offending pattern. Never add a banned bare keyword.

    Layer 1 only, which is the release gate. Asserting this with layer 2 switched
    on would measure the model rather than the rule set.
    """
    offenders = [
        (r["id"], classify(r["question"], sources=sources, use_llm=False).reason)
        for r in rows if r["outcome"] == "A"
        and classify(r["question"], sources=sources, use_llm=False).outcome
        is not Outcome.A_ANSWERED
    ]
    assert not offenders, f"class-A questions refused: {offenders}"


def test_layer_one_is_silent_on_every_class_a_question(rows, sources):
    """No RULE can refuse a question the brief requires the assistant to answer.

    This is the release gate. Every class-A question is a fact question, so layer 1
    correctly returns "unsure" -- which means layer 2 is then consulted, and a
    misfiring classifier CAN refuse an answerable question. That is inherent to the
    two-layer design and the spec does not claim otherwise; what layer 1 guarantees
    is that the deterministic layer never contributes such a refusal.
    """
    offenders = [
        (r["id"], r["question"])
        for r in rows if r["outcome"] == "A" and classify_rules(r["question"]) is not None
    ]
    assert not offenders, f"layer 1 fired on a class-A question: {offenders}"


def test_a_broken_classifier_degrades_toward_answering(rows, sources):
    """The safety property that actually holds at layer 2: unusable in, class A out.

    A provider that raises, returns nothing, or invents a label yields None, and
    None becomes "assume A". So the failure mode of the fallback layer is a wasted
    300 ms, not a false refusal of a required answer.
    """
    for provider in (ExplodingLLM(), CountingLLM(""), CountingLLM("maybe?"),
                     CountingLLM("C_advice, D_performance")):
        for row in rows:
            if row["outcome"] != "A":
                continue
            result = classify(row["question"], sources=sources, provider=provider,
                              use_llm=True)
            assert result.outcome is Outcome.A_ANSWERED, (row["id"], result.reason)


def test_rule_set_catches_every_labelled_c_and_d_row(rows, sources):
    """The other half of the contract. A rule set that refused nothing at all would
    pass the gate above, and would be a useless router."""
    missed = [
        (r["id"], r["question"])
        for r in rows if r["outcome"] in ("C", "D")
        and classify(r["question"], sources=sources, use_llm=False).outcome.value
        != r["outcome"]
    ]
    assert not missed, f"labelled C/D rows that layer 1 misses: {missed}"


def test_b_rows_stay_out_of_triage(rows, sources):
    """Class B is the retriever's job, not the router's.

    A B question that triage claims is not what the corpus-not-found path is for,
    and a B question routed to D would trade "here is what I cover" for a
    factsheet link on a question that was never about returns.
    """
    wrong = [
        (r["id"], classify(r["question"], sources=sources, use_llm=False).reason)
        for r in rows if r["outcome"] == "B"
        and classify(r["question"], sources=sources, use_llm=False).outcome
        is not Outcome.A_ANSWERED
    ]
    assert not wrong, wrong


def test_in_scope_topic_questions_never_triage(rows, sources):
    """Architecture 13.2's named gate: the six in-scope topic questions by name."""
    questions = [
        "What is the ELSS lock-in period?",
        "What is the exit load?",
        "What is the expense ratio?",
        "What is the minimum SIP?",
        "What is the riskometer level?",
        "Which index is the benchmark?",
        "Riskometer level for the small cap fund, please.",
        "Where does the balanced advantage fund sit on the risk scale?",
        "Against which index does the tax saver fund perform?",
    ]
    bad = [q for q in questions
           if classify(q, sources=sources, use_llm=False).outcome is not Outcome.A_ANSWERED]
    assert not bad, bad


# ---------------------------------------------------------------------------
# layer 1: rules
# ---------------------------------------------------------------------------

def test_demo_questions_are_layer_one_hits_with_zero_llm_calls(sources):
    llm = CountingLLM("A_or_B")
    for question, expected in DEMO_QUESTIONS:
        result = classify(question, sources=sources, provider=llm, use_llm=True)
        assert result.outcome is expected, (question, result)
        assert result.layer == "rules", (question, result.layer)
    assert llm.calls == 0, "the refusal demo must not spend an LLM call"


@pytest.mark.parametrize("rules", [RULES_C, RULES_D])
def test_no_rule_is_a_bare_banned_keyword(rules):
    """A bare banned keyword is a class-A fact question waiting to be refused."""
    for name, pattern in rules:
        text = pattern.pattern.lower()
        for banned in BANNED_BARE_KEYWORDS:
            assert not re_is_bare(text, banned), f"{name} is the bare keyword {banned!r}"


def re_is_bare(pattern_text: str, word: str) -> bool:
    """True if the pattern matches the word and nothing else (no qualifier)."""
    import re

    return re.fullmatch(rf"\\b(?:{re.escape(word)})\\b", pattern_text) is not None


def test_banned_keywords_still_work_inside_longer_phrases():
    """"return" is banned alone but "5 year return" must still fire."""
    assert classify_rules("What is the 5 year return?").outcome is Outcome.D_PERFORMANCE_REFUSED
    assert classify_rules("Should I buy it?").outcome is Outcome.C_ADVICE_REFUSED
    # "invest" alone is banned; a class-A corpus question contains it.
    assert classify_rules("How much money is invested in the small cap fund?") is None
    # "rank" alone is banned; the preposition or the peer noun is required.
    assert classify_rules("What rank does it hold?") is None
    assert classify_rules("How does it rank against its peers?") is not None


def test_best_returns_is_a_performance_question_even_when_advice_shaped():
    """The spec's own example. It has no "perform" in it at all."""
    result = classify_rules("Should I buy the fund with the best returns?")
    assert result.outcome is Outcome.D_PERFORMANCE_REFUSED
    assert result.reason == "d_best_return"


def test_class_d_is_checked_before_class_c():
    """'should I buy the fund with the best returns' is a performance question first.

    Routing it to C would produce a no-advice refusal that never mentions the
    request was for a figure.
    """
    result = classify_rules("Should I buy the fund with the best returns?")
    assert result.outcome is Outcome.D_PERFORMANCE_REFUSED
    assert result.reason.startswith("d_")


def test_rules_returns_none_when_unsure():
    assert classify_rules("What is the expense ratio of the large cap fund?") is None
    assert classify_rules("") is None
    assert classify_rules("   ") is None


def test_classify_falls_back_to_a_when_unsure(sources):
    result = classify("What is the expense ratio of the large cap fund?", sources=sources)
    assert result.outcome is Outcome.A_ANSWERED
    assert result.reason == "no_rule_matched"
    assert result.layer is None


def test_reason_is_the_matched_rule_name():
    result = classify_rules("Which of these has the best 1-year return?")
    assert result.reason == "d_year_return"
    assert result.layer == "rules"


@pytest.mark.parametrize("question,expected", [
    ("Which of these HDFC funds has the best 1-year return?", "d_year_return"),
    # reason is the FIRST declared rule that matches, so the CAGR row reports
    # since_inception. Both are D rules and either name is a correct audit trail;
    # what matters is that the reason is a rule that genuinely fired.
    ("The large cap fund's CAGR since inception?", "d_since_inception"),
    ("How does the small cap fund rank against its peers?", "d_rank_against_peers"),
    ("Show me the returns I would have made with a 5000 rupee SIP for 5 years.",
     "d_returns_would_have_made"),
    ("What is the NAV of the fund?", "d_nav_of"),
])
def test_each_d_rule_catches_its_own_phrasing(question, expected):
    assert classify_rules(question).reason == expected


@pytest.mark.parametrize("question,expected", [
    ("Should I buy the HDFC Large Cap Fund?", "c_should_i"),
    ("Is the ELSS tax saver a good investment for me?", "c_good_investment"),
    ("Is small cap better for me?", "c_better_for_me"),
    ("Which is better for me?", "c_which_better_for_me"),
    ("Is it safe to invest in the ELSS?", "c_is_it_safe_to"),
    ("What is a good time to buy?", "c_good_time_to"),
    ("Is the ELSS worth buying?", "c_worth_buying"),
    ("What is my portfolio allocation?", "c_portfolio_allocation"),
    ("Can I exit the ELSS early?", "c_can_i_exit"),
])
def test_each_c_rule_catches_its_own_phrasing(question, expected):
    assert classify_rules(question).reason == expected


def test_word_and_digit_year_forms_both_fire():
    """Same question, two spellings, one class. Firing on '5 year' while missing
    'five-year' would be indefensible, and the spec mandates the digit form."""
    for q in ("the 5 year return", "the five year return", "the 3-year return",
              "the three-year return"):
        assert classify_rules(q).outcome is Outcome.D_PERFORMANCE_REFUSED, q


# ---------------------------------------------------------------------------
# layer 2: the LLM classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply,expected", [
    ("C_advice", Outcome.C_ADVICE_REFUSED),
    ("D_performance", Outcome.D_PERFORMANCE_REFUSED),
    ("A_or_B", Outcome.A_ANSWERED),
])
def test_classify_llm_maps_labels(reply, expected):
    result = classify_llm("some question", CountingLLM(reply))
    assert result is not None and result.outcome is expected
    assert result.layer == "llm"


def test_classify_llm_never_raises_and_returns_none_on_failure():
    assert classify_llm("q", ExplodingLLM()) is None
    assert classify_llm("q", CountingLLM("nonsense label")) is None
    assert classify_llm("q", CountingLLM("")) is None
    assert classify_llm("", CountingLLM()) is None
    assert classify_llm("q", None) is None


def test_classify_llm_tolerates_quoting_and_whitespace():
    assert classify_llm("q", CountingLLM('  "C_advice".  ')).reason == "llm:C_ADVICE"


def test_layer_one_short_circuits_layer_two(sources):
    llm = CountingLLM("A_or_B")
    result = classify("Should I buy the HDFC Large Cap Fund?", sources=sources,
                      provider=llm, use_llm=True)
    assert result.reason == "c_should_i" and result.layer == "rules"
    assert llm.calls == 0


def test_layer_two_runs_only_when_layer_one_is_unsure(sources):
    llm = CountingLLM("D_performance")
    result = classify("Tell me about the fund in a way nobody has asked before",
                      sources=sources, provider=llm, use_llm=True)
    assert result.outcome is Outcome.D_PERFORMANCE_REFUSED
    assert result.layer == "llm"
    assert llm.calls == 1


def test_use_llm_false_never_calls_the_provider(sources):
    llm = CountingLLM("D_performance")
    result = classify("Tell me something entirely novel", sources=sources,
                      provider=llm, use_llm=False)
    assert result.outcome is Outcome.A_ANSWERED and result.layer is None
    assert llm.calls == 0


# ---------------------------------------------------------------------------
# scheme resolution is always attached
# ---------------------------------------------------------------------------

def test_classify_always_sets_the_scheme(sources):
    result = classify("What is the expense ratio of the ELSS tax saver fund?",
                      sources=sources)
    assert result.scheme_id == "S3"
    assert result.scheme_ambiguous is False


def test_classify_marks_two_schemes_ambiguous(sources):
    result = classify("Compare the expense ratio of the large cap fund and the ELSS fund",
                      sources=sources)
    assert result.scheme_id is None
    assert result.scheme_ambiguous is True


def test_classify_works_without_a_registry():
    """The spec's own verify snippet calls classify(question) with no sources."""
    result = classify("Should I buy the HDFC Large Cap Fund?")
    assert result.outcome is Outcome.C_ADVICE_REFUSED
    assert result.scheme_id is None


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_refusal_c_names_the_limit_and_links_education():
    text, url = refusal_c("Should I buy the ELSS?", DEFAULT_EDUCATIONAL_URL)
    assert url == DEFAULT_EDUCATIONAL_URL
    assert url in text
    assert "expense ratio" in text          # says what it CAN do
    assert "http" in text                   # carries the link


def test_refusal_c_has_no_advisory_verb():
    """'should' / 'recommend' / 'best' are banned from class C.

    A refusal that ends 'so the ELSS is a good choice' has given advice while
    claiming not to.
    """
    for question in ("Should I buy the ELSS?", "Which of these is best for me?",
                     "Can you recommend the small cap fund?"):
        text, _ = refusal_c(question)
        for verb in ("should", "recommend", "best"):
            assert verb not in text.lower(), (question, verb, text)


def test_refusal_c_does_not_restate_the_premise():
    """The user's question is not echoed back, so no template can finish it as a
    recommendation."""
    text, _ = refusal_c("Should I switch from the ELSS to the small cap fund?")
    assert "switch from" not in text.lower()
    assert "small cap" not in text.lower()


@pytest.mark.parametrize("text", [
    refusal_c("Should I buy?")[0],
    refusal_d("HDFC Large Cap Fund – Direct Growth", "https://x.example/factsheet.pdf")[0],
    refusal_d("HDFC Flexi Cap Fund", None)[0],
    refusal_b()[0],
])
def test_no_refusal_quotes_a_figure(text):
    """The spec's rule: no digit followed by '%'.

    Note what is NOT banned: the word "NAV". A refusal is required to say what it
    withholds -- "I don't state, compare or estimate returns, NAV or performance"
    is the whole point -- so the copy may NAME a category and may never quote a
    value from it. Banning the word instead of the figure would make the refusal
    unable to explain itself.
    """
    import re
    assert not re.search(r"\d+(?:\.\d+)?\s?%", text), text
    assert not re.search(r"\b\d+(?:\.\d+)?\s?(?:cagr|nav)\b", text, re.I), text


def test_refusal_d_states_the_limit_and_links_the_factsheet():
    text, url = refusal_d("HDFC Large Cap Fund – Direct Growth",
                          "https://www.hdfcfund.com/mutual-funds/factsheets")
    assert url == "https://www.hdfcfund.com/mutual-funds/factsheets"
    assert "factsheet" in text.lower()
    assert "don't state, compare or estimate" in text


def test_refusal_d_quotes_no_figure_even_with_a_scheme_name():
    text, _ = refusal_d("HDFC Large Cap Fund – Direct Growth", None)
    import re
    assert not re.search(r"\d", text), text
    assert "factsheet" in text.lower()


def test_refusal_d_survives_a_missing_factsheet_url():
    text, url = refusal_d("HDFC Flexi Cap Fund", None)
    assert url is None
    assert "http" not in text
    assert "factsheet" in text.lower()


def test_refusal_b_names_the_six_topics_and_has_no_url():
    text, url = refusal_b()
    assert url is None
    assert "http" not in text
    for label in ("expense ratio", "exit load", "minimum SIP", "lock-in",
                  "riskometer", "benchmark"):
        assert label in text, label


def test_refusal_b_is_the_gate_copy_too():
    """One source of truth. Two copies of the same user-facing string is exactly the
    stale-copy bug the phase renumbering had to fix."""
    from rag_bot.retrieve.gate import DEFAULT_COVERED_TOPICS, _class_b_message

    assert _class_b_message(list(DEFAULT_COVERED_TOPICS)) == refusal_b()[0]
    assert DEFAULT_COVERED_TOPICS == (
        "expense_ratio", "exit_load", "min_sip", "lock_in", "riskometer", "benchmark")


# ---------------------------------------------------------------------------
# layer 1 recall: measured, and layer 2 is the designed answer
# ---------------------------------------------------------------------------

PARAPHRASES = [
    "which one made me the most money",
    "is it worth putting my savings in",
    "do you think I should sit on cash",
    "rank these funds for me",
    "what has been the hottest performer",
    "give me a buy sell call",
    "what's the safest option here",
    "how would you split my portfolio",
    "what did it earn last year",
    "is now the right moment to enter",
    "compare the performance of all five",
]


def test_layer_one_recall_on_paraphrases_is_recorded_not_assumed():
    """Architecture 13.2's premise, asserted so it cannot quietly rot.

    Layer 1 matches named phrases, so paraphrases of advice and performance
    questions MISS it -- measured at 0 of 11 here. That is the designed state, not
    a defect: layer 2 exists precisely so those blind spots are a tuning problem
    rather than a product boundary. What must not change is that layer 1 still
    catches every labelled C/D row and every prepared demo question; if this ratio
    ever moves, the rule list has been over-fitted to the eval set and the
    class-A gate is the thing at risk.
    """
    caught = [q for q in PARAPHRASES if classify_rules(q) is not None]
    assert len(caught) == 0, (
        f"layer 1 now catches {len(caught)}/11 paraphrases: {caught}. If this is "
        f"intentional, update the count AND re-check the class-A gate -- added "
        f"paraphrase rules are the most likely way to break it."
    )


def test_paraphrases_are_still_routed_when_a_classifier_is_available(sources):
    """The end-to-end promise: a paraphrase is handled, by layer 2 if not layer 1."""
    llm = CountingLLM("D_performance")
    for question in PARAPHRASES:
        result = classify(question, sources=sources, provider=llm, use_llm=True)
        assert result.outcome is Outcome.D_PERFORMANCE_REFUSED, (question, result)
        assert result.layer == "llm"
    assert llm.calls == len(PARAPHRASES)
