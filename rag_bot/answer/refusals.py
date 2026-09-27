"""Pre-written refusal copy (architecture 13.3).

A refusal is the one place where a fixed, reviewable string is strictly better
than a plausible one, so none of this is model-generated.

Constraints these templates exist to hold, and why each one is a real risk:

- **No figure, ever.** A refusal that quotes a return or a percentage while
  refusing to discuss returns is self-refuting, and it is the one place the user
  is most likely to believe the number.
- **No advisory verb in class C.** "should", "recommend" and "best" are banned
  from the C copy: a refusal that ends "...so the ELSS is a good choice" has
  given advice while claiming not to.
- **No URL in class B.** A class-B answer that cites a source would point the
  user at a page that does not contain what they asked for.
- **Class D links the official factsheet** because the corpus deliberately holds
  zero returns; handing over the AMC's own document is the honest exit rather
  than a dead end.

`refusal_b` is the single source of truth for the class-B message. The relevance
gate in `rag_bot/retrieve/gate.py` imports it from here rather than keeping its
own copy -- two copies of the same user-facing string is exactly the stale-copy
bug the phase renumbering had to fix.
"""
from __future__ import annotations

# Class C's educational link. AMFI is the mutual-fund industry body and its site
# carries investor-education material, which is what a "I won't advise you"
# refusal should point at.
#
# VERIFIED 200 on 2026-09-27. SEBI's investor portal (investor.gov.in) was the
# other candidate and did NOT respond from this machine, so it is not used: per
# the P1 rule, an unverified status is recorded rather than asserted. The root
# rather than a deep education path, because a guessed deep path that 404s is
# worse than a verified homepage. It is a parameter so a deeper, verified page
# can replace it without touching the copy.
DEFAULT_EDUCATIONAL_URL = "https://www.amfiindia.com/"

TOPIC_LABELS: dict[str, str] = {
    "expense_ratio": "expense ratio",
    "exit_load": "exit load",
    "min_sip": "minimum SIP",
    "lock_in": "ELSS lock-in period",
    "riskometer": "riskometer level",
    "benchmark": "benchmark index",
}

DEFAULT_COVERED_TOPICS: tuple[str, ...] = (
    "expense_ratio", "exit_load", "min_sip", "lock_in", "riskometer", "benchmark",
)


def _topic_list(covered_topics: list[str] | tuple[str, ...]) -> str:
    labels = [TOPIC_LABELS.get(t, str(t).replace("_", " ")) for t in covered_topics]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def refusal_c(question: str, educational_url: str = DEFAULT_EDUCATIONAL_URL) -> tuple[str, str | None]:
    """(text, url). Names the facts-only limit; never restates the premise as advice.

    `question` is accepted for logging context but deliberately not quoted back:
    echoing a user's "which of these is best?" invites a template that finishes
    the sentence as a recommendation.
    """
    text = (
        "I answer only from the scheme pages I have indexed -- expense ratio, exit "
        "load, minimum SIP, lock-in, riskometer and benchmark -- and I do not take "
        "a view on which scheme to buy, hold or switch to."
    )
    if educational_url:
        return (f"{text} For background on choosing a mutual fund: {educational_url}",
                educational_url)
    return (text, None)


def refusal_d(scheme_name: str, factsheet_url: str | None) -> tuple[str, str | None]:
    """(text, url). Returns are not provided or compared; NO figure is quoted.

    The scheme name is echoed because "which of these five" needs to know which
    fund the refusal is about; the name is a label, not a figure.
    """
    who = scheme_name or "these schemes"
    text = (
        f"I don't state, compare or estimate returns, NAV or performance figures, "
        f"for {who} or any other scheme here -- the indexed pages don't contain "
        f"them, so any number I gave you would be invented."
    )
    if factsheet_url:
        return (f"{text} The AMC's official factsheet has the full performance table: "
                f"{factsheet_url}", factsheet_url)
    return (f"{text} The AMC's official factsheet for the scheme carries the full "
            f"performance table.", None)


def refusal_b(covered_topics: list[str] | tuple[str, ...] = DEFAULT_COVERED_TOPICS) -> tuple[str, None]:
    """(text, None). Names the covered topics. Never contains a URL."""
    listed = _topic_list(covered_topics)
    if not listed:
        return (
            "I don't have that in the indexed sources. I can only answer from the "
            "scheme pages I have indexed.", None
        )
    return (
        "I don't have that in the indexed sources. I can only answer questions "
        f"about {listed}, for these five HDFC schemes. I deliberately do not "
        "carry returns, NAV or performance figures.", None
    )
