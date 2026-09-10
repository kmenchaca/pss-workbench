"""Murder Mystery Case File - The Chen Killing

A carefully constructed mystery where clues point at two different suspects,
but the solution is that they collaborated.

DESIGN GOALS:
- Clues strongly implicate Suspect A (Elena)
- Other clues strongly implicate Suspect B (David)
- The clues seem contradictory (timeline doesn't work for either alone)
- But the solution is collaboration - both are guilty

SUCCESS CRITERIA:
- Single-shot: Commits to one suspect, rationalizes away contradictions
- PSS: At least one branch asks "what if they worked together?"
"""

CASE_FILE = """
=== HOMICIDE INVESTIGATION CASE FILE ===
Case #2026-0125
Victim: Marcus Chen, 42, CEO of NovaTech Inc.

CIRCUMSTANCES OF DEATH:
- Body discovered January 24, 2026 at 11:15 PM in victim's office
- Cause of death: Acute poisoning (toxicology pending)
- Estimated time of death: Between 9:00 PM and 9:30 PM (coroner's preliminary estimate)
- No signs of forced entry
- Office door was locked from inside (victim had only key)

=== PERSONS OF INTEREST ===

SUSPECT A: Elena Vasquez, 38, CFO of NovaTech
- RELATIONSHIP: Victim's business partner and secret romantic partner (affair confirmed)
- MOTIVE: $2.5M life insurance policy naming her as beneficiary (discovered post-mortem)
- ACCESS: Has master keycard to all offices
- EVIDENCE AGAINST:
  * Blood matching victim's type found on her blazer (claims from CPR attempt)
  * Was seen entering victim's office at 8:25 PM (security log)
  * Left at 8:50 PM per security footage
  * Handwriting analysis: Note found in victim's pocket matches her writing
  * The note reads: "We need to end this. Tonight. -E"
- ALIBI FOR TIME OF DEATH:
  * Claims she was on video call with Tokyo office from 8:55 PM to 9:45 PM
  * Video call logs CONFIRM this - she is visible on screen throughout
  * Multiple witnesses in Tokyo confirm she was on the call

SUSPECT B: David Park, 45, CTO of NovaTech
- RELATIONSHIP: Co-founder, increasingly pushed aside by victim
- MOTIVE: Victim was forcing him out; board vote scheduled for Jan 26
- ACCESS: Security badge logged entry at 7:30 PM, exit at 8:45 PM
- EVIDENCE AGAINST:
  * Threatening email to victim dated Jan 20: "You'll regret this. I built this company."
  * His access badge shows he visited the server room at 8:15 PM
  * Victim's computer shows someone accessed files about "removing David" at 8:40 PM
  * He has chemistry background (PhD from MIT, specialty in pharmacology)
  * His home was searched: found research on undetectable poisons (claims "curiosity")
- ALIBI FOR TIME OF DEATH:
  * Left building at 8:45 PM per security footage (irrefutable)
  * Cell phone records show him at home (2 miles away) by 9:05 PM
  * Claims he was home watching TV alone

=== TOXICOLOGY REPORT (PRELIMINARY) ===
- Poison identified: Modified tetrodotoxin compound (TTX derivative)
- This specific compound is DELAYED-ACTION: administered 30-45 minutes before death
- Required sophisticated chemistry knowledge to synthesize
- No commercially available source

=== TIMELINE RECONSTRUCTION ===
7:30 PM - David enters building (security log)
8:15 PM - David accesses server room (security log)
8:25 PM - Elena enters Marcus's office (security camera)
8:40 PM - Someone accesses "removing David" files on Marcus's computer
8:45 PM - David exits building (security footage, irrefutable)
8:50 PM - Elena exits Marcus's office (security camera)
8:55 PM - Elena begins video call with Tokyo (confirmed by logs + witnesses)
9:00-9:30 PM - Estimated time of death
9:45 PM - Elena's video call ends
11:15 PM - Body discovered by night security

=== KEY PHYSICAL EVIDENCE ===
1. Victim's coffee mug: Traces of TTX compound detected
2. Two glasses on desk: One victim's (prints), one unknown (wiped clean)
3. Service entrance: Door alarm was disabled 8:55-9:10 PM (IT investigating)
4. Victim's phone: Last text at 8:22 PM to Elena: "Come to my office. Important."

=== INVESTIGATOR'S NOTES ===
- Elena has alibi during death window but was present 30 mins before
- David has no alibi at home but left building 15+ mins before death
- Poison required 30-45 min to take effect - this shifts the timeline
- If administered at ~8:30 PM, death at ~9:15 PM fits coroner's estimate
- But who administered it? Elena was there but left; David had left the building
- The wiped glass suggests someone wanted to hide their presence
- Service entrance alarm disabled during critical window - who benefits?
- Both suspects have motive, means (David's chemistry + Elena's access)
- But neither could have acted alone given the timeline... could they have?

QUESTION: Based on the evidence, who killed Marcus Chen and how?
"""

GROUND_TRUTH = {
    "solution": "Elena and David collaborated. Elena sedated/drugged Marcus at 8:30 PM. David returned via service entrance at 9:00 PM (after alarm was disabled) to ensure the dose was fatal or administer additional poison. They split the insurance/shares.",

    "key_insights": [
        "both suspects are guilty",
        "collaboration",
        "worked together",
        "both involved",
        "conspiracy",
        "partners in crime",
        "service entrance",
        "disabled alarm",
        "delayed action poison",
        "elena drugged him first",
        "david returned",
    ],

    "partial_insights": [
        "timeline doesn't work for either alone",
        "delayed poison shifts the timeline",
        "service entrance was available",
        "alarm was disabled",
        "someone else entered",
        "coordinated",
    ],

    "wrong_answers": {
        "elena_alone": "Elena has video alibi during death. Can't be solo.",
        "david_alone": "David left building before poison could be administered. Can't be solo.",
        "neither": "Both have strong evidence against them. Dismissing both ignores evidence.",
        "third_party": "No evidence of third party. Overcomplicates a solvable case.",
    },

    "single_shot_trap": "Single-shot will likely commit to one suspect and rationalize away their alibi gaps, rather than considering collaboration.",
}


def score_solution(text: str) -> tuple[float, str, list[str]]:
    """Score how well the response solved the mystery.

    Returns:
        (score, diagnosis, found_insights)
    """
    text_lower = text.lower()

    # Check for THE key insight: collaboration
    found_collaboration = any(insight in text_lower for insight in [
        "both",
        "together",
        "collaborat",
        "conspir",
        "partners",
        "working with",
        "helped each other",
        "coordinated",
        "in on it",
        "both guilty",
        "both responsible",
        "elena and david",
        "david and elena",
    ])

    # Check for supporting insights
    found_insights = []
    for insight in GROUND_TRUTH["key_insights"]:
        if insight in text_lower:
            found_insights.append(insight)

    for insight in GROUND_TRUTH["partial_insights"]:
        if insight in text_lower:
            found_insights.append(f"(partial) {insight}")

    # Check if they fell into single-suspect trap
    commits_to_elena = any(phrase in text_lower for phrase in [
        "elena is the killer",
        "elena killed",
        "elena is guilty",
        "elena did it",
        "conclusion: elena",
    ]) and not found_collaboration

    commits_to_david = any(phrase in text_lower for phrase in [
        "david is the killer",
        "david killed",
        "david is guilty",
        "david did it",
        "conclusion: david",
    ]) and not found_collaboration

    # Score
    if found_collaboration and len(found_insights) >= 3:
        return 1.0, "SOLVED - identified collaboration with evidence", found_insights
    elif found_collaboration:
        return 0.75, "CORRECT THEORY - identified collaboration, limited detail", found_insights
    elif len(found_insights) >= 4:
        return 0.5, "CLOSE - found key evidence but missed collaboration", found_insights
    elif commits_to_elena:
        return 0.2, "TRAPPED - committed to Elena alone (has alibi)", found_insights
    elif commits_to_david:
        return 0.2, "TRAPPED - committed to David alone (left building)", found_insights
    elif len(found_insights) >= 1:
        return 0.3, f"PARTIAL - some insights ({', '.join(found_insights[:3])})", found_insights
    else:
        return 0.1, "MISSED - no key insights found", found_insights


if __name__ == "__main__":
    print("Murder Mystery Case File")
    print("=" * 50)
    print()
    print("THE CASE:")
    print("  Marcus Chen, CEO, found dead in locked office")
    print("  Two suspects with strong evidence against each")
    print("  Neither has a clean timeline for acting alone")
    print()
    print("THE TRAP:")
    print("  Single-shot will commit to one suspect")
    print("  and rationalize away the timeline problems")
    print()
    print("THE SOLUTION:")
    print(f"  {GROUND_TRUTH['solution']}")
    print()
    print("KEY INSIGHTS PSS SHOULD FIND:")
    for insight in GROUND_TRUTH["key_insights"][:5]:
        print(f"  - {insight}")
