# Reflection (~100 words)

**What went wrong?** The obvious "official" alert dataset silently stopped on 7 Sep 2026 and records a different
(raion-level) target. My first model mostly learned base rates; "smarter" sub-hour features gave no measurable gain;
Telegram regexes missed Ukrainian declensions ("балістичного", "Миколаєва"), hiding most ballistic threats.

**How did I adjust?** Switched to the daily-updated oblast dataset, wrote an automated leakage test, judged every
idea against baselines on a validation month, fixed regexes by reading real matches, added weekly refits after
seeing the model age, and bootstrap CIs to separate real gains from noise.

**Why is the final version better?** Latest data, provably leakage-free, calibrated, beats climatology everywhere;
Telegram's gain is proven, not assumed.
