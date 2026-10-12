# Reflection (~100 words)

**What went wrong?** The obvious "official" alert dataset silently stopped on 7 Sep 2026 and records a different
(raion-level) target. My first model mostly learned base rates; a "smarter" sub-hour feature idea gave zero gain;
my Telegram regexes missed Ukrainian declensions ("балістичного", "Миколаєва"), so ballistic threats were nearly invisible.

**How did I adjust?** Switched to the daily-updated oblast dataset, wrote an automated leakage test, compared every
idea against baselines and a validation month, inspected regex matches on real posts and fixed them, and added
weekly walk-forward refits after seeing the model age.

**Why is the final version better?** It uses the latest data, is provably leakage-free, calibrated, beats
climatology on every horizon, and Telegram's contribution is measured, not assumed.
