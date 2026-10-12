"""Oblast adjacency and Ukrainian name patterns (for Telegram text matching)."""
import re

_EDGES = {
    "Vinnytska oblast": ["Zhytomyrska oblast", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast",
                         "Odeska oblast", "Chernivetska oblast", "Khmelnytska oblast"],
    "Volynska oblast": ["Rivnenska oblast", "Lvivska oblast"],
    "Dnipropetrovska oblast": ["Poltavska oblast", "Kharkivska oblast", "Donetska oblast", "Zaporizka oblast",
                               "Khersonska oblast", "Mykolaivska oblast", "Kirovohradska oblast"],
    "Donetska oblast": ["Kharkivska oblast", "Zaporizka oblast"],
    "Zhytomyrska oblast": ["Rivnenska oblast", "Khmelnytska oblast", "Kyivska oblast"],
    "Zakarpatska oblast": ["Lvivska oblast", "Ivano-Frankivska oblast"],
    "Zaporizka oblast": ["Khersonska oblast"],
    "Ivano-Frankivska oblast": ["Lvivska oblast", "Chernivetska oblast", "Ternopilska oblast"],
    "Kyivska oblast": ["Cherkaska oblast", "Poltavska oblast", "Chernihivska oblast", "Kyiv City"],
    "Kirovohradska oblast": ["Cherkaska oblast", "Poltavska oblast", "Mykolaivska oblast", "Odeska oblast"],
    "Lvivska oblast": ["Rivnenska oblast", "Ternopilska oblast"],
    "Mykolaivska oblast": ["Odeska oblast", "Khersonska oblast"],
    "Poltavska oblast": ["Chernihivska oblast", "Sumska oblast", "Kharkivska oblast", "Cherkaska oblast"],
    "Rivnenska oblast": ["Ternopilska oblast", "Khmelnytska oblast"],
    "Sumska oblast": ["Chernihivska oblast", "Kharkivska oblast"],
    "Ternopilska oblast": ["Khmelnytska oblast", "Chernivetska oblast"],
    "Khmelnytska oblast": ["Chernivetska oblast"],
}


def neighbours() -> dict[str, set[str]]:
    nb: dict[str, set[str]] = {}
    for a, bs in _EDGES.items():
        for b in bs:
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
    return nb


# Ukrainian stems used by the Air Force channel (oblast names, "-щина" forms,
# oblast centres). Lower-case regexes.
REGION_PATTERNS = {
    "Vinnytska oblast": r"вінни|жмерин|могилів-поділ|тульчин|гайсин|калинів",
    "Volynska oblast": r"волин|луцьк|ковел",
    "Dnipropetrovska oblast": r"дніпр|кривий ріг|кривого рогу|кривому розі|криворіж|павлоград|нікопол|кам.янськ|марганц|синельников",
    "Donetska oblast": r"донеч|донецьк|краматорськ|слов.янськ|покровськ|костянтинівк|дружківк",
    "Zhytomyrska oblast": r"житомир|бердичів|бердичев|коростен|овруч|звягел",
    "Zakarpatska oblast": r"закарпат|ужгород",
    "Zaporizka oblast": r"запоріж|запорізьк|вільнянськ|енергодар",
    "Ivano-Frankivska oblast": r"івано-франків|прикарпат",
    "Kyivska oblast": r"київщин|київськ|борисп|бровар|біла церква|білої церкви|фастів|обухів|вишгород|ірпін|буча|бучі|вишневе|боярк|славутич|васильк|макарів|бородянк|переяслав",
    "Kyiv City": r"\b(?:київ|києв)(?!щин|ськ)|дарниц|троєщин|оболон|святошин|голосіїв|поділ(?:і|ьськ)",
    "Kirovohradska oblast": r"кіровоград|кропивницьк|олександрі|світловодськ|знам.янк",
    "Lvivska oblast": r"львів|стрий|дрогоб|червоноград|шептицьк",
    "Mykolaivska oblast": r"миколаїв|миколає|вознесенськ|первомайськ|очаків|южноукраїнськ",
    "Odeska oblast": r"одес|одещ|ізмаїл|чорноморськ|південне|білгород-дністров|затоц",
    "Poltavska oblast": r"полтав|кременчу|миргород|лубн|горішні плавні",
    "Rivnenska oblast": r"рівне|рівнен|рівного|дубн|сарн|вараш",
    "Sumska oblast": r"\bсум(?:щин|и\b|ськ|ах\b|\b)|шостк|конотоп|охтир|ромн|глухів|тростян|лебедин",
    "Ternopilska oblast": r"тернопіл",
    "Kharkivska oblast": r"харків|харков|ізюм|куп.янськ|лозов|чугуїв|богодухів|балакліє|балаклі",
    "Khersonska oblast": r"херсон|берислав",
    "Khmelnytska oblast": r"хмельни|шепетівк|старокостянтинів|кам.янець-поділ|нетішин",
    "Cherkaska oblast": r"черкас|черкащ|умань|умані|чигирин|сміл|золотонош|канев|канів",
    "Chernivetska oblast": r"чернівец|чернівц|буковин",
    "Chernihivska oblast": r"чернігів|чернігов|ніжин|прилук|новгород-сівер|мена\b|бахмач|корюків",
}

THREAT_PATTERNS = {
    "drone": r"бпла|шахед|дрон",
    "ballistic": r"балістик|балістичн",
    "missile": r"ракет",
    "kab": r"\bкаб",
    "strategic": r"міг-31|ту-95|ту-22|стратегічн",
    "all_clear": r"відбій",
    "nationwide": r"вся україна|всій території|по всій країні",
}

REGION_RE = {k: re.compile(v) for k, v in REGION_PATTERNS.items()}
THREAT_RE = {k: re.compile(v) for k, v in THREAT_PATTERNS.items()}
