"""Which tools are worth showing the router, and in what order.

A BM25 ranker over each tool's own published text -- its name, its description,
and its arguments' names and descriptions -- with one small, catalogue-blind
synonym table so a request phrased in a clinician's words reaches a tool
described in an engineer's. Nothing here knows that any particular tool exists.

**Why not a cross-encoder.** Upstream ranked with
`sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`,
which drags ~2 GB of torch wheels into the image to score a query against 23
short strings. Both rankers were measured on 30 realistic requests against the
real 16-tool catalogue; the numbers and the method are in README.md. The short
version is that they agree, and the one that costs nothing was kept.

**A ranker that cannot rank must not narrow.** Upstream caught every exception
here and returned `scripts[:k]` -- the first three tools in file order -- while
the router prompt went on instructing the model to "choose ONLY from the
candidate list". That is the defect this module is shaped around, in two
places: an internal failure raises (`RankingError`), and a ranking with no
signal at all hands over the WHOLE catalogue rather than an arbitrary prefix.
Narrowing is something a ranker earns.
"""

import math
import re

from .errors import RankingError

# BM25's usual constants. `b` is turned down from the standard 0.75 because
# these documents are 20-200 words rather than web pages, and a tool that
# happens to document its arguments thoroughly should not be penalised for it.
K1 = 1.2
B = 0.4

# How much each field of a tool's published text counts. The name is the
# strongest signal a catalogue has and the shortest, so it is weighted up; the
# argument text is the weakest, and there is a lot of it.
WEIGHT_NAME = 4
WEIGHT_DESCRIPTION = 3
WEIGHT_ARGUMENTS = 1

# A query term matched through a synonym rather than literally is worth less.
SYNONYM_DISCOUNT = 0.55

# Words that carry no signal in a one-sentence request. Deliberately short: a
# long stop list starts removing terms that do discriminate ("scan", "image").
STOPWORDS = frozenset("""
a an the of on in to for from with and or my our i we me it its this that these
those is are be am was were do does did please can could would should want need
have has get got make made run use using take takes give gives at by as into
onto over under between each all any some more most new old
""".split())

# Equivalence classes of vocabulary, NOT of tools. Every group below is a set of
# words that mean the same thing in imaging prose; none of them names a tool, a
# structure or a body part, and adding a tool to the catalogue never touches
# this table. That is the line the upstream manifest crossed -- it wrote the
# anatomy down beside the tool list, and the two drifted.
SYNONYMS = (
    frozenset(["register", "registration", "registered", "registering",
               "align", "aligned", "aligning", "alignment", "superimpose",
               "superimposition", "superimposed", "overlay", "match", "matched"]),
    frozenset(["segment", "segmentation", "segmented", "segmenting", "mask",
               "masks", "label", "labels", "labelled", "labeled", "labelling",
               "labeling", "delineate", "delineation", "contour", "contours"]),
    frozenset(["landmark", "landmarks", "point", "points", "fiducial",
               "fiducials", "cephalometric", "marker", "markers", "annotate",
               "annotation"]),
    frozenset(["orient", "orientation", "oriented", "orienting", "reorient",
               "standardize", "standardise", "standardized", "standardised",
               "frame", "pose"]),
    frozenset(["crop", "cropped", "cropping", "trim", "trimmed", "roi",
               "region", "bounding", "box"]),
    frozenset(["tooth", "teeth", "dental", "dentition", "crown", "crowns",
               "molar", "molars", "incisor", "incisors", "canine", "canines"]),
    frozenset(["scan", "scans", "volume", "volumes", "image", "images",
               "cbct", "ct", "dicom", "nifti", "nii", "nrrd", "gipl"]),
    frozenset(["mesh", "meshes", "surface", "surfaces", "stl", "vtk", "vtp",
               "obj", "ios", "intraoral", "oral"]),
    frozenset(["transform", "transforms", "matrix", "matrices", "tfm"]),
    frozenset(["predict", "prediction", "predicted", "forecast", "estimate",
               "estimated", "estimation"]),
    frozenset(["classify", "classification", "classified", "class", "classes",
               "category", "categories", "grade", "grading", "severity"]),
    frozenset(["note", "notes", "report", "reports", "text", "clinical",
               "narrative", "free"]),
    frozenset(["timepoint", "timepoints", "followup", "follow", "baseline",
               "longitudinal", "before", "after", "t1", "t2", "pre", "post"]),
    frozenset(["extract", "extraction", "extracted", "pull", "read", "parse"]),
    frozenset(["apply", "applied", "applying"]),
    frozenset(["skull", "craniofacial", "bone", "bones", "skeletal", "jaw",
               "jaws", "mandible", "maxilla", "condyle", "airway"]),
    frozenset(["surgery", "surgical", "surgeon", "operation", "operative",
               "movement", "movements", "displacement"]),
    frozenset(["impacted", "impaction", "eruption", "unerupted"]),
    frozenset(["patient", "patients", "case", "cases", "cohort", "batch",
               "series", "folder", "dataset"]),
)

_TOKEN = re.compile(r"[a-z0-9]+")
# Split before an upper-case letter that STARTS a word, from either a lower-case
# run or an acronym: `AutoCrop3D` -> `Auto Crop3D`, `IOSCBCTReg` -> `IOSCBCT Reg`.
# Splitting before every capital would cut `Crop3D` into `Crop3 D` and lose the
# `3D` a request is most likely to say.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])|(?<=[A-Z])(?=[A-Z][a-z])")
# A word run followed by a digit run, split once: `crop3d` -> `crop`, `3d`.
_ALPHANUM = re.compile(r"^([a-z]{2,})([0-9][a-z0-9]*)$")


def _expansions():
    """`{word: the whole class it belongs to}`, built once from SYNONYMS."""
    table = {}
    for group in SYNONYMS:
        for word in group:
            table.setdefault(word, set()).update(group)
    return table


EXPANSIONS = _expansions()


def tokenize(text: str):
    """Words of `text`, lowercased, camelCase and separators split, stops out.

    A tool NAME is the hardest part: `Batch_Dental_Seg`, `AutoCrop3D` and
    `AREG_IOSCBCT` all have to become words a request can match. Underscores
    split, camelCase splits, and digits stay attached to the word they were
    written against (`AutoCrop3D` -> `auto`, `crop3d`) as well as separately,
    so `3D` in a request still matches.
    """
    spaced = _CAMEL.sub(" ", str(text)).replace("_", " ").replace("-", " ")
    tokens = []
    for token in _TOKEN.findall(spaced.lower()):
        if token in STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
        # `crop3d` also yields `crop` and `3d`, so a request saying either
        # matches. Cheap, and it is the only splitting rule that needs a
        # special case.
        parts = _ALPHANUM.match(token)
        if parts:
            tokens.append(parts.group(1))
            if len(parts.group(2)) >= 2:
                tokens.append(parts.group(2))
    return tokens


def document_terms(tool):
    """`{term: weight}` for one tool, from what the catalogue publishes about it.

    Every field is used whole. Upstream truncated the description to 140
    characters and the tag list to 8 before the model ever saw it
    (`Agent_CLI.py:29,31`), mid-sentence, which is a strange economy on a
    prompt that then sends a 23-tool manifest's worth of parameter definitions.
    """
    terms = {}

    def add(text, weight):
        for token in tokenize(text):
            terms[token] = terms.get(token, 0.0) + weight

    add(tool.get("name", ""), WEIGHT_NAME)
    add(tool.get("description", ""), WEIGHT_DESCRIPTION)
    for argument, spec in (tool.get("arguments") or {}).items():
        add(argument, WEIGHT_ARGUMENTS)
        add(spec.get("description", ""), WEIGHT_ARGUMENTS)
        add(spec.get("label", ""), WEIGHT_ARGUMENTS)
    return terms


def query_terms(text: str):
    """`{term: weight}` for a request, literal terms plus discounted synonyms."""
    terms = {}
    for token in tokenize(text):
        terms[token] = max(terms.get(token, 0.0), 1.0)
        for synonym in EXPANSIONS.get(token, ()):
            if synonym == token:
                continue
            terms[synonym] = max(terms.get(synonym, 0.0), SYNONYM_DISCOUNT)
    return terms


def score_tools(tools, request: str):
    """`[(tool, score)]` in descending score, ties broken by catalogue order.

    Plain BM25: an IDF over the catalogue times a saturating term frequency,
    normalised by document length. The catalogue is a dozen or two short
    documents, so this is microseconds and needs no index.
    """
    if not tools:
        raise RankingError("There are no tools to rank.")

    documents = [document_terms(tool) for tool in tools]
    lengths = [sum(document.values()) for document in documents]
    total = len(documents)
    average = (sum(lengths) / total) or 1.0

    frequency = {}
    for document in documents:
        for term in document:
            frequency[term] = frequency.get(term, 0) + 1

    query = query_terms(request)
    scored = []
    for index, (tool, document) in enumerate(zip(tools, documents)):
        score = 0.0
        for term, weight in query.items():
            occurrences = document.get(term, 0.0)
            if not occurrences:
                continue
            # The `+ 1` inside the log keeps every IDF positive: a term every
            # tool mentions should add nothing, never subtract.
            idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
            norm = K1 * (1 - B + B * lengths[index] / average)
            score += weight * idf * occurrences * (K1 + 1) / (occurrences + norm)
        scored.append((score, index, tool))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(tool, score) for score, _index, tool in scored]


def select_candidates(tools, request: str, limit: int):
    """`(candidates, scores, narrowed)`: which tools the router is shown.

    `limit <= 0`, or a limit covering the catalogue, means every tool. So does a
    ranking with no signal: if the request shares nothing with any tool's
    published text, every score is zero and the order is the catalogue's own,
    so taking a prefix of it would be exactly upstream's "first k tools"
    dressed up as a ranking. `narrowed` says which of the two happened, and it
    is recorded in the run report.
    """
    ranked = score_tools(tools, request)
    scores = {tool["name"]: round(score, 4) for tool, score in ranked}

    if limit <= 0 or limit >= len(ranked):
        return [tool for tool, _score in ranked], scores, False
    if not any(score > 0 for _tool, score in ranked):
        return [tool for tool, _score in ranked], scores, False
    return [tool for tool, _score in ranked[:limit]], scores, True
