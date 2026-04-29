"""
Heuristic classifier for radiology prior study relevance.

Extracts body regions and modalities from study descriptions,
then predicts relevance based on anatomical overlap.
"""

import re
from typing import Optional

# Each pattern is (region_name, compiled_regex)
# Order matters: more specific patterns should come first so they get priority

_BODY_REGION_PATTERNS: list[tuple[str, re.Pattern]] = []
_MODALITY_PATTERNS: list[tuple[str, re.Pattern]] = []

_RAW_BODY_REGIONS = {
    # Cardiac / vascular heart
    "cardiac": [
        r"cardiac", r"heart", r"coronar", r"myocard", r"myo\s*perf",
        r"aort(?:ic|a)\b", r"echocard", r"\becho\b", r"\btte\b", r"\btee\b",
        r"transthorac", r"transesoph", r"stress\s*test", r"definity",
        r"chemo\s*tte", r"doppl.*complete", r"\bffr\b",
    ],
    # Chest / lung / thorax (NOT cardiac, NOT thoracic spine)
    "chest": [
        r"\bchest\b", r"\blung\b", r"thorax", r"pulmon", r"bronch",
        r"\bribs?\b", r"thoracent", r"pul\s*perfus",
    ],
    # Brain / head
    "brain": [
        r"\bbrain\b", r"\bhead\b(?!\s*(and|&)\s*neck)", r"cranial",
        r"intracranial", r"\bskull\b", r"transcranial", r"\beeg\b",
    ],
    # Face / sinuses
    "face": [
        r"facial", r"\bface\b", r"sinus(?:es)?", r"\borbit",
        r"mandib", r"maxfac", r"maxil", r"nasal", r"\bjaw\b",
        r"temporal\s*bone", r"\bear\b",
    ],
    # Neck / thyroid
    "neck": [
        r"\bneck\b", r"thyroid", r"carotid", r"parathyroid",
        r"soft\s*tissue.*neck", r"laryn",
    ],
    # Breast / mammography
    "breast": [
        r"breast", r"mammo", r"\bmamm\b", r"\bmam\s", r"\bmam$",
        r"mammogra", r"seed\s*local", r"tomo(?:synth)?",
        r"r2\s*mammo", r"screen.*bilat", r"bilat.*screen",
        r"screener.*cad", r"digital\s*screen", r"combohd",
        r"standard\s*screen",
    ],
    # Abdomen
    "abdomen": [
        r"abdomen", r"abdominal", r"\babd\b", r"abd[_/]",
        r"liver", r"hepat", r"pancrea", r"gallbladder",
        r"spleen", r"splenic", r"adrenal", r"mesentery",
        r"retroperit", r"enterogr", r"paracentesis",
        r"cholangio", r"barium", r"esophag", r"swallow(?:ing)?",
        r"\bugi\b", r"upper\s*gi\b", r"small\s*bowel",
    ],
    # Pelvis
    "pelvis": [
        r"pelvi", r"\bpelv\b", r"bladder", r"uterus", r"uterine",
        r"ovari", r"prostat", r"rectal", r"rectum",
        r"endovag", r"transvag", r"vaginal",
    ],
    # Spine - cervical
    "spine_cervical": [
        r"cervical\s*spine", r"\bc[\-\s]?spine\b",
    ],
    # Spine - thoracic
    "spine_thoracic": [
        r"thoracic\s*spine", r"\bt[\-\s]?spine\b",
    ],
    # Spine - lumbar
    "spine_lumbar": [
        r"lumbar", r"\bl[\-\s]?spine\b",
    ],
    # Spine - general
    "spine": [
        r"\bspine\b", r"\bspinal\b", r"vertebr", r"scoliosis",
    ],
    # Knee
    "knee": [r"\bknee\b", r"patell"],
    # Shoulder
    "shoulder": [r"shoulder", r"rotator"],
    # Hip
    "hip": [r"\bhip\b", r"femur", r"femoral", r"acetabul"],
    # Ankle / foot
    "ankle_foot": [r"\bankle\b", r"\bfoot\b", r"\bfeet\b", r"calcaneus", r"\btoe\b"],
    # Wrist / hand
    "wrist_hand": [r"\bwrist\b", r"\bhand\b", r"\bfinger\b", r"carpal", r"scaphoid"],
    # Elbow
    "elbow": [r"\belbow\b"],
    # Kidney / renal (overlaps with abdomen, but useful as its own region)
    "renal": [r"renal", r"kidney", r"nephro", r"ureter", r"urolog"],
    # Vascular / angiography
    "vascular": [
        r"angiogra", r"angiogram", r"\bangio\b",
        r"venous", r"\bdvt\b", r"thrombos",
        r"arter(?:y|ial|ies)", r"\bvas\b", r"vascular",
    ],
    # Bone density
    "bone_density": [r"bone\s*dens", r"\bdexa\b", r"\bdxa\b"],
    # Whole body / PET
    "whole_body": [
        r"whole\s*body", r"skull.*thigh", r"vertex.*toe",
        r"total\s*body", r"skull\s*base.*thigh",
    ],
    # Bone scan (whole body nuclear)
    "bone_scan": [r"\bbone\s*scan\b"],
}

_RAW_MODALITIES = {
    "CT": [r"\bCT\b"],
    "MRI": [r"\bMRI\b", r"\bMR\b(?!\s*angio)"],
    "XR": [r"\bXR\b", r"x[\-\s]?ray", r"radiograph", r"\bCR\b"],
    "US": [r"\bUS\b", r"ultra", r"sonogra", r"doppler", r"endovag"],
    "NM": [r"\bNM\b", r"nuclear", r"\bSPECT\b", r"scintig"],
    "PET": [r"\bPET\b"],
    "MAM": [r"mammo", r"\btomo\b", r"tomosyn"],
    "FLUORO": [r"fluoro", r"barium", r"swallow"],
    "DEXA": [r"\bDEXA\b", r"\bDXA\b", r"bone\s*dens"],
    "ECHO": [r"\becho\b", r"\btte\b", r"\btee\b"],
}

def _compile_patterns():
    for region, raw_list in _RAW_BODY_REGIONS.items():
        combined = "|".join(f"(?:{p})" for p in raw_list)
        _BODY_REGION_PATTERNS.append((region, re.compile(combined, re.IGNORECASE)))
    for mod, raw_list in _RAW_MODALITIES.items():
        combined = "|".join(f"(?:{p})" for p in raw_list)
        _MODALITY_PATTERNS.append((mod, re.compile(combined, re.IGNORECASE)))

_compile_patterns()


def extract_regions(desc: str) -> set[str]:
    regions = set()
    for region, pattern in _BODY_REGION_PATTERNS:
        if pattern.search(desc):
            regions.add(region)

    # PET/CT whole-body descriptions cover most regions
    if "whole_body" in regions:
        regions.update(["chest", "abdomen", "pelvis", "brain", "spine"])

    return regions


def extract_modality(desc: str) -> Optional[str]:
    for mod, pattern in _MODALITY_PATTERNS:
        if pattern.search(desc):
            return mod
    return None


# Regions that are clinically related and often relevant to each other
_RELATED_REGIONS = {
    ("abdomen", "pelvis"),
    ("abdomen", "renal"),
    ("pelvis", "renal"),
    ("spine", "spine_cervical"),
    ("spine", "spine_thoracic"),
    ("spine", "spine_lumbar"),
    ("neck", "spine_cervical"),
}

def _regions_related(r1: str, r2: str) -> bool:
    return (r1, r2) in _RELATED_REGIONS or (r2, r1) in _RELATED_REGIONS


def _normalize_for_comparison(desc: str) -> str:
    """Strip contrast info so 'CT CHEST WITH CONTRAST' == 'CT CHEST WITHOUT CNTRST'."""
    desc = desc.strip().lower()
    desc = re.sub(
        r'\b(without|with|w/o|wo|w/)\s+(and\s+without\s+)?(iv\s+)?(contrast|cntrst|con)\b',
        '', desc,
    )
    desc = re.sub(r'\b(contrast|cntrst)\b', '', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()
    return desc


def heuristic_predict(
    current_desc: str,
    prior_desc: str,
    current_date: str = "",
    prior_date: str = "",
) -> tuple[bool, float]:
    """
    Predict whether a prior study is relevant to the current study.

    Returns (prediction, confidence) where confidence is 0.0-1.0.
    High confidence means the heuristic is sure; low confidence means
    it should be deferred to the LLM.
    """
    # Same-description shortcut: identical (modulo contrast) is always relevant
    if _normalize_for_comparison(current_desc) == _normalize_for_comparison(prior_desc):
        return True, 0.95

    current_regions = extract_regions(current_desc)
    prior_regions = extract_regions(prior_desc)
    current_mod = extract_modality(current_desc)
    prior_mod = extract_modality(prior_desc)

    # Both sides unrecognized — low confidence, defer to LLM
    if not current_regions and not prior_regions:
        return False, 0.1

    # One side unrecognized — probably not related, moderate-high confidence
    if not current_regions or not prior_regions:
        return False, 0.8

    # Direct region overlap = very likely relevant
    direct_overlap = current_regions & prior_regions
    if direct_overlap:
        return True, 0.9

    # Check related regions (abdomen/pelvis, spine variants, neck/c-spine, etc.)
    for cr in current_regions:
        for pr in prior_regions:
            if _regions_related(cr, pr):
                return True, 0.76

    # No region overlap at all = very likely NOT relevant
    return False, 0.85
