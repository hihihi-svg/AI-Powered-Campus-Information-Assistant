"""Domain entity extraction module for the RVCE Campus Information Assistant.

Extracts key entities independently of intent classification:
- Exam (KCET, COMEDK, JEE, PGCET, DCET)
- Branch (CSE, AIML, ISE, ECE, EEE, ME, CV, CD, CY, etc.)
- Admission Quota (kea_cet, comedk, management, pgcet, dcet, multiple)
- Reservation Category (GM, 1G, 2A, 2B, 3A, 3B, SCG, STG, etc.)
- Academic Year / Level (2024, 2025, 2026, 2027, 1st year, higher semester)
- Counselling Round (Round 1, Round 2, Round 3 / Extended, Mock Round)
- Recruiter Company (LinkedIn, Auxia, HPE, Cisco, IBM, Microsoft, etc.)
- CTC / Package (float in LPA and raw string)
- Data Type (previous_cutoff, cutoff, fee_structure, placement_record, schedule, rules, etc.)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

EXAM_PATTERNS = [
    (r"\b(kcet|k-?cet|kkea|kea)\b", "KCET"),
    (r"\b(comed-?k|comed)\b", "COMEDK"),
    (r"\b(jee\s*(?:mains?|advanced)?)\b", "JEE"),
    (r"\b(pgcet)\b", "PGCET"),
    (r"\b(dcet)\b", "DCET"),
]

BRANCH_PATTERNS = [
    (r"\b(computer science and engineering|computer science|cse|cs)\b", "CSE"),
    (r"\b(information science and engineering|information science|ise|is)\b", "ISE"),
    (r"\b(artificial intelligence and machine learning|artificial intelligence|aiml|ai/ml|ai & ml)\b", "AIML"),
    (r"\b(computer science and design|csd|cd)\b", "CD"),
    (r"\b(cyber security|cybersecurity|cy|cys)\b", "CY"),
    (r"\b(electronics and communication engineering|electronics and communication|ece|ec)\b", "ECE"),
    (r"\b(electrical and electronics engineering|electrical and electronics|eee|ee)\b", "EEE"),
    (r"\b(mechanical engineering|mechanical|mech|me)\b", "ME"),
    (r"\b(civil engineering|civil|cv)\b", "CV"),
    (r"\b(aerospace engineering|aerospace|ae|as)\b", "AS"),
    (r"\b(biotechnology engineering|biotechnology|biotech|bt)\b", "BT"),
    (r"\b(chemical engineering|chemical|ch)\b", "CH"),
    (r"\b(industrial engineering and management|industrial engineering|iem|im)\b", "IM"),
    (r"\b(electronics and telecommunication engineering|telecommunication|ete|te)\b", "TE"),
    (r"\b(electronics and instrumentation engineering|instrumentation|eie|it)\b", "IT"),
    (r"\b(master of computer applications|mca)\b", "MCA"),
    (r"\b(master of technology|m\.?tech|mtech)\b", "MTech"),
]

BRANCH_CANONICAL_NAMES = {
    "CSE": "Computer Science & Engineering",
    "ISE": "Information Science & Engineering",
    "AIML": "Artificial Intelligence & Machine Learning",
    "CD": "Computer Science & Design",
    "CY": "Cyber Security",
    "ECE": "Electronics & Communication Engineering",
    "EEE": "Electrical & Electronics Engineering",
    "ME": "Mechanical Engineering",
    "CV": "Civil Engineering",
    "AS": "Aerospace Engineering",
    "BT": "Biotechnology",
    "CH": "Chemical Engineering",
    "IM": "Industrial Engineering & Management",
    "TE": "Electronics & Telecommunication Engineering",
    "IT": "Electronics & Instrumentation Engineering",
    "MCA": "Master of Computer Applications",
    "MTech": "Master of Technology",
}

CATEGORY_PATTERNS = [
    (r"\b(gm|general\s+merit|general)\b", "GM"),
    (r"\b(1g|1r|1k|cat-?1|category\s+1)\b", "1G"),
    (r"\b(2a|2ag|2ar|2ak)\b", "2A"),
    (r"\b(2b|2bg|2br|2bk)\b", "2B"),
    (r"\b(3a|3ag|3ar|3ak)\b", "3A"),
    (r"\b(3b|3bg|3br|3bk)\b", "3B"),
    (r"\b(sc|scg|scr|sck)\b", "SCG"),
    (r"\b(st|stg|str|stk)\b", "STG"),
    (r"\b(open\s+dream)\b", "Open Dream"),
    (r"\b(super\s+dream)\b", "Super Dream"),
    (r"\b(dream)\b", "Dream"),
]

KNOWN_COMPANIES = [
    "LinkedIn", "Auxia", "HPE", "Cisco", "IBM", "Microsoft", "InMobi", "Palo Alto Networks",
    "Wells Fargo", "MUFG", "Telstra", "Smartsheet", "ZS Associates", "Walmart", "Deutsche Bank",
    "ShareChat", "Arcesium", "Tekion", "Eternal", "Green Light", "SAP Labs", "AMD", "CDOT",
    "Mercedes-Benz", "NatWest", "Texas Instruments", "NXP Semiconductors", "Ather", "Airbus",
    "Aveva", "British Telecom", "Applogic",
]


def extract_campus_entities(query: str, intent: Optional[str] = None) -> Dict[str, Any]:
    """Extract domain entities independently from query text.

    Returns a structured dictionary containing:
      - exam (KCET, COMEDK, JEE, PGCET, DCET, or None)
      - branch (CSE, ISE, AIML, etc., or None)
      - branch_name (Full name or None)
      - admission_quota (kea_cet, comedk, management, pgcet, dcet, multiple, or None)
      - category (GM, 2A, 3B, SCG, etc., or None)
      - academic_year (e.g. '2024', '2025', '2026', '2027', '1st year', or None)
      - year (numeric year or None for cutoff compatibility)
      - counselling_round (e.g. 'Round 1', 'Round 2', or None)
      - company (Company name or None)
      - ctc (numeric float in LPA or None)
      - ctc_raw (raw string representation or None)
      - data_type (e.g. 'previous_cutoff', 'cutoff', 'fee_structure', 'placement_record', etc.)
      - is_comparison (bool)
    """
    if not query or not query.strip():
        return {
            "exam": None,
            "branch": None,
            "branch_name": None,
            "admission_quota": None,
            "category": None,
            "academic_year": None,
            "year": None,
            "counselling_round": None,
            "company": None,
            "ctc": None,
            "ctc_raw": None,
            "data_type": None,
            "is_comparison": False,
        }

    raw_q = query.strip()
    q = raw_q.lower()

    # 1. Exam detection
    exam = None
    for pattern, code in EXAM_PATTERNS:
        if re.search(pattern, q):
            exam = code
            break

    # 2. Branch detection
    branch = None
    for pattern, code in BRANCH_PATTERNS:
        # Avoid matching 'is' as Information Science when it's just the English verb
        if code == "ISE" and re.search(r"\bis\b", q):
            if not (re.search(r"\bise\b", q) or "information science" in q or re.search(r"\bis\s+(?:branch|dept|department)\b", q)):
                continue
        if re.search(pattern, q):
            branch = code
            break

    branch_name = BRANCH_CANONICAL_NAMES.get(branch) if branch else None

    # 3. Admission Quota detection
    has_kea = bool(re.search(r"\b(kcet|k-?cet|kea|kkea|govt(?:\s+quota)?|government(?:\s+quota)?)\b", q) or re.search(r"\bfees?\s+(?:through|for|under|via)\s+cet\b", q))
    has_comedk = bool(re.search(r"\b(comed-?k|comed)\b", q))
    has_mgmt = bool(re.search(r"\b(management(?:\s+quota)?|mgmt(?:\s+quota)?)\b", q))
    has_pgcet = bool(re.search(r"\b(pgcet|m\.?tech|mtech|mca)\b", q))
    has_dcet = bool(re.search(r"\b(dcet|lateral\s+entry)\b", q))

    quota_matches = [q_code for q_code, flag in [("kea_cet", has_kea), ("comedk", has_comedk), ("management", has_mgmt), ("pgcet", has_pgcet), ("dcet", has_dcet)] if flag]
    is_comparison = bool(re.search(r"\b(compare|comparison|difference\s+between|vs|versus)\b", q) or len(quota_matches) >= 2)

    admission_quota = None
    if is_comparison or len(quota_matches) >= 2:
        admission_quota = "multiple"
    elif quota_matches:
        admission_quota = quota_matches[0]

    # 4. Category detection
    category = None
    for pattern, code in CATEGORY_PATTERNS:
        if re.search(pattern, q):
            category = code
            break

    # 5. Academic Year & Year detection
    academic_year = None
    year_num = None
    year_match = re.search(r"\b(202[0-9](?:-2[0-9])?)\b", q)
    if year_match:
        academic_year = year_match.group(1)
        try:
            year_num = int(academic_year[:4])
        except ValueError:
            pass
    elif re.search(r"\b(1st\s+year|first\s+year|freshers?)\b", q):
        academic_year = "1st year"
    elif re.search(r"\b(2nd\s+year|second\s+year|lateral\s+entry)\b", q):
        academic_year = "2nd year"
    elif re.search(r"\b(higher\s+semester|3rd\s+sem|5th\s+sem|7th\s+sem)\b", q):
        academic_year = "higher semester"

    # 6. Counselling Round detection
    counselling_round = None
    if re.search(r"\bround\s*1\b|\br1\b", q):
        counselling_round = "Round 1"
    elif re.search(r"\bround\s*2\b|\br2\b", q):
        counselling_round = "Round 2"
    elif re.search(r"\b(?:round\s*3|extended(?:\s+round)?|r3)\b", q):
        counselling_round = "Round 3 / Extended"
    elif re.search(r"\bmock\s*(?:round)?\b", q):
        counselling_round = "Mock Round"

    # 7. Recruiter Company detection
    company = None
    for comp_name in KNOWN_COMPANIES:
        if re.search(rf"\b{re.escape(comp_name.lower())}\b", q):
            company = comp_name
            break

    # 8. CTC / Package detection
    ctc = None
    ctc_raw = None
    ctc_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|cr)", q)
    if ctc_match:
        try:
            ctc = float(ctc_match.group(1))
            ctc_raw = f"{ctc:.2f} LPA"
        except ValueError:
            pass

    # 9. Data Type detection
    data_type = None
    is_previous = bool(re.search(r"\b(previous|prior|last\s+year|past\s+year|historical|earlier)\b", q))
    has_cutoff = bool(re.search(r"\b(cutoff|cut-off|closing\s+rank|cut\s+off|last\s+rank|closing)\b", q))
    has_fees = bool(re.search(r"\b(fees?|tuition|cost|charges?|challan|payment|penalty|fine)\b", q))
    has_placement = bool(re.search(r"\b(placement|placements|package|salary|recruiter|internship|hiring|hired|offers?|ppo)\b", q))
    has_schedule = bool(re.search(r"\b(timetable|calendar|schedule|exam\s+dates?|timing|dates?)\b", q))
    has_procedure = bool(re.search(r"\b(procedure|process|steps|documents?|verification|eligibility|how\s+to\s+apply)\b", q))

    if has_cutoff:
        data_type = "previous_cutoff" if is_previous else "cutoff"
    elif has_fees:
        data_type = "fee_structure"
    elif has_placement:
        data_type = "placement_record"
    elif has_schedule:
        data_type = "schedule"
    elif has_procedure:
        data_type = "procedure"

    return {
        "exam": exam,
        "branch": branch,
        "branch_name": branch_name,
        "admission_quota": admission_quota,
        "category": category,
        "academic_year": academic_year,
        "year": year_num,
        "counselling_round": counselling_round,
        "company": company,
        "ctc": ctc,
        "ctc_raw": ctc_raw,
        "data_type": data_type,
        "is_comparison": is_comparison,
    }
