#!/usr/bin/env python3
"""
Generate synthetic eICR training data for fine-tuning Gemma 4 E4B as an
edge clinical NLP engine (ClinIQ).

Produces a JSONL dataset where each row is a conversation:
  user: structured text summary of eICR (matching server extract_key_data())
  assistant: compact entity extraction JSON with codes + case summary

Replaces the cloud NLP pipeline (Comprehend Medical + IMO) with a fine-tuned
model that does entity extraction, ontology mapping, and case summarization.
"""

import json
import random
import uuid
from datetime import datetime, timedelta

# Reportable conditions with SNOMED, ICD-10, trigger codes
CONDITIONS = [
    {
        "name": "COVID-19",
        "snomed": "840539006",
        "icd10": "U07.1",
        "lab_loinc": "94500-6",
        "lab_name": "SARS-CoV-2 RNA NAA+probe Ql (Resp)",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Nasopharynx",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "2599543", "name": "nirmatrelvir 150 MG / ritonavir 100 MG"},
        ],
    },
    {
        "name": "Hepatitis A",
        "snomed": "40468003",
        "icd10": "B15.9",
        "lab_loinc": "13950-1",
        "lab_name": "Hepatitis A virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Hepatitis C",
        "snomed": "50711007",
        "icd10": "B18.2",
        "lab_loinc": "11259-9",
        "lab_name": "Hepatitis C virus Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "1940261", "name": "sofosbuvir 400 MG / velpatasvir 100 MG"},
        ],
    },
    {
        "name": "Pertussis",
        "snomed": "27836007",
        "icd10": "A37.90",
        "lab_loinc": "43913-3",
        "lab_name": "Bordetella pertussis DNA [Presence] in Nasopharynx by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Nasopharynx",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197650", "name": "azithromycin 250 MG Oral Tablet"},
        ],
    },
    {
        "name": "Measles",
        "snomed": "14189004",
        "icd10": "B05.9",
        "lab_loinc": "35275-7",
        "lab_name": "Measles virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Salmonella infection",
        "snomed": "302231008",
        "icd10": "A02.0",
        "lab_loinc": "625-4",
        "lab_name": "Bacteria identified in Stool by Culture",
        "lab_result_code": "27268008",
        "lab_result_display": "Salmonella",
        "specimen": "Stool",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197511", "name": "ciprofloxacin 500 MG Oral Tablet"},
        ],
    },
    {
        "name": "Tuberculosis",
        "snomed": "56717001",
        "icd10": "A15.0",
        "lab_loinc": "38379-4",
        "lab_name": "Mycobacterium tuberculosis complex DNA [Presence] in Sputum by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Nasopharynx",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197622", "name": "isoniazid 300 MG Oral Tablet"},
            {"rxnorm": "199279", "name": "rifampin 300 MG Oral Capsule"},
        ],
    },
    {
        "name": "Chlamydia trachomatis infection",
        "snomed": "240589008",
        "icd10": "A56.0",
        "lab_loinc": "21613-5",
        "lab_name": "Chlamydia trachomatis DNA [Presence] in Specimen by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Blood",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197650", "name": "azithromycin 250 MG Oral Tablet"},
        ],
    },
    {
        "name": "Gonorrhea",
        "snomed": "15628003",
        "icd10": "A54.9",
        "lab_loinc": "21415-5",
        "lab_name": "Neisseria gonorrhoeae DNA [Presence] in Urethra by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Blood",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "1665021", "name": "ceftriaxone 500 MG Injection"},
        ],
    },
    {
        "name": "Mpox",
        "snomed": "359814004",
        "icd10": "B04",
        "lab_loinc": "100434-0",
        "lab_name": "Monkeypox virus DNA [Presence] in Specimen by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Blood",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "2469470", "name": "tecovirimat 200 MG Oral Capsule"},
        ],
    },
    # --- Tier 2: STIs & bloodborne ---
    {
        "name": "Syphilis",
        "snomed": "76272004",
        "icd10": "A53.9",
        "lab_loinc": "20507-0",
        "lab_name": "Treponema pallidum Ab [Presence] in Serum by Immunoassay",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "105220", "name": "penicillin G benzathine 2400000 UNT/injection"},
        ],
    },
    {
        "name": "HIV infection",
        "snomed": "86406008",
        "icd10": "B20",
        "lab_loinc": "75622-1",
        "lab_name": "HIV 1 and 2 Ag+Ab [Presence] in Serum by Immunoassay",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "1999563", "name": "bictegravir 50 MG / emtricitabine 200 MG / tenofovir alafenamide 25 MG"},
        ],
    },
    {
        "name": "Influenza A virus infection (novel)",
        "snomed": "442696006",
        "icd10": "J09.X2",
        "lab_loinc": "80382-5",
        "lab_name": "Influenza virus A RNA [Presence] in Nasopharynx by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Nasopharynx",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "284635", "name": "oseltamivir 75 MG Oral Capsule"},
        ],
    },
    {
        "name": "Respiratory syncytial virus infection",
        "snomed": "55735004",
        "icd10": "J12.1",
        "lab_loinc": "92131-2",
        "lab_name": "Respiratory syncytial virus RNA [Presence] in Respiratory specimen by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Nasopharynx",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Legionellosis",
        "snomed": "5765008",
        "icd10": "A48.1",
        "lab_loinc": "32781-7",
        "lab_name": "Legionella pneumophila Ag [Presence] in Urine by Immunoassay",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197650", "name": "azithromycin 250 MG Oral Tablet"},
        ],
    },
    {
        "name": "Meningococcal disease",
        "snomed": "23511006",
        "icd10": "A39.9",
        "lab_loinc": "49672-8",
        "lab_name": "Neisseria meningitidis DNA [Presence] in Specimen by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "CSF",
        "lab_status": "preliminary",
        "medications": [
            {"rxnorm": "1665021", "name": "ceftriaxone 500 MG Injection"},
        ],
    },
    {
        "name": "Botulism",
        "snomed": "414488002",
        "icd10": "A05.1",
        "lab_loinc": "20703-5",
        "lab_name": "Clostridium botulinum toxin [Presence] in Specimen",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Stool",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "1603564", "name": "botulism antitoxin heptavalent"},
        ],
    },
    # --- Tier 3: vector-borne & emerging ---
    {
        "name": "Dengue",
        "snomed": "38362002",
        "icd10": "A90",
        "lab_loinc": "6386-1",
        "lab_name": "Dengue virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Zika virus disease",
        "snomed": "3928002",
        "icd10": "A92.5",
        "lab_loinc": "80825-3",
        "lab_name": "Zika virus RNA [Presence] in Serum by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "West Nile virus disease",
        "snomed": "417093003",
        "icd10": "A92.31",
        "lab_loinc": "32361-8",
        "lab_name": "West Nile virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Lyme disease",
        "snomed": "23502006",
        "icd10": "A69.20",
        "lab_loinc": "5061-1",
        "lab_name": "Borrelia burgdorferi Ab [Presence] in Serum by Immunoassay",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197984", "name": "doxycycline 100 MG Oral Tablet"},
        ],
    },
    {
        "name": "Malaria",
        "snomed": "61462000",
        "icd10": "B54",
        "lab_loinc": "32700-7",
        "lab_name": "Plasmodium sp identified in Blood by Light microscopy",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Blood",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "198137", "name": "chloroquine 250 MG Oral Tablet"},
        ],
    },
    # --- Tier 4: enteric & foodborne ---
    {
        "name": "Shiga toxin-producing E. coli infection",
        "snomed": "398565003",
        "icd10": "A04.3",
        "lab_loinc": "16832-8",
        "lab_name": "Escherichia coli O157 Ag [Presence] in Stool",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Stool",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Shigellosis",
        "snomed": "36188001",
        "icd10": "A03.9",
        "lab_loinc": "625-4",
        "lab_name": "Bacteria identified in Stool by Culture",
        "lab_result_code": "77352002",
        "lab_result_display": "Shigella",
        "specimen": "Stool",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197511", "name": "ciprofloxacin 500 MG Oral Tablet"},
        ],
    },
    {
        "name": "Campylobacteriosis",
        "snomed": "4834000",
        "icd10": "A04.5",
        "lab_loinc": "625-4",
        "lab_name": "Bacteria identified in Stool by Culture",
        "lab_result_code": "35408001",
        "lab_result_display": "Campylobacter",
        "specimen": "Stool",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197650", "name": "azithromycin 250 MG Oral Tablet"},
        ],
    },
    # --- Tier 5: vaccine-preventable ---
    {
        "name": "Hepatitis B",
        "snomed": "66071002",
        "icd10": "B16.9",
        "lab_loinc": "5196-1",
        "lab_name": "Hepatitis B virus surface Ag [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "597718", "name": "entecavir 0.5 MG Oral Tablet"},
        ],
    },
    {
        "name": "Mumps",
        "snomed": "36989005",
        "icd10": "B26.9",
        "lab_loinc": "21401-5",
        "lab_name": "Mumps virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Rubella",
        "snomed": "36653000",
        "icd10": "B06.9",
        "lab_loinc": "20458-6",
        "lab_name": "Rubella virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [],
    },
    {
        "name": "Varicella",
        "snomed": "38907003",
        "icd10": "B01.9",
        "lab_loinc": "21597-0",
        "lab_name": "Varicella zoster virus IgM Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "315523", "name": "acyclovir 800 MG Oral Tablet"},
        ],
    },
    # --- Tier 6: bioterrorism & rare ---
    {
        "name": "Anthrax",
        "snomed": "409498004",
        "icd10": "A22.9",
        "lab_loinc": "11470-2",
        "lab_name": "Bacillus anthracis DNA [Presence] in Blood by NAA",
        "lab_result_code": "260373001",
        "lab_result_display": "Detected",
        "specimen": "Blood",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197511", "name": "ciprofloxacin 500 MG Oral Tablet"},
            {"rxnorm": "197984", "name": "doxycycline 100 MG Oral Tablet"},
        ],
    },
    {
        "name": "Plague",
        "snomed": "58750007",
        "icd10": "A20.9",
        "lab_loinc": "11475-1",
        "lab_name": "Yersinia pestis Ab [Presence] in Serum",
        "lab_result_code": "10828004",
        "lab_result_display": "Positive",
        "specimen": "Serum",
        "lab_status": "final",
        "medications": [
            {"rxnorm": "197450", "name": "gentamicin 80 MG/2ML Injectable Solution"},
        ],
    },
]

FIRST_NAMES_F = ["Maria", "Sarah", "Emily", "Jessica", "Ashley", "Jennifer", "Amanda", "Stephanie", "Nicole", "Lisa", "Priya", "Aisha", "Mei", "Fatima", "Ana"]
FIRST_NAMES_M = ["James", "Robert", "Michael", "David", "John", "William", "Richard", "Joseph", "Thomas", "Daniel", "Raj", "Mohammed", "Wei", "Carlos", "Andre"]
LAST_NAMES = ["Smith", "Johnson", "Garcia", "Brown", "Davis", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Patel", "Kim", "Nguyen", "Ali", "Rodriguez"]
CITIES = [
    ("Denver", "CO", "80202"), ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Seattle", "WA", "98101"), ("Atlanta", "GA", "30301"),
    ("Boston", "MA", "02101"), ("Miami", "FL", "33101"), ("Portland", "OR", "97201"),
    ("Minneapolis", "MN", "55401"),
]
ORGS = [
    ("Denver Health Medical Center", "1234567800"),
    ("Northwestern Memorial Hospital", "1234567801"),
    ("Houston Methodist Hospital", "1234567802"),
    ("Mayo Clinic Phoenix", "1234567803"),
    ("Virginia Mason Medical Center", "1234567804"),
    ("Emory University Hospital", "1234567805"),
    ("Mass General Brigham", "1234567806"),
    ("Jackson Memorial Hospital", "1234567807"),
    ("OHSU Hospital", "1234567808"),
    ("Hennepin Healthcare", "1234567809"),
]
RACES = [
    ("2106-3", "White"), ("2054-5", "Black or African American"),
    ("2028-9", "Asian"), ("1002-5", "American Indian or Alaska Native"),
]
ETHNICITIES = [
    ("2135-2", "Hispanic or Latino"), ("2186-5", "Non Hispanic or Latino"),
]
SYMPTOMS = {
    "COVID-19": "fever ({}C), dry cough for {} days, fatigue, and shortness of breath",
    "Hepatitis A": "jaundice, abdominal pain, nausea, fatigue for {} days, dark urine",
    "Hepatitis C": "fatigue, abdominal discomfort, nausea for {} weeks, mild jaundice",
    "Pertussis": "persistent paroxysmal cough for {} weeks, post-tussive vomiting, inspiratory whoop",
    "Measles": "high fever ({}C), maculopapular rash for {} days, cough, coryza, conjunctivitis, Koplik spots",
    "Salmonella infection": "diarrhea for {} days, abdominal cramps, fever ({}C), nausea",
    "Tuberculosis": "persistent cough for {} weeks, night sweats, weight loss, hemoptysis",
    "Chlamydia trachomatis infection": "dysuria, abnormal discharge for {} days, lower abdominal pain",
    "Gonorrhea": "dysuria, purulent urethral discharge for {} days, pelvic pain",
    "Mpox": "vesicular/pustular rash for {} days, fever ({}C), lymphadenopathy, myalgia",
    "Syphilis": "painless chancre for {} days, regional lymphadenopathy, maculopapular rash on palms and soles",
    "HIV infection": "fever ({}C), weight loss, night sweats for {} weeks, lymphadenopathy, oral thrush",
    "Influenza A virus infection (novel)": "high fever ({}C), myalgia, cough, sore throat for {} days, severe fatigue",
    "Respiratory syncytial virus infection": "wheezing, nasal congestion, cough for {} days, low-grade fever ({}C), tachypnea",
    "Legionellosis": "high fever ({}C), productive cough for {} days, confusion, diarrhea, myalgia",
    "Meningococcal disease": "high fever ({}C), severe headache, stiff neck for {} days, petechial rash, photophobia",
    "Botulism": "descending paralysis, diplopia, dysphagia for {} days, dry mouth, ptosis",
    "Dengue": "high fever ({}C), severe headache, retro-orbital pain for {} days, myalgia, petechial rash",
    "Zika virus disease": "low-grade fever ({}C), maculopapular rash for {} days, conjunctivitis, arthralgia",
    "West Nile virus disease": "fever ({}C), headache, fatigue for {} days, body aches, skin rash",
    "Lyme disease": "erythema migrans rash for {} days, fever ({}C), fatigue, arthralgia, headache",
    "Malaria": "cyclical fever ({}C), rigors, diaphoresis for {} days, headache, splenomegaly",
    "Shiga toxin-producing E. coli infection": "bloody diarrhea for {} days, severe abdominal cramps, fever ({}C)",
    "Shigellosis": "watery diarrhea progressing to bloody for {} days, fever ({}C), tenesmus, abdominal pain",
    "Campylobacteriosis": "watery to bloody diarrhea for {} days, fever ({}C), abdominal cramping, nausea",
    "Hepatitis B": "fatigue, jaundice, abdominal pain for {} weeks, dark urine, nausea, arthralgia",
    "Mumps": "parotid gland swelling for {} days, fever ({}C), headache, myalgia, anorexia",
    "Rubella": "low-grade fever ({}C), maculopapular rash for {} days, lymphadenopathy, arthralgia",
    "Varicella": "pruritic vesicular rash for {} days, fever ({}C), malaise, anorexia",
    "Anthrax": "fever ({}C), malaise, chest discomfort for {} days, nonproductive cough, dyspnea",
    "Plague": "high fever ({}C), painful lymphadenopathy (bubo) for {} days, chills, prostration",
}
VITALS_LOINC = [
    ("8310-5", "Body temperature", "Cel"),
    ("8867-4", "Heart rate", "/min"),
    ("9279-1", "Respiratory rate", "/min"),
    ("2708-6", "Oxygen saturation", "%"),
    ("8480-6", "Systolic blood pressure", "mm[Hg]"),
]


def uid():
    return str(uuid.uuid4())


def rand_date(start_year=2025, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days))
    return d


def fmt_cda_date(dt, include_time=True):
    if include_time:
        return dt.strftime("%Y%m%d%H%M%S") + "-0600"
    return dt.strftime("%Y%m%d")


def fmt_fhir_date(dt, include_time=True):
    if include_time:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + "-06:00"
    return dt.strftime("%Y-%m-%d")


def generate_symptom_text(condition_name):
    template = SYMPTOMS.get(condition_name, "symptoms for {} days")
    days = random.randint(2, 14)
    temp = round(random.uniform(38.0, 40.5), 1)
    weeks = random.randint(2, 8)
    try:
        return template.format(temp, days)
    except (IndexError, KeyError):
        try:
            return template.format(days)
        except Exception:
            return template.format(weeks, temp)


def generate_vitals():
    temp = round(random.uniform(37.5, 40.5), 1)
    hr = random.randint(60, 120)
    rr = random.randint(12, 28)
    spo2 = random.randint(88, 100)
    sbp = random.randint(90, 160)
    return [
        ("8310-5", "Body temperature", temp, "Cel"),
        ("8867-4", "Heart rate", hr, "/min"),
        ("9279-1", "Respiratory rate", rr, "/min"),
        ("2708-6", "Oxygen saturation", spo2, "%"),
        ("8480-6", "Systolic blood pressure", sbp, "mm[Hg]"),
    ]


SYSTEM_PROMPT = (
    "Extract clinical entities from this eICR summary. Output JSON with: "
    "patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), "
    "medications (RxNorm), vitals, and a case summary. "
    "Include confidence scores. Output valid JSON only."
)

SYSTEM_PROMPT_COMPACT = (
    "Extract clinical entities from this eICR. Return minified JSON: "
    "{\"patient\":{...},\"conditions\":[...],\"labs\":[...],\"meds\":[...],\"vitals\":[...]}. "
    "All sections are arrays. Include SNOMED for conditions, LOINC for labs, "
    "RxNorm for meds. No summary. No markdown. JSON only."
)


def compute_age(birth_date, encounter_date):
    age = encounter_date.year - birth_date.year
    if (encounter_date.month, encounter_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def generate_sample(variation="normal", compact=False):
    """Generate one training sample: user input text + extraction JSON.

    variation: "normal", "missing", "negative_lab", "multi_condition"
    compact: if True, produce shorter output (no summary, no conf, fewer fields)
    """
    cond = random.choice(CONDITIONS)
    # For multi-condition, pick a second different condition
    cond2 = None
    if variation == "multi_condition":
        others = [c for c in CONDITIONS if c["name"] != cond["name"]]
        cond2 = random.choice(others)

    gender_is_female = random.choice([True, False])
    first = random.choice(FIRST_NAMES_F if gender_is_female else FIRST_NAMES_M)
    last = random.choice(LAST_NAMES)
    city, state, zipcode = random.choice(CITIES)
    org_name, org_npi = random.choice(ORGS)
    race_code, race_display = random.choice(RACES)
    eth_code, eth_display = random.choice(ETHNICITIES)

    encounter_date = rand_date()
    onset_date = encounter_date - timedelta(days=random.randint(1, 14))
    birth_year = random.randint(1940, 2005)
    birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))

    symptom_text = generate_symptom_text(cond["name"])
    vitals = generate_vitals()
    gender_code = "F" if gender_is_female else "M"
    phone = f"+1-{random.randint(200,999)}-555-{random.randint(1000,9999)}"
    age = compute_age(birth_date, encounter_date)

    # Determine if lab is negative for this variation
    lab_negative = variation == "negative_lab"
    lab_result = "Not detected" if lab_negative else cond["lab_result_display"]
    lab_result_snomed = "260415000" if lab_negative else cond["lab_result_code"]

    # Decide which optional fields to include
    include_vitals = variation != "missing" or random.random() > 0.5
    include_meds = variation != "missing" or random.random() > 0.5

    meds = cond.get("medications", []) if include_meds else []
    if cond2:
        meds = meds + cond2.get("medications", [])

    # --- Build user input text (must match server extract_key_data format) ---
    lines = [
        f"Patient: {first} {last}",
        f"Gender: {gender_code}",
        f"DOB: {birth_date.strftime('%Y-%m-%d')}",
        f"Race: {race_display}",
        f"Ethnicity: {eth_display}",
        f"Location: {city}, {state} {zipcode}",
        f"Phone: {phone}",
        f"Facility: {org_name} (NPI: {org_npi})",
        f"Encounter: {fmt_fhir_date(encounter_date, False)}",
        f"Reason: {symptom_text}",
        f"Dx: {cond['name']} (SNOMED {cond['snomed']})",
        f"Lab: {cond['lab_name']} (LOINC {cond['lab_loinc']}) - {lab_result} [{cond.get('specimen', 'Blood')}, {cond.get('lab_status', 'final')}]",
    ]
    if cond2:
        onset2 = encounter_date - timedelta(days=random.randint(1, 14))
        lines.append(f"Dx: {cond2['name']} (SNOMED {cond2['snomed']})")
        lines.append(f"Lab: {cond2['lab_name']} (LOINC {cond2['lab_loinc']}) - {cond2['lab_result_display']} [{cond2.get('specimen', 'Blood')}, {cond2.get('lab_status', 'final')}]")

    if include_vitals:
        v = vitals
        lines.append(f"Vitals: Temp {v[0][2]}C, HR {v[1][2]}, RR {v[2][2]}, SpO2 {v[3][2]}%, BP {v[4][2]}")

    for med in meds:
        lines.append(f"Meds: {med['name']} (RxNorm {med['rxnorm']})")

    user_input = "\n".join(lines)

    # --- Build extraction JSON (model output) ---
    def conf(low=0.88):
        """Generate a confidence score, occasionally low."""
        if random.random() < 0.05:
            return round(random.uniform(0.70, 0.85), 2)
        return round(random.uniform(low, 0.99), 2)

    if compact:
        # Minimal format matching proven benchmark schema (exp25).
        # Only fields the scorer checks: snomed, loinc, rxnorm + schema keys.
        conditions_list = [{"name": cond["name"], "snomed": cond["snomed"]}]
        if cond2:
            conditions_list.append({"name": cond2["name"], "snomed": cond2["snomed"]})

        labs_list = [{"name": cond["lab_name"], "loinc": cond["lab_loinc"], "result": lab_result}]
        if cond2:
            labs_list.append({"name": cond2["lab_name"], "loinc": cond2["lab_loinc"], "result": cond2["lab_result_display"]})

        meds_list = [{"name": m["name"], "rxnorm": m["rxnorm"]} for m in meds]
    else:
        conditions_list = [{
            "name": cond["name"],
            "snomed": cond["snomed"],
            "icd10": cond["icd10"],
            "onset": fmt_fhir_date(onset_date, False),
            "status": "suspected" if lab_negative else "active",
            "conf": conf(),
        }]
        if cond2:
            conditions_list.append({
                "name": cond2["name"],
                "snomed": cond2["snomed"],
                "icd10": cond2["icd10"],
                "onset": fmt_fhir_date(onset2, False),
                "status": "active",
                "conf": conf(),
            })

        labs_list = [{
            "name": cond["lab_name"],
            "loinc": cond["lab_loinc"],
            "result": lab_result,
            "result_snomed": lab_result_snomed,
            "specimen": cond.get("specimen", "Blood"),
            "lab_status": cond.get("lab_status", "final"),
            "date": fmt_fhir_date(encounter_date, False),
            "conf": conf(0.90),
        }]
        if cond2:
            labs_list.append({
                "name": cond2["lab_name"],
                "loinc": cond2["lab_loinc"],
                "result": cond2["lab_result_display"],
                "result_snomed": cond2["lab_result_code"],
                "specimen": cond2.get("specimen", "Blood"),
                "lab_status": cond2.get("lab_status", "final"),
                "date": fmt_fhir_date(encounter_date, False),
                "conf": conf(0.90),
            })

        meds_list = [{"name": m["name"], "rxnorm": m["rxnorm"], "conf": conf(0.85)} for m in meds]

    vitals_obj = None
    if include_vitals:
        vitals_obj = {
            "temp": vitals[0][2],
            "hr": vitals[1][2],
            "rr": vitals[2][2],
            "spo2": vitals[3][2],
            "sbp": vitals[4][2],
        }

    # Build summary
    dx_text = cond["name"]
    if lab_negative:
        dx_text = f"{cond['name']} (suspected, lab negative)"
    if cond2:
        dx_text += f", {cond2['name']}"

    med_text = ", ".join(m["name"].split(" ")[0] for m in meds) if meds else "supportive care"

    summary = (
        f"{first} {last}, {age}{gender_code}, {eth_display}, "
        f"presents at {org_name} on {fmt_fhir_date(encounter_date, False)} "
        f"with {symptom_text}. "
        f"{cond['lab_name'].split('[')[0].strip()} {lab_result.lower()}. "
        f"Dx: {dx_text}. Rx: {med_text}. "
        f"Reportable condition per CSTE criteria."
    )

    if compact:
        # Vitals as array (matches benchmark scorer expectation)
        vitals_arr = []
        if include_vitals:
            v = vitals
            vitals_arr = [
                {"name": "temp", "value": f"{v[0][2]}C"},
                {"name": "hr", "value": str(v[1][2])},
                {"name": "rr", "value": str(v[2][2])},
                {"name": "spo2", "value": f"{v[3][2]}%"},
                {"name": "sbp", "value": str(v[4][2])},
            ]

        extraction = {
            "patient": {
                "name": f"{first} {last}",
                "gender": gender_code,
                "dob": birth_date.strftime("%Y-%m-%d"),
            },
            "conditions": conditions_list,
            "labs": labs_list,
            "meds": meds_list,
            "vitals": vitals_arr,
        }
        return user_input, json.dumps(extraction, separators=(",", ":"))

    extraction = {
        "patient": {
            "name": f"{first} {last}",
            "dob": birth_date.strftime("%Y-%m-%d"),
            "sex": gender_code,
            "race": race_display,
            "ethnicity": eth_display,
            "addr": f"{city}, {state} {zipcode}",
            "phone": phone,
        },
        "encounter": {
            "date": fmt_fhir_date(encounter_date, False),
            "type": "Office visit",
            "facility": org_name,
            "npi": org_npi,
        },
        "conditions": conditions_list,
        "labs": labs_list,
        "vitals": vitals_obj,
        "meds": meds_list,
        "summary": summary,
        "reportable": True,
        "jurisdiction": state,
    }

    return user_input, json.dumps(extraction, separators=(",", ":"))


def build_conversation(user_input, extraction_json, compact=False):
    """Format as a chat conversation for Unsloth SFT."""
    prompt = SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT
    return {
        "conversations": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": extraction_json},
        ]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ClinIQ training data")
    parser.add_argument("--compact", action="store_true",
                        help="Generate compact output (no summary, no conf, fewer fields)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    num_samples = args.samples
    compact = args.compact

    suffix = "-compact" if compact else ""
    print(f"Generating {num_samples} {'compact' if compact else 'verbose'} samples...")

    dataset = []
    for i in range(num_samples):
        # Determine variation type — heavier weight on challenging cases
        r = random.random()
        if r < 0.10:
            variation = "missing"
        elif r < 0.25:
            variation = "negative_lab"
        elif r < 0.40:
            variation = "multi_condition"
        else:
            variation = "normal"

        user_input, extraction = generate_sample(variation, compact=compact)
        conv = build_conversation(user_input, extraction, compact=compact)
        dataset.append(conv)

    # Write training set (80%) and validation set (20%)
    split = int(len(dataset) * 0.8)
    train = dataset[:split]
    val = dataset[split:]

    train_path = f"data/train{suffix}.jsonl"
    val_path = f"data/val{suffix}.jsonl"

    with open(train_path, "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w") as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

    # Show stats
    print(f"Generated {len(train)} training samples → {train_path}")
    print(f"Generated {len(val)} validation samples → {val_path}")
    print(f"Conditions covered: {len(CONDITIONS)}")

    # Spot-check token estimate on first sample
    sample = json.dumps(dataset[0])
    est_tokens = len(sample) // 4  # rough char/token ratio
    print(f"Estimated tokens per sample: ~{est_tokens}")


if __name__ == "__main__":
    main()
