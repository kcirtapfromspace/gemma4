#!/usr/bin/env python3
"""Build a synthetic LLM-required ClinIQ evaluation/training corpus.

The generated user narratives intentionally omit inline SNOMED/LOINC/RxNorm
codes. They are meant to exercise the agent/model path: infer a condition from
clinical pattern or synonym, call lookup tools for labs/medications, validate,
and emit raw-code JSON. Hard negatives cover quoted rumors, exposure-only,
family-history, ruled-out, vaccine-target, and contraindicated-med contexts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUT_JSONL = "build/llm_required/generated/llm_required_synth200.jsonl"
DEFAULT_OUT_MANIFEST = "build/llm_required/generated/llm_required_synth200.manifest.json"


FIRST_NAMES = [
    "Aisha",
    "Ana",
    "Andre",
    "Camila",
    "Daniel",
    "Elena",
    "Huy",
    "Imani",
    "Jonah",
    "Lucia",
    "Mateus",
    "Mina",
    "Noah",
    "Priya",
    "Rafael",
    "Salif",
    "Valeria",
    "Wei",
]
LAST_NAMES = [
    "Brooks",
    "Costa",
    "Diallo",
    "Garcia",
    "Johnson",
    "Kim",
    "Morgan",
    "Nguyen",
    "Oliveira",
    "Patel",
    "Rivera",
    "Soto",
    "Tran",
    "Williams",
]
LOCATIONS = [
    "Atlanta, GA 30303",
    "Boise, ID 83702",
    "Chicago, IL 60612",
    "Denver, CO 80204",
    "Houston, TX 77002",
    "Las Vegas, NV 89101",
    "Minneapolis, MN 55404",
    "Newark, NJ 07102",
    "Orlando, FL 32803",
    "Phoenix, AZ 85004",
    "San Jose, CA 95112",
]


POSITIVE_PROFILES: list[dict[str, Any]] = [
    {
        "key": "measles_pattern_es",
        "axis": "clinical_pattern_multilingual",
        "expected_conditions": ["14189004"],
        "expected_loincs": [],
        "expected_rxnorms": [],
        "snippets": [
            "nota en espanol: fiebre alta por cinco dias, tos seca, coriza y conjuntivitis intensa. En la mucosa bucal se observan pequenas manchas blancas frente a los molares; el exantema maculopapular comenzo en la cara y ahora baja al tronco. Solo tiene una dosis documentada de vacuna triple viral.",
            "dictado bilingue: fever, cough, coryza, conjunctivitis, Koplik-like buccal spots, and a descending maculopapular rash after travel. The clinician writes that this is the classic vaccine-preventable exanthem and starts airborne isolation.",
        ],
    },
    {
        "key": "cholera_pattern_fr",
        "axis": "clinical_pattern_multilingual",
        "expected_conditions": ["1857005"],
        "expected_loincs": [],
        "expected_rxnorms": [],
        "snippets": [
            "note clinique en francais: diarrhee aqueuse profuse depuis dix-huit heures, vomissements, crampes des jambes, voix faible, hypotension et pli cutane persistant apres retour d'une region avec flambee digestive. Les selles sont de type eau de riz.",
            "travel clinic note: sudden rice-water stool, severe dehydration, leg cramps, and rapid fluid loss after a waterborne outbreak exposure. Stool culture for Vibrio was sent to public health.",
        ],
    },
    {
        "key": "pulmonary_tb_vi",
        "axis": "clinical_pattern_multilingual",
        "expected_conditions": ["56717001"],
        "expected_loincs": ["38379-4"],
        "expected_rxnorms": ["197622", "199279"],
        "snippets": [
            "ghi chu khong dau: ho co dom va doi khi ra mau hon hai thang, sot nhe ve chieu, do mo hoi dem va sut can. X quang thay hang o thuy tren phoi; mau dom soi thay truc khuan khang acid, molecular assay for the Mycobacterium tuberculosis complex was positive with no rifampin resistance. Started isoniazid and rifampin.",
            "pulmonary clinic note: chronic productive cough, night sweats, upper-lobe cavitation, acid-fast bacilli on sputum smear, and a positive MTB/RIF nucleic-acid test. Airborne isolation and isoniazid plus rifampin were started.",
        ],
    },
    {
        "key": "pertussis_infant_pt",
        "axis": "clinical_pattern_multilingual",
        "expected_conditions": ["27836007"],
        "expected_loincs": ["71773-2"],
        "expected_rxnorms": ["197650"],
        "snippets": [
            "nota em portugues: lactente com tosse em crises ha quase quatro semanas, som inspiratorio agudo no fim das crises, vomitos depois da tosse e pausas respiratorias. Swab nasofaringeo molecular para Bordetella came back positive. Azithromycin started for treatment and household prophylaxis.",
            "pediatric dictation: paroxysmal cough, inspiratory whoop, post-tussive emesis, apnea spells, delayed vaccines, and a positive Bordetella pertussis molecular swab. Azithromycin was prescribed.",
        ],
    },
    {
        "key": "dengue_breakbone",
        "axis": "syndrome_synonym",
        "expected_conditions": ["38362002"],
        "expected_loincs": ["6386-1"],
        "expected_rxnorms": [],
        "snippets": [
            "reason: fever after Caribbean travel, retro-orbital headache, intense bone-breaking myalgias, petechiae, and falling platelets. Arboviral IgM panel returned positive for the mosquito-borne flavivirus that causes breakbone fever; Zika and chikungunya were negative.",
            "assessment: travel-associated breakbone fever without warning signs. Platelets are down, tourniquet test is positive, and the dengue IgM assay resulted positive.",
        ],
    },
    {
        "key": "rsv_bronchiolitis",
        "axis": "lab_synonym_lookup",
        "expected_conditions": ["55735004"],
        "expected_loincs": ["31933-7"],
        "expected_rxnorms": [],
        "snippets": [
            "pediatric ED note: infant with cough, rhinorrhea, wheeze, mild retractions, and poor feeding. Rapid respiratory syncytial virus antigen testing from a nasal sample was positive; flu and SARS-CoV-2 were negative.",
            "assessment: bronchiolitis due to respiratory syncytial virus. Nasal RSV antigen returned positive and the child improved with suctioning and low-flow oxygen.",
        ],
    },
    {
        "key": "h5n1_avian_flu",
        "axis": "rare_reportable_lookup",
        "expected_conditions": ["442695009"],
        "expected_loincs": ["100343-3"],
        "expected_rxnorms": ["261244"],
        "snippets": [
            "farm worker with fever, conjunctivitis, cough, and poultry exposure during a local outbreak. State lab reported influenza A H5 RNA detected from a nasopharyngeal specimen. Oseltamivir was started and public health was notified.",
            "avian exposure note: severe influenza-like illness after culling sick birds; the H5 RNA nucleic acid test was positive. Treating team started oseltamivir and respiratory isolation.",
        ],
    },
    {
        "key": "mpox_lesion",
        "axis": "renamed_condition_lookup",
        "expected_conditions": ["22253000"],
        "expected_loincs": ["96741-4"],
        "expected_rxnorms": ["1923432"],
        "snippets": [
            "clinical impression: painful disseminated vesiculopustular rash with inguinal lymphadenopathy after new sexual exposure. Lesion swab showed monkeypox virus DNA by PCR. Tecovirimat was started.",
            "rash clinic note: febrile patient with umbilicated genital and trunk lesions, tender nodes, and positive orthopox/monkeypox DNA testing from a lesion swab. Treatment is tecovirimat.",
        ],
    },
    {
        "key": "hiv_seroconversion",
        "axis": "lab_synonym_lookup",
        "expected_conditions": ["86406008"],
        "expected_loincs": ["75622-1", "24467-3"],
        "expected_rxnorms": ["1999563"],
        "snippets": [
            "assessment: acute retroviral syndrome with fever, rash, oral ulcers, and positive fourth-generation HIV antigen/antibody screen. CD4 lymphocyte count is 180 cells/uL. Started bictegravir/emtricitabine/tenofovir alafenamide.",
            "infectious disease consult: new human immunodeficiency virus disease confirmed by HIV 1 and 2 Ag/Ab testing; CD4 T-cell count is low. Began bictegravir combination therapy.",
        ],
    },
    {
        "key": "lyme_borreliosis",
        "axis": "synonym_plus_med",
        "expected_conditions": ["23502006"],
        "expected_loincs": ["5061-1"],
        "expected_rxnorms": ["197984"],
        "snippets": [
            "outdoor exposure note: expanding erythema migrans rash after tick bite, facial palsy, and positive Borrelia burgdorferi antibody testing. Doxycycline was prescribed.",
            "assessment: tick-borne borreliosis with migratory annular rash and positive Borrelia serology. Treat with doxycycline.",
        ],
    },
    {
        "key": "syphilis_chancre",
        "axis": "synonym_plus_med",
        "expected_conditions": ["76272004"],
        "expected_loincs": ["20507-0"],
        "expected_rxnorms": ["105220"],
        "snippets": [
            "sexual health visit: painless chancre, regional lymphadenopathy, and palmar rash. Treponema pallidum antibody is reactive. Penicillin G benzathine was administered.",
            "assessment: primary treponemal infection with positive T. pallidum antibody screen. Treating with penicillin G benzathine.",
        ],
    },
    {
        "key": "covid_paxlovid",
        "axis": "standard_positive_no_codes",
        "expected_conditions": ["840539006"],
        "expected_loincs": ["94500-6"],
        "expected_rxnorms": ["2599543"],
        "snippets": [
            "urgent care note: fever, dry cough, anosmia, and household exposure. SARS-CoV-2 RNA nucleic acid test from respiratory sample was detected. Nirmatrelvir with ritonavir was prescribed.",
            "assessment: coronavirus disease 2019 confirmed by positive SARS-CoV-2 RNA assay. Treating high-risk patient with Paxlovid.",
        ],
    },
]


NEGATIVE_TEMPLATES: list[dict[str, Any]] = [
    {
        "key": "family_history_only",
        "axis": "hard_negative_family_history",
        "snippet": "Reason: occupational paperwork. Family history lists HIV infection in the patient's father, but the patient is asymptomatic. The patient's own HIV antigen/antibody screen is negative today. Assessment says exposure concern only and no reportable infectious disease diagnosed.",
    },
    {
        "key": "quoted_rumor_only",
        "axis": "hard_negative_quoted_text",
        "snippet": "Reason: mild viral upper respiratory symptoms without rash. Parent brought a printed internet post claiming measles, mumps, rubella, and vaccine shedding. Clinician documents this as quoted misinformation only, not the patient's diagnosis, symptoms, or lab result.",
    },
    {
        "key": "ruled_out_differential",
        "axis": "hard_negative_ruled_out",
        "snippet": "Reason: fever after travel. Differential included measles, rubella, dengue, and Zika, but all were ruled out by exam and negative serology. Final assessment is non-reportable viral syndrome; no public-health condition diagnosed.",
    },
    {
        "key": "vaccine_target_only",
        "axis": "hard_negative_vaccine_target",
        "snippet": "Reason: pre-travel vaccine counseling. The note lists measles, hepatitis B, varicella, and influenza as vaccine targets reviewed. The patient is well, has no symptoms, no positive labs, and no active infectious diagnosis.",
    },
    {
        "key": "contraindicated_med",
        "axis": "hard_negative_contraindicated_med",
        "snippet": "Prenatal visit: patient has asymptomatic bacteriuria. Doxycycline is mentioned only as contraindicated in pregnancy and explicitly not prescribed. No chlamydia, Lyme disease, or other reportable infection is diagnosed.",
    },
    {
        "key": "exposure_negative_tests",
        "axis": "hard_negative_exposure_only",
        "snippet": "Employee health visit after needlestick from a source patient initially rumored to have hepatitis C and HIV. Source testing and employee baseline tests were negative. Assessment: bloodborne pathogen exposure only, no infection diagnosis.",
    },
]


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def demographics(rng: random.Random, idx: int) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    year = rng.randint(1948, 2025)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return {
        "name": f"{first} {last}",
        "gender": rng.choice(["F", "M"]),
        "dob": f"{year:04d}-{month:02d}-{day:02d}",
        "location": rng.choice(LOCATIONS),
        "encounter": f"2026-{(idx % 12) + 1:02d}-{(idx % 27) + 1:02d}",
    }


def render_case_header(demo: dict[str, str]) -> str:
    return (
        f"Patient: {demo['name']}\n"
        f"Gender: {demo['gender']}\n"
        f"DOB: {demo['dob']}\n"
        f"Location: {demo['location']}\n"
        f"Encounter: {demo['encounter']}\n"
    )


def positive_case(profile: dict[str, Any], idx: int, rng: random.Random) -> dict[str, Any]:
    demo = demographics(rng, idx)
    snippet = rng.choice(profile["snippets"])
    # Add benign distractors that should not alter expected codes.
    distractors = [
        "Family asks about unrelated measles news; clinician documents it as background only.",
        "Past medical history includes hypertension and seasonal allergies.",
        "A vaccine status review is included, but today's assessment is based on active symptoms and testing.",
        "A ruled-out alternative is mentioned in the differential and explicitly not carried forward.",
    ]
    if idx % 3 == 0:
        snippet = f"{snippet}\n\nContext note: {rng.choice(distractors)}"
    user = render_case_header(demo) + "\n" + snippet
    text_for_id = f"{profile['key']}:{idx}:{user}"
    return {
        "case_id": stable_id("llmreq_synth", text_for_id),
        "description": (
            "Synthetic LLM-required positive; no inline code strings in user "
            f"text; axis={profile['axis']}; profile={profile['key']}"
        ),
        "axis": profile["axis"],
        "profile": profile["key"],
        "user": user,
        "expected_conditions": list(profile["expected_conditions"]),
        "expected_loincs": list(profile["expected_loincs"]),
        "expected_rxnorms": list(profile["expected_rxnorms"]),
    }


def negative_case(template: dict[str, Any], idx: int, rng: random.Random) -> dict[str, Any]:
    demo = demographics(rng, idx)
    user = render_case_header(demo) + "\n" + template["snippet"]
    text_for_id = f"{template['key']}:{idx}:{user}"
    return {
        "case_id": stable_id("llmreq_synth_neg", text_for_id),
        "description": (
            "Synthetic LLM-required hard negative; disease/drug words appear "
            f"only in non-asserted context; axis={template['axis']}"
        ),
        "axis": template["axis"],
        "profile": template["key"],
        "user": user,
        "expected_conditions": [],
        "expected_loincs": [],
        "expected_rxnorms": [],
    }


def build_cases(limit: int, seed: int, positive_ratio: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    n_positive = round(limit * positive_ratio)
    n_negative = limit - n_positive
    for idx in range(n_positive):
        profile = POSITIVE_PROFILES[idx % len(POSITIVE_PROFILES)]
        cases.append(positive_case(profile, idx, rng))
    for idx in range(n_negative):
        template = NEGATIVE_TEMPLATES[idx % len(NEGATIVE_TEMPLATES)]
        cases.append(negative_case(template, n_positive + idx, rng))
    rng.shuffle(cases)
    return cases


def manifest_for(cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    axis_counts = Counter(case["axis"] for case in cases)
    profile_counts = Counter(case["profile"] for case in cases)
    code_counts: dict[str, Counter[str]] = {
        "conditions": Counter(),
        "loincs": Counter(),
        "rxnorms": Counter(),
    }
    for case in cases:
        code_counts["conditions"].update(case.get("expected_conditions") or [])
        code_counts["loincs"].update(case.get("expected_loincs") or [])
        code_counts["rxnorms"].update(case.get("expected_rxnorms") or [])
    expected_totals = {bucket: sum(counter.values()) for bucket, counter in code_counts.items()}
    expected_totals["total"] = sum(expected_totals.values())
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/build_llm_required_corpus.py",
        "limit": args.limit,
        "seed": args.seed,
        "positive_ratio": args.positive_ratio,
        "cases": len(cases),
        "axis_counts": dict(sorted(axis_counts.items())),
        "profile_counts": dict(sorted(profile_counts.items())),
        "expected_codes": expected_totals,
        "unique_expected_codes": {
            bucket: len(counter) for bucket, counter in code_counts.items()
        },
        "top_expected_codes": {
            bucket: [
                {"code": code, "count": count}
                for code, count in counter.most_common(10)
            ]
            for bucket, counter in code_counts.items()
        },
        "claim_boundary": (
            "Synthetic LLM-required cases are generated from fixed templates "
            "and should be used for model behavior training/evaluation, not as "
            "an independent real-world eICR benchmark."
        ),
        "case_id_sample": [case["case_id"] for case in cases[:10]],
    }


def render_summary(manifest: dict[str, Any], dry_run: bool) -> str:
    prefix = "DRY RUN" if dry_run else "WROTE"
    return "\n".join(
        [
            f"{prefix}: {manifest['cases']} LLM-required synthetic cases",
            f"Expected codes: {json.dumps(manifest['expected_codes'], sort_keys=True)}",
            f"Axes: {json.dumps(manifest['axis_counts'], sort_keys=True)}",
            f"Unique expected codes: {json.dumps(manifest['unique_expected_codes'], sort_keys=True)}",
            f"Case ID sample: {', '.join(manifest['case_id_sample'])}",
            f"Claim boundary: {manifest['claim_boundary']}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--positive-ratio", type=float, default=0.72)
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-manifest", default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not 0.0 <= args.positive_ratio <= 1.0:
        raise SystemExit("--positive-ratio must be between 0 and 1")

    cases = build_cases(args.limit, args.seed, args.positive_ratio)
    manifest = manifest_for(cases, args)
    if not args.dry_run:
        out_jsonl = Path(args.out_jsonl)
        out_manifest = Path(args.out_manifest)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl.write_text(
            "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases)
        )
        out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(render_summary(manifest, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
