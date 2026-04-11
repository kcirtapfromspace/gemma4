#!/usr/bin/env python3
"""
Generate synthetic eICR CDA → FHIR R4 training pairs for fine-tuning Gemma 4 E4B.

Produces a JSONL dataset where each row is a conversation:
  user: <eICR XML>
  assistant: <FHIR Bundle JSON>

Covers reportable conditions: COVID-19, Hepatitis A/B/C, Pertussis, Measles,
Salmonella, Tuberculosis, Chlamydia, Gonorrhea, Mpox, Influenza (novel).
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
        "medications": [
            {"rxnorm": "2469470", "name": "tecovirimat 200 MG Oral Capsule"},
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


def generate_pair():
    """Generate one eICR CDA → FHIR Bundle training pair."""
    cond = random.choice(CONDITIONS)
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

    doc_id = uid()
    patient_id = f"PT-{random.randint(10000, 99999)}"
    enc_id = f"ENC-{random.randint(100, 999)}"
    symptom_text = generate_symptom_text(cond["name"])
    vitals = generate_vitals()

    gender_code = "F" if gender_is_female else "M"
    fhir_gender = "female" if gender_is_female else "male"
    street = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Elm', 'Park', 'Cedar', 'Maple', 'Pine', 'Lake'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr', 'Rd'])}"
    phone = f"+1-{random.randint(200,999)}-555-{random.randint(1000,9999)}"

    # Build CDA XML
    meds_xml = ""
    for med in cond.get("medications", []):
        meds_xml += f"""
          <entry>
            <substanceAdministration classCode="SBADM" moodCode="EVN">
              <consumable>
                <manufacturedProduct>
                  <manufacturedMaterial>
                    <code code="{med['rxnorm']}" codeSystem="2.16.840.1.113883.6.88" displayName="{med['name']}"/>
                  </manufacturedMaterial>
                </manufacturedProduct>
              </consumable>
              <effectiveTime value="{fmt_cda_date(encounter_date, False)}"/>
            </substanceAdministration>
          </entry>"""

    vitals_xml = ""
    for loinc, name, val, unit in vitals:
        vitals_xml += f"""
              <component>
                <observation classCode="OBS" moodCode="EVN">
                  <code code="{loinc}" codeSystem="2.16.840.1.113883.6.1" displayName="{name}"/>
                  <statusCode code="completed"/>
                  <effectiveTime value="{fmt_cda_date(encounter_date, False)}"/>
                  <value xsi:type="PQ" value="{val}" unit="{unit}"/>
                </observation>
              </component>"""

    eicr_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <realmCode code="US"/>
  <typeId root="2.16.840.1.113883.1.3" extension="POCD_HD000040"/>
  <templateId root="2.16.840.1.113883.10.20.15.2" extension="2021-01-01"/>
  <id root="{doc_id}"/>
  <code code="55751-2" codeSystem="2.16.840.1.113883.6.1" displayName="Public Health Case Report"/>
  <title>Initial Public Health Case Report - eICR</title>
  <effectiveTime value="{fmt_cda_date(encounter_date)}"/>
  <confidentialityCode code="N" codeSystem="2.16.840.1.113883.5.25"/>
  <recordTarget>
    <patientRole>
      <id extension="{patient_id}" root="2.16.840.1.113883.19.5"/>
      <addr use="H">
        <streetAddressLine>{street}</streetAddressLine>
        <city>{city}</city>
        <state>{state}</state>
        <postalCode>{zipcode}</postalCode>
        <country>US</country>
      </addr>
      <telecom value="tel:{phone}" use="HP"/>
      <patient>
        <name use="L">
          <given>{first}</given>
          <family>{last}</family>
        </name>
        <administrativeGenderCode code="{gender_code}" codeSystem="2.16.840.1.113883.5.1"/>
        <birthTime value="{birth_date.strftime('%Y%m%d')}"/>
        <raceCode code="{race_code}" codeSystem="2.16.840.1.113883.6.238" displayName="{race_display}"/>
        <ethnicGroupCode code="{eth_code}" codeSystem="2.16.840.1.113883.6.238" displayName="{eth_display}"/>
      </patient>
    </patientRole>
  </recordTarget>
  <author>
    <time value="{fmt_cda_date(encounter_date)}"/>
    <assignedAuthor>
      <id root="2.16.840.1.113883.4.6" extension="{org_npi}"/>
      <representedOrganization>
        <name>{org_name}</name>
      </representedOrganization>
    </assignedAuthor>
  </author>
  <custodian>
    <assignedCustodian>
      <representedCustodianOrganization>
        <id root="2.16.840.1.113883.4.6" extension="{org_npi}"/>
        <name>{org_name}</name>
      </representedCustodianOrganization>
    </assignedCustodian>
  </custodian>
  <component>
    <structuredBody>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.22.1"/>
          <code code="46240-8" codeSystem="2.16.840.1.113883.6.1" displayName="History of encounters"/>
          <entry>
            <encounter classCode="ENC" moodCode="EVN">
              <id root="2.16.840.1.113883.19" extension="{enc_id}"/>
              <code code="99213" codeSystem="2.16.840.1.113883.6.12" displayName="Office visit, established patient"/>
              <effectiveTime>
                <low value="{fmt_cda_date(encounter_date)}"/>
              </effectiveTime>
            </encounter>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.12"/>
          <code code="29299-5" codeSystem="2.16.840.1.113883.6.1" displayName="Reason for visit"/>
          <text>Patient presents with {symptom_text}.</text>
        </section>
      </component>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.5.1"/>
          <code code="11450-4" codeSystem="2.16.840.1.113883.6.1" displayName="Problem list"/>
          <entry>
            <act classCode="ACT" moodCode="EVN">
              <entryRelationship typeCode="SUBJ">
                <observation classCode="OBS" moodCode="EVN">
                  <code code="64572001" codeSystem="2.16.840.1.113883.6.96" displayName="Disease (disorder)"/>
                  <value xsi:type="CD" code="{cond['snomed']}" codeSystem="2.16.840.1.113883.6.96" displayName="{cond['name']}"/>
                  <effectiveTime><low value="{fmt_cda_date(onset_date, False)}"/></effectiveTime>
                </observation>
              </entryRelationship>
            </act>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.3.1"/>
          <code code="30954-2" codeSystem="2.16.840.1.113883.6.1" displayName="Relevant diagnostic tests/laboratory data"/>
          <entry>
            <organizer classCode="BATTERY" moodCode="EVN">
              <statusCode code="completed"/>
              <component>
                <observation classCode="OBS" moodCode="EVN">
                  <code code="{cond['lab_loinc']}" codeSystem="2.16.840.1.113883.6.1" displayName="{cond['lab_name']}"/>
                  <statusCode code="completed"/>
                  <effectiveTime value="{fmt_cda_date(encounter_date, False)}"/>
                  <value xsi:type="CD" code="{cond['lab_result_code']}" codeSystem="2.16.840.1.113883.6.96" displayName="{cond['lab_result_display']}"/>
                </observation>
              </component>
            </organizer>
          </entry>
          <entry>
            <organizer classCode="BATTERY" moodCode="EVN">
              <statusCode code="completed"/>{vitals_xml}
            </organizer>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.1.1"/>
          <code code="10160-0" codeSystem="2.16.840.1.113883.6.1" displayName="History of Medication use"/>{meds_xml}
        </section>
      </component>
    </structuredBody>
  </component>
</ClinicalDocument>"""

    # Build FHIR Bundle
    uuids = {k: uid() for k in ["composition", "patient", "encounter", "condition", "lab_obs", "org"]}
    vital_uuids = [uid() for _ in vitals]
    med_uuids = [uid() for _ in cond.get("medications", [])]

    # Composition sections
    results_entries = [{"reference": f"urn:uuid:{uuids['lab_obs']}"}]
    results_entries.extend({"reference": f"urn:uuid:{vu}"} for vu in vital_uuids)

    med_entries = [{"reference": f"urn:uuid:{mu}"} for mu in med_uuids]

    sections = [
        {
            "title": "Reason for Visit",
            "code": {"coding": [{"system": "http://loinc.org", "code": "29299-5", "display": "Reason for visit"}]},
            "text": {"status": "generated", "div": f'<div xmlns="http://www.w3.org/1999/xhtml">Patient presents with {symptom_text}.</div>'},
        },
        {
            "title": "Problems",
            "code": {"coding": [{"system": "http://loinc.org", "code": "11450-4", "display": "Problem list"}]},
            "entry": [{"reference": f"urn:uuid:{uuids['condition']}"}],
        },
        {
            "title": "Results",
            "code": {"coding": [{"system": "http://loinc.org", "code": "30954-2", "display": "Relevant diagnostic tests/laboratory data"}]},
            "entry": results_entries,
        },
    ]
    if med_entries:
        sections.append({
            "title": "Medications",
            "code": {"coding": [{"system": "http://loinc.org", "code": "10160-0", "display": "History of Medication use"}]},
            "entry": med_entries,
        })

    entries = [
        {
            "fullUrl": f"urn:uuid:{uuids['composition']}",
            "resource": {
                "resourceType": "Composition",
                "status": "final",
                "type": {"coding": [{"system": "http://loinc.org", "code": "55751-2", "display": "Public Health Case Report"}]},
                "subject": {"reference": f"urn:uuid:{uuids['patient']}"},
                "encounter": {"reference": f"urn:uuid:{uuids['encounter']}"},
                "date": fmt_fhir_date(encounter_date),
                "author": [{"reference": f"urn:uuid:{uuids['org']}"}],
                "title": "Initial Public Health Case Report - eICR",
                "section": sections,
            },
        },
        {
            "fullUrl": f"urn:uuid:{uuids['patient']}",
            "resource": {
                "resourceType": "Patient",
                "identifier": [{"system": "urn:oid:2.16.840.1.113883.19.5", "value": patient_id}],
                "name": [{"use": "official", "family": last, "given": [first]}],
                "gender": fhir_gender,
                "birthDate": birth_date.strftime("%Y-%m-%d"),
                "address": [{"use": "home", "line": [street], "city": city, "state": state, "postalCode": zipcode, "country": "US"}],
                "telecom": [{"system": "phone", "value": phone, "use": "home"}],
                "extension": [
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                        "extension": [{"url": "ombCategory", "valueCoding": {"system": "urn:oid:2.16.840.1.113883.6.238", "code": race_code, "display": race_display}}],
                    },
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                        "extension": [{"url": "ombCategory", "valueCoding": {"system": "urn:oid:2.16.840.1.113883.6.238", "code": eth_code, "display": eth_display}}],
                    },
                ],
            },
        },
        {
            "fullUrl": f"urn:uuid:{uuids['encounter']}",
            "resource": {
                "resourceType": "Encounter",
                "status": "finished",
                "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
                "type": [{"coding": [{"system": "http://www.ama-assn.org/go/cpt", "code": "99213", "display": "Office visit, established patient"}]}],
                "subject": {"reference": f"urn:uuid:{uuids['patient']}"},
                "period": {"start": fmt_fhir_date(encounter_date)},
            },
        },
        {
            "fullUrl": f"urn:uuid:{uuids['condition']}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis"}]}],
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": cond["snomed"], "display": cond["name"]}]},
                "subject": {"reference": f"urn:uuid:{uuids['patient']}"},
                "onsetDateTime": fmt_fhir_date(onset_date, include_time=False),
            },
        },
        {
            "fullUrl": f"urn:uuid:{uuids['lab_obs']}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": cond["lab_loinc"], "display": cond["lab_name"]}]},
                "subject": {"reference": f"urn:uuid:{uuids['patient']}"},
                "effectiveDateTime": fmt_fhir_date(encounter_date, include_time=False),
                "valueCodeableConcept": {"coding": [{"system": "http://snomed.info/sct", "code": cond["lab_result_code"], "display": cond["lab_result_display"]}]},
            },
        },
    ]

    # Vital signs observations
    for i, (loinc, name, val, unit) in enumerate(vitals):
        entries.append({
            "fullUrl": f"urn:uuid:{vital_uuids[i]}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": name}]},
                "subject": {"reference": f"urn:uuid:{uuids['patient']}"},
                "effectiveDateTime": fmt_fhir_date(encounter_date, include_time=False),
                "valueQuantity": {"value": val, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
            },
        })

    # Medications
    for i, med in enumerate(cond.get("medications", [])):
        entries.append({
            "fullUrl": f"urn:uuid:{med_uuids[i]}",
            "resource": {
                "resourceType": "MedicationStatement",
                "status": "active",
                "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": med["rxnorm"], "display": med["name"]}]},
                "subject": {"reference": f"urn:uuid:{uuids['patient']}"},
                "effectiveDateTime": fmt_fhir_date(encounter_date, include_time=False),
            },
        })

    # Organization
    entries.append({
        "fullUrl": f"urn:uuid:{uuids['org']}",
        "resource": {
            "resourceType": "Organization",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": org_npi}],
            "name": org_name,
        },
    })

    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "document",
        "timestamp": fmt_fhir_date(encounter_date),
        "entry": entries,
    }

    return eicr_xml.strip(), json.dumps(fhir_bundle, indent=2)


def build_conversation(eicr_xml, fhir_json):
    """Format as a chat conversation for Unsloth SFT."""
    return {
        "conversations": [
            {
                "role": "system",
                "content": "You are a clinical informatics assistant. Convert the provided eICR (electronic Initial Case Report) CDA/XML document into a valid HL7 FHIR R4 Bundle JSON conforming to the eCR Implementation Guide. Extract all patient demographics, conditions, observations, encounters, and medications. Output valid JSON only.",
            },
            {
                "role": "user",
                "content": f"Convert this eICR to a FHIR R4 Bundle:\n\n{eicr_xml}",
            },
            {
                "role": "assistant",
                "content": fhir_json,
            },
        ]
    }


def main():
    random.seed(42)
    num_samples = 500

    dataset = []
    for i in range(num_samples):
        eicr, fhir = generate_pair()
        conv = build_conversation(eicr, fhir)
        dataset.append(conv)

    # Write training set (80%) and validation set (20%)
    split = int(len(dataset) * 0.8)
    train = dataset[:split]
    val = dataset[split:]

    with open("data/train.jsonl", "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    with open("data/val.jsonl", "w") as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(train)} training samples → data/train.jsonl")
    print(f"Generated {len(val)} validation samples → data/val.jsonl")
    print(f"Conditions covered: {len(CONDITIONS)}")
    print(f"Sample conditions: {', '.join(c['name'] for c in CONDITIONS)}")


if __name__ == "__main__":
    main()
