use regex::Regex;
use serde::Serialize;
use std::sync::LazyLock;

macro_rules! re {
    ($pat:expr) => {
        LazyLock::new(|| Regex::new($pat).unwrap())
    };
}

static RE_GIVEN: LazyLock<Regex> = re!(r"<given>([^<]+)</given>");
static RE_FAMILY: LazyLock<Regex> = re!(r"<family>([^<]+)</family>");
static RE_GENDER: LazyLock<Regex> = re!(r#"administrativeGenderCode code="([^"]+)""#);
static RE_DOB: LazyLock<Regex> = re!(r#"<birthTime value="(\d{4})(\d{2})(\d{2})"/>"#);
static RE_RACE: LazyLock<Regex> = re!(r#"raceCode[^>]*displayName="([^"]+)""#);
static RE_ETH: LazyLock<Regex> = re!(r#"ethnicGroupCode[^>]*displayName="([^"]+)""#);
static RE_CITY: LazyLock<Regex> = re!(r"<city>([^<]+)</city>");
static RE_STATE: LazyLock<Regex> = re!(r"<state>([^<]+)</state>");
static RE_ZIP: LazyLock<Regex> = re!(r"<postalCode>([^<]+)</postalCode>");
static RE_PHONE: LazyLock<Regex> = re!(r#"telecom value="tel:([^"]+)""#);
static RE_ORG: LazyLock<Regex> =
    re!(r"(?s)representedCustodianOrganization[^>]*>.*?<name>([^<]+)</name>");
static RE_ORG2: LazyLock<Regex> =
    re!(r"(?s)representedOrganization[^>]*>.*?<name>([^<]+)</name>");
static RE_NPI: LazyLock<Regex> =
    re!(r#"<id root="2\.16\.840\.1\.113883\.4\.6" extension="([^"]+)""#);
static RE_ENC_TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<encounter[^>]*>.*?<effectiveTime>.*?<low value="(\d{4})(\d{2})(\d{2})"#)
        .unwrap()
});
static RE_REASON: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)displayName="Reason for visit"[^/]*/>\s*<text>([^<]+)</text>"#).unwrap()
});
static RE_REASON2: LazyLock<Regex> =
    re!(r"<text>Patient presents with ([^<]+)</text>");
static RE_DX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<value[^>]*code="(\d+)"[^>]*codeSystem="2\.16\.840\.1\.113883\.6\.96"[^>]*displayName="([^"]+)""#).unwrap()
});
static RE_LAB: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<code code="([^"]+)"[^>]*codeSystem="2\.16\.840\.1\.113883\.6\.1"[^>]*displayName="([^"]+)""#).unwrap()
});
static RE_VITAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<code code="([^"]+)"[^>]*displayName="([^"]+)"[^/]*/>\s*<statusCode[^/]*/>\s*<effectiveTime[^/]*/>\s*<value[^>]*value="([^"]+)"[^>]*unit="([^"]+)""#).unwrap()
});
static RE_MED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"code="([^"]+)"[^>]*codeSystem="2\.16\.840\.1\.113883\.6\.88"[^>]*displayName="([^"]+)""#).unwrap()
});

const SKIP_DX: &[&str] = &["Detected", "Positive", "Negative", "Not detected", "Salmonella"];
const SKIP_LOINC: &[&str] = &[
    "55751-2", "46240-8", "29299-5", "11450-4", "30954-2", "10160-0",
];

#[derive(Debug, Clone, Serialize, Default)]
pub struct EicrSummary {
    pub patient_name: String,
    pub given_name: String,
    pub family_name: String,
    pub gender: String,
    pub dob: String,
    pub race: String,
    pub ethnicity: String,
    pub city: String,
    pub state: String,
    pub zip: String,
    pub phone: String,
    pub facility: String,
    pub npi: String,
    pub encounter_date: String,
    pub reason: String,
    pub conditions: Vec<EicrCondition>,
    pub labs: Vec<EicrLab>,
    pub vitals: Vec<EicrVital>,
    pub medications: Vec<EicrMed>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EicrCondition {
    pub display: String,
    pub snomed: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EicrLab {
    pub name: String,
    pub loinc: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EicrVital {
    pub loinc: String,
    pub name: String,
    pub value: String,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EicrMed {
    pub name: String,
    pub rxnorm: String,
}

impl EicrSummary {
    /// Format as the text summary that matches training data input format.
    pub fn to_prompt_text(&self) -> String {
        let mut parts = Vec::new();

        if !self.patient_name.is_empty() {
            parts.push(format!("Patient: {}", self.patient_name));
        }
        if !self.gender.is_empty() {
            parts.push(format!("Gender: {}", self.gender));
        }
        if !self.dob.is_empty() {
            parts.push(format!("DOB: {}", self.dob));
        }
        if !self.race.is_empty() {
            parts.push(format!("Race: {}", self.race));
        }
        if !self.ethnicity.is_empty() {
            parts.push(format!("Ethnicity: {}", self.ethnicity));
        }
        if !self.city.is_empty() && !self.state.is_empty() {
            let mut loc = format!("{}, {}", self.city, self.state);
            if !self.zip.is_empty() {
                loc.push(' ');
                loc.push_str(&self.zip);
            }
            parts.push(format!("Location: {loc}"));
        }
        if !self.phone.is_empty() {
            parts.push(format!("Phone: {}", self.phone));
        }
        if !self.facility.is_empty() {
            let mut fac = format!("Facility: {}", self.facility);
            if !self.npi.is_empty() {
                fac.push_str(&format!(" (NPI: {})", self.npi));
            }
            parts.push(fac);
        }
        if !self.encounter_date.is_empty() {
            parts.push(format!("Encounter: {}", self.encounter_date));
        }
        if !self.reason.is_empty() {
            parts.push(format!("Reason: {}", self.reason));
        }
        for c in &self.conditions {
            parts.push(format!("Dx: {} (SNOMED {})", c.display, c.snomed));
        }
        for l in &self.labs {
            let mut line = format!("Lab: {} (LOINC {})", l.name, l.loinc);
            if !l.result.is_empty() {
                line.push_str(&format!(" - {}", l.result));
            }
            parts.push(line);
        }
        if !self.vitals.is_empty() {
            let labels = [
                ("8310-5", "Temp", "C"),
                ("8867-4", "HR", ""),
                ("9279-1", "RR", ""),
                ("2708-6", "SpO2", "%"),
                ("8480-6", "BP", ""),
            ];
            let mut vparts = Vec::new();
            for (loinc, label, suffix) in labels {
                if let Some(v) = self.vitals.iter().find(|v| v.loinc == loinc) {
                    vparts.push(format!("{label} {}{suffix}", v.value));
                }
            }
            if !vparts.is_empty() {
                parts.push(format!("Vitals: {}", vparts.join(", ")));
            }
        }
        for m in &self.medications {
            parts.push(format!("Meds: {} (RxNorm {})", m.name, m.rxnorm));
        }

        if parts.is_empty() {
            return String::new();
        }
        parts.join("\n")
    }
}

/// Extract structured data from eICR CDA/XML.
pub fn extract(xml: &str) -> EicrSummary {
    let mut s = EicrSummary::default();

    // Patient name
    let given = RE_GIVEN.captures(xml).map(|c| c[1].to_string());
    let family = RE_FAMILY.captures(xml).map(|c| c[1].to_string());
    if let (Some(g), Some(f)) = (&given, &family) {
        s.patient_name = format!("{g} {f}");
        s.given_name = g.clone();
        s.family_name = f.clone();
    }

    if let Some(m) = RE_GENDER.captures(xml) {
        s.gender = m[1].to_string();
    }
    if let Some(m) = RE_DOB.captures(xml) {
        s.dob = format!("{}-{}-{}", &m[1], &m[2], &m[3]);
    }
    if let Some(m) = RE_RACE.captures(xml) {
        s.race = m[1].to_string();
    }
    if let Some(m) = RE_ETH.captures(xml) {
        s.ethnicity = m[1].to_string();
    }
    if let Some(c) = RE_CITY.captures(xml) {
        s.city = c[1].to_string();
    }
    if let Some(m) = RE_STATE.captures(xml) {
        s.state = m[1].to_string();
    }
    if let Some(m) = RE_ZIP.captures(xml) {
        s.zip = m[1].to_string();
    }
    if let Some(m) = RE_PHONE.captures(xml) {
        s.phone = m[1].to_string();
    }

    // Facility + NPI
    let org = RE_ORG.captures(xml).or_else(|| RE_ORG2.captures(xml));
    if let Some(o) = org {
        s.facility = o[1].to_string();
    }
    if let Some(n) = RE_NPI.captures(xml) {
        s.npi = n[1].to_string();
    }

    // Encounter date
    if let Some(m) = RE_ENC_TIME.captures(xml) {
        s.encounter_date = format!("{}-{}-{}", &m[1], &m[2], &m[3]);
    }

    // Reason for visit
    if let Some(m) = RE_REASON.captures(xml).or_else(|| RE_REASON2.captures(xml)) {
        let reason = &m[1];
        s.reason = if reason.len() > 150 {
            reason[..150].to_string()
        } else {
            reason.to_string()
        };
    }

    // Diagnoses (SNOMED)
    for m in RE_DX.captures_iter(xml) {
        let display = m[2].to_string();
        if !SKIP_DX.contains(&display.as_str()) {
            s.conditions.push(EicrCondition {
                snomed: m[1].to_string(),
                display,
            });
        }
    }

    // Lab results (LOINC)
    for m in RE_LAB.captures_iter(xml) {
        let code = &m[1];
        if SKIP_LOINC.contains(&code) {
            continue;
        }
        let result_re = Regex::new(&format!(
            r#"(?s)code="{}".*?<value[^>]*displayName="([^"]+)""#,
            regex::escape(code)
        ));
        let result = result_re
            .ok()
            .and_then(|r| r.captures(xml))
            .map(|c| c[1].to_string())
            .unwrap_or_default();
        s.labs.push(EicrLab {
            name: m[2].to_string(),
            loinc: code.to_string(),
            result,
        });
    }

    // Vitals
    for m in RE_VITAL.captures_iter(xml) {
        s.vitals.push(EicrVital {
            loinc: m[1].to_string(),
            name: m[2].to_string(),
            value: m[3].to_string(),
            unit: m[4].to_string(),
        });
    }

    // Medications (RxNorm)
    for m in RE_MED.captures_iter(xml) {
        s.medications.push(EicrMed {
            name: m[2].to_string(),
            rxnorm: m[1].to_string(),
        });
    }

    s
}
