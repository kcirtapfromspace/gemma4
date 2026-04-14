//! Generate HL7 v2.5.1 ORU^R01 messages from Extraction output.

use chrono::Utc;

use crate::extraction::Extraction;

/// Build an HL7 v2.5.1 ORU^R01 message from the model's Extraction.
pub fn build_hl7_from_extraction(ext: &Extraction) -> String {
    let mut segments = Vec::new();
    let now = Utc::now().format("%Y%m%d%H%M%S").to_string();
    let (given, family) = ext.patient.name_parts();
    let dob_compact = ext.patient.dob.replace('-', "");
    let enc_date_compact = ext.encounter.date.replace('-', "");
    let (city, state, zip) = ext.patient.addr_parts();

    // MSH — Message Header
    segments.push(format!(
        "MSH|^~\\&|ClinIQ|{}|SURVEILLANCE|LHD|{now}||ORU^R01^ORU_R01|{msg_id}|P|2.5.1",
        ext.encounter.facility,
        msg_id = uuid::Uuid::new_v4().to_string().replace('-', "")[..20].to_uppercase(),
    ));

    // PID — Patient Identification
    let phone = ext.patient.phone.as_deref().unwrap_or("");
    let race = ext.patient.race.as_deref().unwrap_or("");
    let ethnicity = ext.patient.ethnicity.as_deref().unwrap_or("");
    segments.push(format!(
        "PID|1||{pid}||{family}^{given}||{dob_compact}|{sex}||{race}|^^^^^{city}^{state}^{zip}^US||{phone}||||||||{ethnicity}",
        pid = uuid::Uuid::new_v4().to_string()[..8].to_uppercase(),
        sex = ext.patient.sex,
    ));

    // PV1 — Patient Visit
    segments.push(format!(
        "PV1|1|O|||||||||||||||||V|{enc_date_compact}",
    ));

    // DG1 — Diagnosis segments
    for (i, c) in ext.conditions.iter().enumerate() {
        segments.push(format!(
            "DG1|{seq}||{snomed}^{name}^SCT||{enc_date_compact}|A",
            seq = i + 1,
            snomed = c.snomed,
            name = c.name,
        ));
    }

    // OBX — Lab observations
    let mut obx_seq = 1;
    for lab in &ext.labs {
        let result_display = if lab.result.is_empty() { "N/A" } else { &lab.result };
        segments.push(format!(
            "OBX|{obx_seq}|CWE|{loinc}^{name}^LN||{result_snomed}^{result}^SCT||||||F|||{date}",
            loinc = lab.loinc,
            name = lab.name,
            result_snomed = lab.result_snomed,
            result = result_display,
            date = lab.date.replace('-', ""),
        ));
        obx_seq += 1;
    }

    // OBX — Vital signs
    if let Some(v) = &ext.vitals {
        let vital_defs: Vec<(&str, &str, Option<f64>, &str)> = vec![
            ("8310-5", "Body temperature", v.temp, "Cel"),
            ("8867-4", "Heart rate", v.hr, "/min"),
            ("9279-1", "Respiratory rate", v.rr, "/min"),
            ("2708-6", "Oxygen saturation", v.spo2, "%"),
            ("8480-6", "Systolic blood pressure", v.sbp, "mm[Hg]"),
        ];
        for (loinc, name, val, unit) in vital_defs {
            if let Some(value) = val {
                segments.push(format!(
                    "OBX|{obx_seq}|NM|{loinc}^{name}^LN||{value}|{unit}|||||F|||{enc_date_compact}",
                ));
                obx_seq += 1;
            }
        }
    }

    // RXA — Medication Administration
    for med in &ext.meds {
        segments.push(format!(
            "RXA|0|1|{enc_date_compact}||{rxnorm}^{name}^RxNorm|1||",
            rxnorm = med.rxnorm,
            name = med.name,
        ));
    }

    segments.join("\r")
}
