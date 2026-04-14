//! HL7 v2.5.1 pipe-delimited message parser.
//!
//! Parses relevant segments (MSH, PID, PV1, DG1, OBX, RXA) and formats
//! them as prompt text matching `EicrSummary::to_prompt_text()` output
//! so the fine-tuned model can process them identically to eICR input.

use std::collections::HashMap;

/// Parse HL7 v2.5.1 message into prompt text for the LLM.
pub fn format_prompt(input: &str) -> Result<String, String> {
    let segments = parse_segments(input);
    let mut parts = Vec::new();

    // PID — Patient Identification
    if let Some(pid_list) = segments.get("PID") {
        if let Some(pid) = pid_list.first() {
            // PID-5: Patient Name (Family^Given^Middle^^^)
            if let Some(name_field) = pid.get(5) {
                let comps: Vec<&str> = name_field.split('^').collect();
                let family = comps.first().unwrap_or(&"");
                let given = comps.get(1).unwrap_or(&"");
                if !family.is_empty() || !given.is_empty() {
                    parts.push(format!("Patient: {given} {family}"));
                }
            }
            // PID-8: Administrative Sex
            if let Some(sex) = pid.get(8) {
                if !sex.is_empty() {
                    parts.push(format!("Gender: {sex}"));
                }
            }
            // PID-7: Date of Birth (YYYYMMDD)
            if let Some(dob_raw) = pid.get(7) {
                if dob_raw.len() >= 8 {
                    let dob = format!("{}-{}-{}", &dob_raw[..4], &dob_raw[4..6], &dob_raw[6..8]);
                    parts.push(format!("DOB: {dob}"));
                }
            }
            // PID-10: Race
            if let Some(race_field) = pid.get(10) {
                let race_text = race_field.split('^').nth(1).unwrap_or(race_field);
                if !race_text.is_empty() {
                    parts.push(format!("Race: {race_text}"));
                }
            }
            // PID-22: Ethnic Group
            if let Some(eth_field) = pid.get(22) {
                let eth_text = eth_field.split('^').nth(1).unwrap_or(eth_field);
                if !eth_text.is_empty() {
                    parts.push(format!("Ethnicity: {eth_text}"));
                }
            }
            // PID-11: Patient Address (Street^City^State^ZIP^Country)
            if let Some(addr_field) = pid.get(11) {
                let comps: Vec<&str> = addr_field.split('^').collect();
                let city = comps.get(2).unwrap_or(&"");
                let state = comps.get(3).unwrap_or(&"");
                let zip = comps.get(4).unwrap_or(&"");
                if !city.is_empty() && !state.is_empty() {
                    let mut loc = format!("{city}, {state}");
                    if !zip.is_empty() {
                        loc.push(' ');
                        loc.push_str(zip);
                    }
                    parts.push(format!("Location: {loc}"));
                }
            }
            // PID-13: Phone Number
            if let Some(phone) = pid.get(13) {
                let phone_num = phone.split('^').next().unwrap_or(phone);
                if !phone_num.is_empty() {
                    parts.push(format!("Phone: {phone_num}"));
                }
            }
        }
    }

    // MSH — Message Header (facility info)
    if let Some(msh_list) = segments.get("MSH") {
        if let Some(msh) = msh_list.first() {
            // MSH-4: Sending Facility
            if let Some(fac_field) = msh.get(4) {
                let fac_name = fac_field.split('^').next().unwrap_or(fac_field);
                if !fac_name.is_empty() {
                    // MSH-6 might have NPI in some implementations
                    let npi = msh
                        .get(6)
                        .and_then(|f| {
                            let n = f.split('^').next().unwrap_or("");
                            if n.len() == 10 && n.chars().all(|c| c.is_ascii_digit()) {
                                Some(n)
                            } else {
                                None
                            }
                        });
                    let mut fac = format!("Facility: {fac_name}");
                    if let Some(npi_val) = npi {
                        fac.push_str(&format!(" (NPI: {npi_val})"));
                    }
                    parts.push(fac);
                }
            }
            // MSH-7: Message Date/Time (encounter date proxy)
            if let Some(dt_raw) = msh.get(7) {
                if dt_raw.len() >= 8 {
                    let enc = format!("{}-{}-{}", &dt_raw[..4], &dt_raw[4..6], &dt_raw[6..8]);
                    parts.push(format!("Encounter: {enc}"));
                }
            }
        }
    }

    // DG1 — Diagnosis
    if let Some(dg1_list) = segments.get("DG1") {
        for dg1 in dg1_list {
            // DG1-3: Diagnosis Code (Code^Text^CodingSystem)
            if let Some(dx_field) = dg1.get(3) {
                let comps: Vec<&str> = dx_field.split('^').collect();
                let code = comps.first().unwrap_or(&"");
                let text = comps.get(1).unwrap_or(&"");
                let system = comps.get(2).map(|s| s.to_uppercase()).unwrap_or_default();

                if !code.is_empty() && !text.is_empty() {
                    let sys_label = if system.contains("SNO") || system == "SCT" {
                        "SNOMED"
                    } else if system.contains("ICD") || system == "I10" {
                        "ICD-10"
                    } else {
                        "SNOMED" // default
                    };
                    parts.push(format!("Dx: {text} ({sys_label} {code})"));
                }
            }
        }
    }

    // OBX — Observations (labs + vitals)
    if let Some(obx_list) = segments.get("OBX") {
        let mut vital_parts = Vec::new();

        for obx in obx_list {
            // OBX-3: Observation Identifier (Code^Text^CodingSystem)
            // OBX-5: Observation Value
            // OBX-6: Units
            let obs_id = obx.get(3).unwrap_or(&String::new()).clone();
            let obs_val = obx.get(5).unwrap_or(&String::new()).clone();
            let _obs_unit = obx.get(6).unwrap_or(&String::new()).clone();

            let comps: Vec<&str> = obs_id.split('^').collect();
            let code = comps.first().unwrap_or(&"");
            let name = comps.get(1).unwrap_or(&"");
            let system = comps.get(2).map(|s| s.to_uppercase()).unwrap_or_default();

            if code.is_empty() || name.is_empty() {
                continue;
            }

            let is_loinc = system.contains("LN") || system.contains("LOINC");

            // Check if this is a vital sign by LOINC code
            let vital_label = match *code {
                "8310-5" => Some(("Temp", "C")),
                "8867-4" => Some(("HR", "")),
                "9279-1" => Some(("RR", "")),
                "2708-6" => Some(("SpO2", "%")),
                "8480-6" => Some(("BP", "")),
                _ => None,
            };

            if let Some((label, suffix)) = vital_label {
                let val_text = obs_val.split('^').next().unwrap_or(&obs_val);
                vital_parts.push(format!("{label} {val_text}{suffix}"));
            } else {
                // Lab result
                let result_text = obs_val.split('^').nth(1).unwrap_or(
                    obs_val.split('^').next().unwrap_or(&obs_val),
                );
                let sys_label = if is_loinc { "LOINC" } else { "LOINC" };
                let mut line = format!("Lab: {name} ({sys_label} {code})");
                if !result_text.is_empty() {
                    line.push_str(&format!(" - {result_text}"));
                }
                parts.push(line);
            }
        }

        if !vital_parts.is_empty() {
            parts.push(format!("Vitals: {}", vital_parts.join(", ")));
        }
    }

    // RXA — Pharmacy/Treatment Administration
    if let Some(rxa_list) = segments.get("RXA") {
        for rxa in rxa_list {
            // RXA-5: Administered Code (Code^Text^CodingSystem)
            if let Some(med_field) = rxa.get(5) {
                let comps: Vec<&str> = med_field.split('^').collect();
                let code = comps.first().unwrap_or(&"");
                let name = comps.get(1).unwrap_or(&"");
                if !code.is_empty() && !name.is_empty() {
                    parts.push(format!("Meds: {name} (RxNorm {code})"));
                }
            }
        }
    }

    // RXE — Pharmacy/Treatment Encoded Order (fallback if no RXA)
    if !segments.contains_key("RXA") {
        if let Some(rxe_list) = segments.get("RXE") {
            for rxe in rxe_list {
                // RXE-2: Give Code (Code^Text^CodingSystem)
                if let Some(med_field) = rxe.get(2) {
                    let comps: Vec<&str> = med_field.split('^').collect();
                    let code = comps.first().unwrap_or(&"");
                    let name = comps.get(1).unwrap_or(&"");
                    if !code.is_empty() && !name.is_empty() {
                        parts.push(format!("Meds: {name} (RxNorm {code})"));
                    }
                }
            }
        }
    }

    if parts.is_empty() {
        Err("Could not extract any data from HL7 v2.5.1 message".into())
    } else {
        Ok(parts.join("\n"))
    }
}

/// Parse HL7 message into a map of segment type -> list of segment field arrays.
/// Handles both `\r` and `\n` as segment separators.
fn parse_segments(input: &str) -> HashMap<String, Vec<Vec<String>>> {
    let mut result: HashMap<String, Vec<Vec<String>>> = HashMap::new();

    // HL7 uses \r as segment separator, but accept \n too
    let lines: Vec<&str> = input.split(|c| c == '\r' || c == '\n')
        .filter(|l| !l.is_empty())
        .collect();

    for line in lines {
        let fields: Vec<String> = line.split('|').map(|s| s.to_string()).collect();
        if let Some(seg_type) = fields.first() {
            let key = seg_type.to_uppercase();
            // For MSH, the first separator IS field 1, so fields are offset by 1
            // We handle this by just using the raw split — callers adjust indices
            result.entry(key).or_default().push(fields);
        }
    }

    result
}
