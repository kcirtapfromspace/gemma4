/// Map state abbreviation to local health department name.
/// Returns (jurisdiction_name, is_out_of_state).
pub fn route(state: &str) -> (&'static str, bool) {
    match state.to_uppercase().as_str() {
        "CO" => ("Colorado DPHE", false),
        "IL" => ("Illinois DPH", false),
        "TX" => ("Texas DSHS", false),
        "AZ" => ("Arizona DHS", false),
        "WA" => ("Washington DOH", false),
        "GA" => ("Georgia DPH", false),
        "MA" => ("Massachusetts DPH", false),
        "FL" => ("Florida DOH", false),
        "OR" => ("Oregon OHA", false),
        "MN" => ("Minnesota DOH", false),
        "UT" => ("Utah DOH", false),
        "CA" => ("California CDPH", false),
        "NY" => ("New York DOH", false),
        "PA" => ("Pennsylvania DOH", false),
        "OH" => ("Ohio ODH", false),
        _ => ("Unknown", true),
    }
}
