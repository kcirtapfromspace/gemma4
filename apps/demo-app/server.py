"""
Demo app: eICR → FHIR Converter powered by fine-tuned Gemma 4 E4B on Jetson Orin Nano.
Provides a web UI for pasting eICR CDA/XML and getting back FHIR R4 JSON.
Talks to the Ollama API running on the same k8s cluster.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama.gemma4.svc:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma4-eicr-fhir")
PORT = int(os.environ.get("PORT", "8080"))

SYSTEM_PROMPT = (
    "You are a clinical informatics assistant. Convert the provided eICR "
    "(electronic Initial Case Report) CDA/XML document into a valid HL7 FHIR R4 "
    "Bundle JSON conforming to the eCR Implementation Guide. Extract all patient "
    "demographics, conditions, observations, encounters, and medications. "
    "Output valid JSON only."
)


def call_ollama(eicr_xml: str) -> str:
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert this eICR to a FHIR R4 Bundle:\n\n{eicr_xml}"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 4096},
    }).encode()

    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        return result["message"]["content"]


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>eICR to FHIR Converter | Gemma 4 on Jetson</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f1117; color: #e1e4e8; min-height: 100vh; }
  .header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .badge { background: #238636; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
  .header .badge.hw { background: #1f6feb; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: calc(100vh - 65px); }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
  .panel-header { padding: 12px 16px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
  .panel-header h2 { font-size: 14px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  textarea { flex: 1; background: #0d1117; color: #c9d1d9; border: none; padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; resize: none; outline: none; line-height: 1.5; }
  pre { flex: 1; background: #0d1117; color: #c9d1d9; padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; overflow: auto; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .actions { padding: 16px; display: flex; gap: 8px; justify-content: center; }
  button { background: #238636; color: #fff; border: none; padding: 10px 24px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
  button:hover { background: #2ea043; }
  button:disabled { background: #30363d; color: #8b949e; cursor: not-allowed; }
  button.secondary { background: #30363d; color: #c9d1d9; }
  button.secondary:hover { background: #3d444d; }
  .status { text-align: center; padding: 8px; font-size: 13px; color: #8b949e; }
  .status.error { color: #f85149; }
  .status.success { color: #3fb950; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #30363d; border-top-color: #238636; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .info-bar { background: #161b22; border-top: 1px solid #30363d; padding: 8px 24px; display: flex; gap: 24px; font-size: 12px; color: #8b949e; }
  .info-bar span { display: flex; align-items: center; gap: 4px; }
  @media (max-width: 768px) { .container { grid-template-columns: 1fr; height: auto; } .panel { min-height: 300px; } }
</style>
</head>
<body>
<div class="header">
  <h1>eICR to FHIR R4 Converter</h1>
  <span class="badge">Gemma 4 E4B</span>
  <span class="badge hw">Jetson Orin Nano</span>
</div>
<div class="container">
  <div class="panel">
    <div class="panel-header">
      <h2>Input: eICR CDA/XML</h2>
      <button class="secondary" onclick="loadSample()">Load Sample</button>
    </div>
    <textarea id="input" placeholder="Paste your eICR CDA/XML document here..."></textarea>
  </div>
  <div class="panel">
    <div class="panel-header">
      <h2>Output: FHIR R4 Bundle</h2>
      <button class="secondary" onclick="copyOutput()">Copy</button>
    </div>
    <pre id="output">Output will appear here after conversion...</pre>
  </div>
</div>
<div class="actions">
  <button id="convertBtn" onclick="convert()">Convert to FHIR</button>
</div>
<div id="status" class="status"></div>
<div class="info-bar">
  <span>Model: gemma4-eicr-fhir (E4B Q4_K_M)</span>
  <span>Hardware: NVIDIA Jetson Orin Nano 8GB</span>
  <span>Runtime: Ollama + llama.cpp</span>
  <span>Hackathon: Gemma 4 Good x Unsloth</span>
</div>
<script>
const SAMPLE = `<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <realmCode code="US"/>
  <typeId root="2.16.840.1.113883.1.3" extension="POCD_HD000040"/>
  <templateId root="2.16.840.1.113883.10.20.15.2" extension="2021-01-01"/>
  <id root="sample-doc-001"/>
  <code code="55751-2" codeSystem="2.16.840.1.113883.6.1" displayName="Public Health Case Report"/>
  <title>Initial Public Health Case Report - eICR</title>
  <effectiveTime value="20260315120000-0600"/>
  <recordTarget>
    <patientRole>
      <id extension="PT-12345" root="2.16.840.1.113883.19.5"/>
      <addr use="H">
        <streetAddressLine>456 Oak Ave</streetAddressLine>
        <city>Denver</city><state>CO</state><postalCode>80202</postalCode><country>US</country>
      </addr>
      <patient>
        <name use="L"><given>Maria</given><family>Garcia</family></name>
        <administrativeGenderCode code="F" codeSystem="2.16.840.1.113883.5.1"/>
        <birthTime value="19850614"/>
      </patient>
    </patientRole>
  </recordTarget>
  <component>
    <structuredBody>
      <component>
        <section>
          <code code="29299-5" codeSystem="2.16.840.1.113883.6.1" displayName="Reason for visit"/>
          <text>Patient presents with fever (39.2C), dry cough for 5 days, and shortness of breath.</text>
        </section>
      </component>
      <component>
        <section>
          <code code="11450-4" codeSystem="2.16.840.1.113883.6.1" displayName="Problem list"/>
          <entry>
            <act classCode="ACT" moodCode="EVN">
              <entryRelationship typeCode="SUBJ">
                <observation classCode="OBS" moodCode="EVN">
                  <value xsi:type="CD" code="840539006" codeSystem="2.16.840.1.113883.6.96" displayName="COVID-19"/>
                  <effectiveTime><low value="20260310"/></effectiveTime>
                </observation>
              </entryRelationship>
            </act>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <code code="30954-2" codeSystem="2.16.840.1.113883.6.1" displayName="Relevant diagnostic tests/laboratory data"/>
          <entry>
            <organizer classCode="BATTERY" moodCode="EVN">
              <component>
                <observation classCode="OBS" moodCode="EVN">
                  <code code="94500-6" codeSystem="2.16.840.1.113883.6.1" displayName="SARS-CoV-2 RNA NAA+probe Ql (Resp)"/>
                  <value xsi:type="CD" code="260373001" codeSystem="2.16.840.1.113883.6.96" displayName="Detected"/>
                </observation>
              </component>
            </organizer>
          </entry>
        </section>
      </component>
    </structuredBody>
  </component>
</ClinicalDocument>`;

function loadSample() { document.getElementById('input').value = SAMPLE; }

function setStatus(text, className) {
  var el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status' + (className ? ' ' + className : '');
}

function showSpinnerStatus(text) {
  var el = document.getElementById('status');
  el.className = 'status';
  // Clear and rebuild safely
  while (el.firstChild) el.removeChild(el.firstChild);
  var spinner = document.createElement('span');
  spinner.className = 'spinner';
  el.appendChild(spinner);
  el.appendChild(document.createTextNode(text));
}

async function convert() {
  var input = document.getElementById('input').value.trim();
  if (!input) return;
  var btn = document.getElementById('convertBtn');
  var output = document.getElementById('output');
  btn.disabled = true;
  showSpinnerStatus('Converting with Gemma 4 on Jetson...');
  output.textContent = 'Processing...';
  var start = Date.now();
  try {
    var resp = await fetch('/api/convert', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({eicr: input}),
    });
    var data = await resp.json();
    var elapsed = ((Date.now() - start) / 1000).toFixed(1);
    if (data.error) {
      setStatus('Error: ' + data.error, 'error');
      output.textContent = data.error;
    } else {
      try { output.textContent = JSON.stringify(JSON.parse(data.fhir), null, 2); }
      catch(e) { output.textContent = data.fhir; }
      setStatus('Converted in ' + elapsed + 's on Jetson Orin Nano', 'success');
    }
  } catch (e) {
    setStatus('Connection error: ' + e.message, 'error');
    output.textContent = 'Failed to reach the model API. Check Ollama status.';
  }
  btn.disabled = false;
}

function copyOutput() {
  var text = document.getElementById('output').textContent;
  navigator.clipboard.writeText(text);
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        if self.path != "/api/convert":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        eicr_xml = body.get("eicr", "")

        try:
            fhir_json = call_ollama(eicr_xml)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"fhir": fhir_json}).encode())
        except URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Ollama unreachable: {e}"}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"eICR to FHIR Demo running on http://0.0.0.0:{PORT}")
    server.serve_forever()
