// SpeechDictationService.swift
// On-device clinical speech-to-text. Mirrors EpiCast's "describe in your
// own language" pattern — clinician dictates, the phone transcribes
// locally, the resulting narrative feeds the same ExtractionService that
// handles typed input. Audio never leaves the device.
//
// Architecture choice: Apple's Speech framework with
// requiresOnDeviceRecognition=true (iOS 13+, fully offline on supported
// locales since iOS 15). We could have used Gemma 4 E4B's native audio
// path, but the iOS Speech framework ships the recognizer in OS, and the
// extraction pipeline is downstream of the text — a dedicated ASR + the
// existing agent stack is exactly the pattern that won the MedGemma
// Impact Challenge (FieldScreen AI = MedGemma + MedASR + HeAR).
//
// Live partial transcripts stream into the UI; the final transcript is
// dispatched to the same path as typed input on the user releasing the
// record button.

import Foundation
import AVFoundation
import Speech

@MainActor
final class SpeechDictationService: NSObject, ObservableObject {
    /// Live transcription updated as the recognizer streams partial results.
    @Published private(set) var liveTranscript: String = ""
    /// True between startRecording() and the recognizer finishing.
    @Published private(set) var isRecording: Bool = false
    /// Surface failures so the UI can show "permission denied" / "device
    /// not supported" states without crashing.
    @Published private(set) var errorMessage: String?
    /// Authorization state, mirrored to the UI for the permission banner.
    @Published private(set) var authorizationStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined
    /// Locale of the active recognizer. Default en-US — same as the bench.
    @Published private(set) var localeIdentifier: String = "en-US"
    /// Audio level (RMS, 0.0–1.0) for the live waveform visualization.
    @Published private(set) var audioLevel: Float = 0.0

    private var recognizer: SFSpeechRecognizer?
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    override init() {
        super.init()
        self.recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
    }

    /// Returns true if the device supports Speech recognition AND on-device
    /// processing is available for our locale. Field-clinic deployment
    /// requires both — we don't ship cloud recognition.
    var isSupported: Bool {
        guard let r = recognizer, r.isAvailable else { return false }
        return r.supportsOnDeviceRecognition
    }

    /// Request microphone + speech-recognition permissions. Resolves with
    /// the final speech-recognition status (microphone failure shows up
    /// as `.denied` alongside an errorMessage).
    func requestAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        let speechStatus: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status)
            }
        }
        let micGranted: Bool = await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { granted in
                cont.resume(returning: granted)
            }
        }
        if !micGranted {
            errorMessage = "Microphone access denied"
        }
        authorizationStatus = speechStatus
        return speechStatus
    }

    /// Begin streaming recognition. The buffer feeds Speech framework as
    /// audio arrives; partials populate liveTranscript. Call stopRecording()
    /// to commit. Throws on permission / hardware failure.
    func startRecording() async throws {
        guard !isRecording else { return }
        guard let recognizer = recognizer, recognizer.isAvailable else {
            throw NSError(domain: "ClinIQ.Speech", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Speech recognizer unavailable for \(localeIdentifier)"])
        }
        if authorizationStatus != .authorized {
            let status = await requestAuthorization()
            guard status == .authorized else {
                throw NSError(domain: "ClinIQ.Speech", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Speech recognition not authorized"])
            }
        }

        // Configure audio session for record + playAndRecord with
        // .measurement mode — the recognizer prefers raw mic input without
        // signal-conditioning that would distort medical terminology.
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        // Add medical-context bias so common reportable conditions get
        // higher prior than the recognizer's default LM.
        request.contextualStrings = Self.medicalContextHints
        recognitionRequest = request

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            request.append(buffer)
            // Compute simple RMS for the level meter.
            guard let channels = buffer.floatChannelData else { return }
            let frames = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<frames { sum += channels[0][i] * channels[0][i] }
            let rms = frames > 0 ? sqrt(sum / Float(frames)) : 0
            let level = min(max(rms * 4.0, 0.0), 1.0)
            Task { @MainActor in self?.audioLevel = level }
        }
        engine.prepare()
        try engine.start()
        audioEngine = engine

        liveTranscript = ""
        errorMessage = nil
        isRecording = true

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self = self else { return }
                if let result = result {
                    self.liveTranscript = result.bestTranscription.formattedString
                    if result.isFinal { self.finishRecognition() }
                }
                if let error = error {
                    self.errorMessage = "Speech: \(error.localizedDescription)"
                    self.finishRecognition()
                }
            }
        }
    }

    /// Stop capturing audio, let the recognizer flush, then return the final
    /// transcript. Safe to call when not recording (returns the last value).
    @discardableResult
    func stopRecording() async -> String {
        guard isRecording else { return liveTranscript }
        recognitionRequest?.endAudio()
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        // Wait briefly for the recognizer to emit the final result.
        let deadline = Date().addingTimeInterval(2.0)
        while recognitionTask != nil, Date() < deadline {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        finishRecognition()
        return liveTranscript
    }

    private func finishRecognition() {
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        isRecording = false
        audioLevel = 0
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Common medical / public-health terminology priors. The Speech
    /// framework biases its language model toward these strings, which
    /// dramatically reduces dictation errors on disease names + drugs.
    /// Mirror the lookup table + RAG db for now; expand as the bench grows.
    static let medicalContextHints: [String] = [
        // Reportable conditions
        "tuberculosis", "TB", "pertussis", "whooping cough",
        "measles", "mumps", "rubella", "diphtheria", "tetanus",
        "syphilis", "gonorrhea", "chlamydia", "HIV", "hepatitis B", "hepatitis C",
        "Lyme disease", "West Nile", "Zika", "dengue", "chikungunya",
        "Salmonella", "Shigella", "Listeria", "Cryptosporidium",
        "Legionnaires disease", "Legionellosis",
        "Rocky Mountain spotted fever", "RMSF",
        "valley fever", "coccidioidomycosis",
        "C diff", "Clostridioides difficile",
        "MRSA", "VRE", "CRE",
        "mpox", "monkeypox",
        "RSV", "respiratory syncytial virus",
        "COVID-19", "SARS-CoV-2",
        "Ebola", "Marburg", "MERS",
        // Drug classes
        "amoxicillin", "doxycycline", "azithromycin", "ceftriaxone",
        "isoniazid", "rifampin", "ethambutol", "pyrazinamide",
        "fluconazole", "acyclovir", "oseltamivir", "tecovirimat",
        "ciprofloxacin", "vancomycin", "nirmatrelvir", "ritonavir",
        // Lab instruments
        "PCR", "EIA", "NAA", "RT-PCR", "IgM", "IgG", "antigen",
    ]
}
