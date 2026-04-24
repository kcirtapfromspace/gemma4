// CaseDetailView.swift
// Shown when the clinician taps a row in the Cases or History list.
// Summarises the case with chips and rows — no raw JSON — and exposes
// status-appropriate actions (review, queue to outbox, retry sync).

import SwiftUI
import SwiftData

struct CaseDetailView: View {
    @Bindable var clinicalCase: ClinicalCase
    @EnvironmentObject private var sync: SyncService
    @Environment(\.modelContext) private var context

    @State private var reviewOpen = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                if let p = clinicalCase.patient {
                    patientCard(p)
                }
                if !clinicalCase.conditions.isEmpty {
                    sectionCard(title: "Conditions") {
                        ForEach(clinicalCase.conditions.sorted(by: { $0.displayName < $1.displayName })) { c in
                            ConditionRow(condition: c)
                            if c.id != clinicalCase.conditions.last?.id {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }
                if !clinicalCase.labs.isEmpty {
                    sectionCard(title: "Labs") {
                        ForEach(clinicalCase.labs.sorted(by: { $0.displayName < $1.displayName })) { lab in
                            LabRow(lab: lab)
                            if lab.id != clinicalCase.labs.last?.id {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }
                if !clinicalCase.medications.isEmpty {
                    sectionCard(title: "Medications") {
                        ForEach(clinicalCase.medications.sorted(by: { $0.displayName < $1.displayName })) { m in
                            MedicationRow(medication: m)
                            if m.id != clinicalCase.medications.last?.id {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }
                if let v = clinicalCase.vitals, !v.isEmpty {
                    sectionCard(title: "Vitals") {
                        VitalsGrid(vitals: v)
                    }
                }
                narrativeCard
                if !clinicalCase.syncHistory.isEmpty {
                    sectionCard(title: "Submission history") {
                        ForEach(clinicalCase.syncHistory.sorted(by: { $0.attemptedAt > $1.attemptedAt })) { r in
                            SyncHistoryRow(record: r)
                            if r.id != clinicalCase.syncHistory.last?.id {
                                Divider().padding(.vertical, 2)
                            }
                        }
                    }
                }
            }
            .padding(16)
        }
        .background(ClinIQTheme.pageBackground)
        .navigationTitle(clinicalCase.patient?.fullName ?? "Case")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                actionMenu
            }
        }
        .sheet(isPresented: $reviewOpen) {
            ReviewFlowView(clinicalCase: clinicalCase)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center) {
                Text(clinicalCase.primaryConditionDisplay)
                    .font(.title3.weight(.semibold))
                Spacer()
                StatusBadge(status: clinicalCase.status)
            }
            Text("Opened \(clinicalCase.createdAt.formatted(date: .abbreviated, time: .shortened))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func patientCard(_ p: Patient) -> some View {
        sectionCard(title: "Patient") {
            VStack(alignment: .leading, spacing: 6) {
                Text(p.fullName).font(.body.weight(.medium))
                HStack(spacing: 14) {
                    Label(p.genderDisplay, systemImage: "person")
                    Label(p.ageDescription, systemImage: "calendar")
                    if let pc = p.postalCode, !pc.isEmpty {
                        Label(pc, systemImage: "mappin.and.ellipse")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                if let f = p.facilityName {
                    Label(f, systemImage: "building.2")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var narrativeCard: some View {
        sectionCard(title: "Narrative") {
            Text(clinicalCase.narrative.isEmpty ? "No narrative captured." : clinicalCase.narrative)
                .font(.callout)
                .foregroundStyle(.primary)
                .lineSpacing(2)
        }
    }

    private var actionMenu: some View {
        Menu {
            Button {
                reviewOpen = true
            } label: {
                Label("Review with AI", systemImage: "wand.and.stars")
            }
            if clinicalCase.status == .draft {
                Button {
                    sync.queue(clinicalCase)
                } label: {
                    Label("Queue to Outbox", systemImage: "tray.and.arrow.up")
                }
            }
            if clinicalCase.status == .failed {
                Button {
                    sync.queue(clinicalCase)
                } label: {
                    Label("Retry sync", systemImage: "arrow.triangle.2.circlepath")
                }
            }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
    }

    @ViewBuilder
    private func sectionCard<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct VitalsGrid: View {
    let vitals: Vitals

    var body: some View {
        let columns: [GridItem] = [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())]
        LazyVGrid(columns: columns, spacing: 10) {
            if let t = vitals.tempC {
                VitalCell(label: "Temp", value: String(format: "%.1f C", t), icon: "thermometer.medium")
            }
            if let h = vitals.heartRate {
                VitalCell(label: "HR", value: "\(h) bpm", icon: "heart")
            }
            if let r = vitals.respRate {
                VitalCell(label: "RR", value: "\(r) /min", icon: "wind")
            }
            if let s = vitals.spo2 {
                VitalCell(label: "SpO₂", value: "\(s)%", icon: "lungs")
            }
            if let b = vitals.bpSystolic {
                VitalCell(label: "BP", value: "\(b)\(vitals.bpDiastolic.map { "/\($0)" } ?? "")", icon: "gauge")
            }
        }
    }
}

struct VitalCell: View {
    let label: String
    let value: String
    let icon: String

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(value)
                .font(.callout.weight(.semibold))
                .foregroundStyle(.primary)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color(.tertiarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct SyncHistoryRow: View {
    let record: SyncRecord

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: record.succeeded ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(record.succeeded ? Color(red: 0.13, green: 0.36, blue: 0.22)
                                                  : Color(red: 0.66, green: 0.15, blue: 0.12))
            VStack(alignment: .leading, spacing: 2) {
                Text(record.attemptedAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.callout.weight(.medium))
                if let m = record.message {
                    Text(m)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(record.endpoint)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(ClinIQTheme.auditMuted)
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }
}
