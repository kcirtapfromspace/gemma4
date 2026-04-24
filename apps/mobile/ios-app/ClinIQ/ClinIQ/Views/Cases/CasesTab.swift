// CasesTab.swift
// Top-level tab: list of active cases + entry point to create a new case.
// List filters out cases the clinician has already seen through to
// "submitted" more than a day ago — those live in History.

import SwiftUI
import SwiftData

struct CasesTab: View {
    @Environment(\.modelContext) private var context
    @Query(sort: [SortDescriptor(\ClinicalCase.createdAt, order: .reverse)])
    private var allCases: [ClinicalCase]

    @State private var presentingNewCase = false
    @State private var reviewTarget: ClinicalCase?
    @State private var navPath: [UUID] = []

    var body: some View {
        NavigationStack(path: $navPath) {
            List {
                if activeCases.isEmpty {
                    ContentUnavailableView {
                        Label("No cases yet", systemImage: "folder.badge.plus")
                    } description: {
                        Text("Tap New Case to report a suspected notifiable disease.")
                    }
                    .listRowBackground(Color.clear)
                } else {
                    Section {
                        ForEach(activeCases) { c in
                            NavigationLink(value: c.id) {
                                CaseListRow(clinicalCase: c)
                            }
                        }
                    } header: {
                        HStack {
                            Text("Active")
                            Spacer()
                            Text("\(activeCases.count)")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Cases")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        presentingNewCase = true
                    } label: {
                        Label("New Case", systemImage: "plus.circle.fill")
                    }
                }
            }
            .onAppear {
                // Env hook: CLINIQ_OPEN_NEW_CASE=1 auto-opens the new-case
                // sheet for the screenshot harness.
                let env = ProcessInfo.processInfo.environment
                if env["CLINIQ_OPEN_NEW_CASE"] == "1" {
                    presentingNewCase = true
                }
                if env["CLINIQ_OPEN_REVIEW"] == "1",
                   let first = allCases.first(where: { !$0.conditions.isEmpty })
                {
                    reviewTarget = first
                }
                // Open a case that has NO prior entities so the Review sheet
                // shows .intro and (if CLINIQ_AUTO_EXTRACT=1 is also set) the
                // ReviewFlowView.onAppear hook fires a real inference run.
                if env["CLINIQ_OPEN_DRAFT_REVIEW"] == "1",
                   let first = allCases.first(where: {
                       $0.conditions.isEmpty && $0.labs.isEmpty && $0.medications.isEmpty
                   })
                {
                    reviewTarget = first
                }
                if env["CLINIQ_OPEN_CASE_DETAIL"] == "1",
                   let first = allCases.first(where: { !$0.conditions.isEmpty })
                {
                    navPath = [first.id]
                }
            }
            .sheet(item: $reviewTarget) { target in
                ReviewFlowView(clinicalCase: target)
            }
            .navigationDestination(for: UUID.self) { caseID in
                if let c = allCases.first(where: { $0.id == caseID }) {
                    CaseDetailView(clinicalCase: c)
                } else {
                    ContentUnavailableView("Case not found",
                                           systemImage: "questionmark.folder")
                }
            }
            .sheet(isPresented: $presentingNewCase) {
                NewCaseView()
            }
        }
    }

    private var activeCases: [ClinicalCase] {
        allCases.filter { c in
            // Show everything that isn't a long-ago submitted case (older
            // than ~2 days). Recently-submitted cases remain visible so the
            // clinician can confirm the report landed before they move on.
            if c.status == .submitted,
               c.createdAt < Date().addingTimeInterval(-48 * 3600) {
                return false
            }
            return true
        }
    }
}

struct CaseListRow: View {
    let clinicalCase: ClinicalCase

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(clinicalCase.patient?.fullName ?? "Unnamed patient")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.primary)
                    Text(clinicalCase.primaryConditionDisplay)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                StatusBadge(status: clinicalCase.status)
            }
            HStack(spacing: 10) {
                if let p = clinicalCase.patient {
                    Label("\(p.genderDisplay), \(p.ageDescription)",
                          systemImage: "person")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("·").foregroundStyle(.secondary)
                Text(clinicalCase.createdAt, style: .relative)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if clinicalCase.hasUnreviewedEntities {
                    Spacer()
                    Label("Needs review", systemImage: "circle.dashed")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(Color(red: 0.45, green: 0.30, blue: 0.00))
                }
            }
        }
        .padding(.vertical, 4)
    }
}
