// HistoryTab.swift
// Read-only view of submitted / failed cases, filterable by condition,
// status, and date window.

import SwiftUI
import SwiftData

struct HistoryTab: View {
    @Environment(\.modelContext) private var context
    @Query(sort: [SortDescriptor(\ClinicalCase.createdAt, order: .reverse)])
    private var allCases: [ClinicalCase]

    @State private var conditionFilter: String = "all"
    @State private var statusFilter: StatusFilter = .all

    enum StatusFilter: String, CaseIterable, Identifiable {
        case all = "All"
        case submitted = "Submitted"
        case failed = "Failed"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                filterBar
                    .padding(.horizontal)
                    .padding(.top, 10)
                List {
                    if filtered.isEmpty {
                        ContentUnavailableView("No history",
                                               systemImage: "clock.arrow.circlepath")
                            .listRowBackground(Color.clear)
                    } else {
                        ForEach(filtered) { c in
                            NavigationLink(value: c.id) {
                                HistoryRow(clinicalCase: c)
                            }
                        }
                    }
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
            }
            .background(ClinIQTheme.pageBackground)
            .navigationTitle("History")
            .navigationDestination(for: UUID.self) { id in
                if let c = allCases.first(where: { $0.id == id }) {
                    CaseDetailView(clinicalCase: c)
                } else {
                    ContentUnavailableView("Case not found",
                                           systemImage: "questionmark.folder")
                }
            }
        }
    }

    private var filterBar: some View {
        HStack(spacing: 10) {
            Picker("Status", selection: $statusFilter) {
                ForEach(StatusFilter.allCases) { s in
                    Text(s.rawValue).tag(s)
                }
            }
            .pickerStyle(.segmented)

            Menu {
                Button("All conditions") { conditionFilter = "all" }
                Divider()
                ForEach(conditionOptions, id: \.self) { c in
                    Button(c) { conditionFilter = c }
                }
            } label: {
                Label(conditionFilter == "all" ? "All conditions" : conditionFilter,
                      systemImage: "line.3.horizontal.decrease.circle")
                    .font(.footnote)
            }
        }
    }

    private var conditionOptions: [String] {
        let names = allCases.flatMap { $0.conditions.map { $0.displayName } }
        return Array(Set(names)).sorted()
    }

    private var filtered: [ClinicalCase] {
        allCases.filter { c in
            let statusOK: Bool = {
                switch statusFilter {
                case .all: return c.status == .submitted || c.status == .failed
                case .submitted: return c.status == .submitted
                case .failed: return c.status == .failed
                }
            }()
            guard statusOK else { return false }
            if conditionFilter == "all" { return true }
            return c.conditions.contains { $0.displayName == conditionFilter }
        }
    }
}

struct HistoryRow: View {
    let clinicalCase: ClinicalCase

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(clinicalCase.primaryConditionDisplay)
                        .font(.body.weight(.semibold))
                    Text(clinicalCase.patient?.fullName ?? "Unnamed patient")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                StatusBadge(status: clinicalCase.status)
            }
            HStack(spacing: 10) {
                Label(clinicalCase.createdAt.formatted(date: .abbreviated, time: .omitted),
                      systemImage: "calendar")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let fac = clinicalCase.patient?.facilityName {
                    Label(fac, systemImage: "building.2")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.vertical, 4)
    }
}
