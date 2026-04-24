// OutboxTab.swift
// Queue of cases waiting to sync (pending), currently syncing, or failed.
// Submitted / draft cases are excluded — those belong in Cases / History.

import SwiftUI
import SwiftData

struct OutboxTab: View {
    @Environment(\.modelContext) private var context
    @EnvironmentObject private var sync: SyncService
    @EnvironmentObject private var monitor: NetworkMonitor
    @Query(sort: [SortDescriptor(\ClinicalCase.updatedAt, order: .reverse)])
    private var allCases: [ClinicalCase]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if outbox.isEmpty {
                    ContentUnavailableView {
                        Label("Outbox empty", systemImage: "checkmark.seal")
                    } description: {
                        Text("No pending reports. Cases queued from the review screen appear here until they sync.")
                    }
                } else {
                    headerCard
                        .padding(.horizontal)
                        .padding(.top, 10)
                    List {
                        ForEach(outbox) { c in
                            NavigationLink(value: c.id) {
                                OutboxRow(clinicalCase: c)
                            }
                        }
                    }
                    .listStyle(.insetGrouped)
                    .scrollContentBackground(.hidden)
                }
            }
            .background(ClinIQTheme.pageBackground)
            .navigationTitle("Outbox")
            .navigationDestination(for: UUID.self) { id in
                if let c = allCases.first(where: { $0.id == id }) {
                    CaseDetailView(clinicalCase: c)
                } else {
                    ContentUnavailableView("Case not found",
                                           systemImage: "questionmark.folder")
                }
            }
            .onChange(of: outbox.count) { _, _ in
                OutboxCounter.shared.pending = outbox.filter { $0.status == .pending || $0.status == .failed }.count
            }
            .onAppear {
                OutboxCounter.shared.pending = outbox.filter { $0.status == .pending || $0.status == .failed }.count
            }
        }
    }

    private var outbox: [ClinicalCase] {
        allCases.filter { c in
            c.status == .pending || c.status == .syncing || c.status == .failed
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(outbox.count) report\(outbox.count == 1 ? "" : "s") queued")
                        .font(.headline)
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                syncButton
            }
            if sync.isDraining {
                ProgressView(value: 0.0)
                    .progressViewStyle(.linear)
                    .tint(ClinIQTheme.accent)
            }
            if !sync.lastMessage.isEmpty {
                Text(sync.lastMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(ClinIQTheme.cardBackground, in: RoundedRectangle(cornerRadius: 12))
    }

    private var statusMessage: String {
        if sync.isDraining { return "Syncing to public-health endpoint..." }
        if !monitor.isOnline { return "Waiting for network. Will auto-sync when online." }
        return "Tap Sync now to drain the outbox immediately."
    }

    private var syncButton: some View {
        Button {
            Task { await sync.drainNow() }
        } label: {
            if sync.isDraining {
                ProgressView().controlSize(.small)
            } else {
                Label("Sync now", systemImage: "arrow.triangle.2.circlepath")
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(sync.isDraining || !monitor.isOnline)
    }
}

struct OutboxRow: View {
    let clinicalCase: ClinicalCase

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(clinicalCase.patient?.fullName ?? "Unnamed patient")
                        .font(.body.weight(.semibold))
                    Text(clinicalCase.primaryConditionDisplay)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                StatusBadge(status: clinicalCase.status)
            }
            HStack(spacing: 10) {
                Label(clinicalCase.updatedAt.formatted(.relative(presentation: .numeric)),
                      systemImage: "clock")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(clinicalCase.acceptedCount) entities")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}
