// PersistenceController.swift
// SwiftData container factory. Configures the store under
// ApplicationSupport/ with data-protection
// `.completeUntilFirstUserAuthentication`, cloud disabled, and a single
// shared container for the app lifetime.

import Foundation
import SwiftData

enum PersistenceController {
    /// Build the model container used by the running app. On first launch the
    /// store is seeded with demo data so the PoC has something to show.
    @MainActor
    static func makeContainer() -> ModelContainer {
        let schema = Schema([
            ClinicalCase.self,
            Patient.self,
            ExtractedCondition.self,
            ExtractedLab.self,
            ExtractedMedication.self,
            Vitals.self,
            SyncRecord.self,
        ])

        let storeURL = Self.storeURL()
        let config = ModelConfiguration(
            "ClinIQ",
            schema: schema,
            url: storeURL,
            cloudKitDatabase: .none)

        do {
            let container = try ModelContainer(for: schema, configurations: [config])
            Self.applyFileProtection(at: storeURL)
            Self.seedIfEmpty(container: container)
            return container
        } catch {
            // If migration fails (e.g. schema drift during development),
            // fall back to an in-memory store so the app still launches.
            NSLog("[ClinIQ] persistent store failed: \(error.localizedDescription). Falling back to in-memory.")
            let mem = ModelConfiguration("ClinIQMemory",
                                         schema: schema,
                                         isStoredInMemoryOnly: true,
                                         cloudKitDatabase: .none)
            // swiftlint:disable:next force_try
            let container = try! ModelContainer(for: schema, configurations: [mem])
            Self.seedIfEmpty(container: container)
            return container
        }
    }

    private static func storeURL() -> URL {
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory,
                                   in: .userDomainMask,
                                   appropriateFor: nil,
                                   create: true))
            ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = support.appendingPathComponent("ClinIQ", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("cliniq.store")
    }

    private static func applyFileProtection(at url: URL) {
        let fm = FileManager.default
        let attrs: [FileAttributeKey: Any] = [
            .protectionKey: FileProtectionType.completeUntilFirstUserAuthentication
        ]
        // Apply to store file + sidecars written by SwiftData / SQLite.
        let suffixes = ["", "-wal", "-shm"]
        for suffix in suffixes {
            let target = URL(fileURLWithPath: url.path + suffix)
            if fm.fileExists(atPath: target.path) {
                try? fm.setAttributes(attrs, ofItemAtPath: target.path)
            }
        }
    }

    // MARK: - Seed

    @MainActor
    private static func seedIfEmpty(container: ModelContainer) {
        let context = container.mainContext
        let descriptor = FetchDescriptor<ClinicalCase>()
        let existing = (try? context.fetchCount(descriptor)) ?? 0
        guard existing == 0 else { return }

        let seeds = DemoSeed.build()
        for seed in seeds {
            context.insert(seed)
        }
        try? context.save()
    }

    /// Wipe every ClinicalCase (and cascading children via SwiftData
    /// relationship rules) from the live store, then re-insert the four
    /// demo cases. Used by Settings → "Reset demo cases" so the presenter
    /// can recover a fresh demo state in two seconds when handing the
    /// phone to the next reviewer.
    @MainActor
    static func resetDemo(container: ModelContainer) {
        let context = container.mainContext
        let descriptor = FetchDescriptor<ClinicalCase>()
        if let cases = try? context.fetch(descriptor) {
            for c in cases {
                context.delete(c)
            }
        }
        let seeds = DemoSeed.build()
        for seed in seeds {
            context.insert(seed)
        }
        try? context.save()
    }
}
