// NetworkMonitor.swift
// Observable wrapper around NWPathMonitor. Publishes connectivity state
// for the offline banner + the sync service.
//
// We use `Observable` (iOS 17+) so SwiftUI views can simply @Environment or
// observe the value. The monitor runs on its own background queue and
// dispatches updates to the main actor.

import Foundation
import Network
import Combine

/// ObservableObject flavour — compatible with both SwiftUI @StateObject and
/// plain @Published observation. Kept as a class so it can be passed via
/// `.environmentObject`.
@MainActor
final class NetworkMonitor: ObservableObject {
    @Published private(set) var isOnline: Bool = false
    @Published private(set) var interfaceDescription: String = "unknown"

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "com.cliniq.NetworkMonitor")
    private var started = false

    /// Dev override so demos / tests can flip offline without ifconfig.
    /// When `true`, the monitor ignores NWPathMonitor and reports offline.
    @Published var simulateOffline: Bool = false {
        didSet { recompute() }
    }

    private var lastSystemPath: NWPath?

    init() {}

    func start() {
        guard !started else { return }
        started = true
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self = self else { return }
            Task { @MainActor in
                self.lastSystemPath = path
                self.interfaceDescription = Self.describe(path: path)
                self.recompute()
            }
        }
        monitor.start(queue: queue)
    }

    private func recompute() {
        if simulateOffline {
            isOnline = false
            return
        }
        if let p = lastSystemPath {
            isOnline = (p.status == .satisfied)
        } else {
            isOnline = false
        }
    }

    private static func describe(path: NWPath) -> String {
        if path.usesInterfaceType(.wifi) { return "Wi-Fi" }
        if path.usesInterfaceType(.cellular) { return "Cellular" }
        if path.usesInterfaceType(.wiredEthernet) { return "Ethernet" }
        if path.status == .satisfied { return "Online" }
        return "Offline"
    }
}
