import SwiftUI

// MARK: - Main App Entry Point
@main
struct ParakeetTranscriptionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
            .windowResizability(.contentSize)
            .defaultSize(width: 800, height: 600)
        #endif
    }
}
