import AVFoundation
import MLX
import ParakeetMLX
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Main Content View
struct ContentView: View {
    @StateObject private var transcriptionService = TranscriptionService()
    @State private var showingFilePicker = false
    @State private var selectedFileURL: URL?

    var body: some View {
        Group {
            #if os(iOS)
                // Use NavigationStack for iOS (iOS 16+)
                if #available(iOS 16.0, *) {
                    NavigationStack {
                        mainContent
                    }
                } else {
                    // Fallback for older iOS versions
                    NavigationView {
                        mainContent
                            #if os(iOS)
                                .navigationBarHidden(true)
                            #endif
                    }
                }
            #else
                // Use NavigationSplitView for macOS for better multi-pane support
                if #available(macOS 13.0, *) {
                    NavigationSplitView {
                        // Sidebar (empty for now)
                        VStack {
                            Image(systemName: "waveform.and.mic")
                                .font(.title)
                                .foregroundColor(.blue)
                            Text("Parakeet")
                                .font(.headline)
                        }
                        .frame(minWidth: 200)
                    } detail: {
                        mainContent
                    }
                } else {
                    // Fallback for older macOS versions
                    NavigationView {
                        mainContent
                    }
                }
            #endif
        }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    selectedFileURL = url
                    transcriptionService.transcribeFile(at: url)
                }
            case .failure(let error):
                transcriptionService.errorMessage =
                    "Failed to select file: \(error.localizedDescription)"
            }
        }
        .task {
            await transcriptionService.loadModel()
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Header
            VStack(spacing: 8) {
                Image(systemName: "waveform.and.mic")
                    .font(.system(size: platformSpecificIconSize))
                    .foregroundColor(.blue)

                Text("Parakeet Transcription")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("Convert speech to text with AI")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .padding(.top)

            Spacer()

            // Main content area
            if transcriptionService.isModelLoading {
                ModelLoadingView(service: transcriptionService)
            } else if transcriptionService.isTranscribing {
                TranscribingView(service: transcriptionService)
            } else {
                MainContentView(
                    selectedFileURL: $selectedFileURL,
                    showingFilePicker: $showingFilePicker,
                    transcriptionService: transcriptionService
                )
            }

            Spacer()

            // Error handling
            if let error = transcriptionService.errorMessage {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
        .padding(platformSpecificPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #if os(macOS)
            .frame(minWidth: 600, minHeight: 400)
        #endif
    }

    // Platform-specific UI properties
    private var platformSpecificIconSize: CGFloat {
        #if os(iOS)
            return 60
        #else
            return 48
        #endif
    }

    private var platformSpecificPadding: CGFloat {
        #if os(iOS)
            return 20
        #else
            return 30
        #endif
    }
}

// MARK: - Model Loading View
struct ModelLoadingView: View {
    let service: TranscriptionService

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(platformSpecificProgressScale)
            Text("Loading AI model...")
                .font(.headline)
            if service.downloadProgress > 0 {
                ProgressView(value: service.downloadProgress)
                    .frame(maxWidth: platformSpecificProgressWidth)
                Text("\(Int(service.downloadProgress * 100))% downloaded")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private var platformSpecificProgressScale: CGFloat {
        #if os(iOS)
            return 1.5
        #else
            return 1.2
        #endif
    }

    private var platformSpecificProgressWidth: CGFloat {
        #if os(iOS)
            return 300
        #else
            return 400
        #endif
    }
}

// MARK: - Transcribing View
struct TranscribingView: View {
    let service: TranscriptionService

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(platformSpecificProgressScale)
            Text("Transcribing audio...")
                .font(.headline)
            if service.transcriptionProgress > 0 {
                ProgressView(value: service.transcriptionProgress)
                    .frame(maxWidth: platformSpecificProgressWidth)
                Text("Processing: \(Int(service.transcriptionProgress * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private var platformSpecificProgressScale: CGFloat {
        #if os(iOS)
            return 1.5
        #else
            return 1.2
        #endif
    }

    private var platformSpecificProgressWidth: CGFloat {
        #if os(iOS)
            return 300
        #else
            return 400
        #endif
    }
}

// MARK: - Main Content View
struct MainContentView: View {
    @Binding var selectedFileURL: URL?
    @Binding var showingFilePicker: Bool
    let transcriptionService: TranscriptionService

    var body: some View {
        VStack(spacing: 20) {
            // File selection button
            Button(action: {
                showingFilePicker = true
            }) {
                VStack(spacing: 8) {
                    Image(systemName: "doc.badge.plus")
                        .font(.system(size: platformSpecificButtonIconSize))
                    Text("Select Audio File")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity, minHeight: platformSpecificButtonHeight)
                .background(Color.blue.opacity(0.1))
                .foregroundColor(.blue)
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.blue, lineWidth: 2)
                        .opacity(0.3)
                )
            }
            .buttonStyle(PlainButtonStyle())
            #if os(macOS)
                .help("Select an audio file to transcribe")
            #endif

            if let url = selectedFileURL {
                Text("Selected: \(url.lastPathComponent)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Transcription results
            if !transcriptionService.transcriptionResult.isEmpty {
                TranscriptionResultsView(service: transcriptionService)
            }
        }
    }

    private var platformSpecificButtonIconSize: CGFloat {
        #if os(iOS)
            return 40
        #else
            return 32
        #endif
    }

    private var platformSpecificButtonHeight: CGFloat {
        #if os(iOS)
            return 120
        #else
            return 100
        #endif
    }
}

// MARK: - Transcription Results View
struct TranscriptionResultsView: View {
    let service: TranscriptionService

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Transcription Result")
                        .font(.headline)
                    Spacer()
                    Button("Clear") {
                        service.clearResults()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    #if os(macOS)
                        .help("Clear transcription results")
                    #endif

                    Button("Copy") {
                        copyToClipboard(service.transcriptionResult)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    #if os(macOS)
                        .help("Copy transcription to clipboard")
                    #endif
                }

                Text(service.transcriptionResult)
                    .font(.body)
                    .textSelection(.enabled)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)

                // Performance Metrics Section
                if let metrics = service.performanceMetrics {
                    PerformanceMetricsView(metrics: metrics)
                }

                if !service.timeAlignedResults.isEmpty {
                    Text("Time-aligned Transcript")
                        .font(.headline)
                        .padding(.top)

                    ForEach(Array(service.timeAlignedResults.enumerated()), id: \.offset) {
                        index, sentence in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(
                                    "[\(String(format: "%.1f", sentence.start))s - \(String(format: "%.1f", sentence.end))s]"
                                )
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .monospaced()
                                Spacer()
                            }
                            Text(sentence.text)
                                .font(.body)
                        }
                        .padding(.vertical, 4)
                        .padding(.horizontal, 12)
                        .background(Color.blue.opacity(0.05))
                        .cornerRadius(6)
                    }
                }
            }
            .padding()
        }
        .background(Color.gray.opacity(0.05))
        .cornerRadius(12)
        #if os(macOS)
            .frame(minHeight: 200)
        #endif
    }

    private func copyToClipboard(_ text: String) {
        #if os(iOS)
            UIPasteboard.general.string = text
        #elseif os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        #endif
    }
}

// MARK: - Performance Metrics View
struct PerformanceMetricsView: View {
    let metrics: PerformanceMetrics

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Performance Metrics")
                .font(.headline)
                .foregroundColor(.primary)

            LazyVGrid(columns: gridColumns, alignment: .leading, spacing: 12) {
                MetricCard(
                    title: "Processing Time",
                    value: "\(metrics.formattedProcessingTime)s",
                    icon: "clock",
                    color: .blue
                )

                MetricCard(
                    title: "Audio Duration",
                    value: "\(metrics.formattedAudioLength)s",
                    icon: "waveform",
                    color: .green
                )

                MetricCard(
                    title: "Real-time Ratio",
                    value: metrics.formattedRealTimeRatio,
                    icon: "speedometer",
                    color: .orange
                )

                MetricCard(
                    title: "Tokens Generated",
                    value: "\(metrics.tokenCount)",
                    icon: "textformat",
                    color: .purple
                )

                MetricCard(
                    title: "Tokens/Second",
                    value: metrics.formattedTokensPerSecond,
                    icon: "gauge.high",
                    color: .red
                )

                MetricCard(
                    title: "Efficiency",
                    value: metrics.realTimeRatio > 1.0 ? "Fast" : "Slow",
                    icon: metrics.realTimeRatio > 1.0
                        ? "checkmark.circle" : "exclamationmark.triangle",
                    color: metrics.realTimeRatio > 1.0 ? .green : .yellow
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
                .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.blue.opacity(0.2), lineWidth: 1)
        )
    }

    private var gridColumns: [GridItem] {
        #if os(iOS)
            return [
                GridItem(.flexible()),
                GridItem(.flexible()),
            ]
        #else
            return [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible()),
            ]
        #endif
    }
}

// MARK: - Metric Card View
struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.system(size: 16, weight: .medium))
                Spacer()
            }

            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(.primary)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
        }
        .padding(12)
        .frame(minHeight: 80)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(.ultraThinMaterial)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(color.opacity(0.3), lineWidth: 1)
        )
    }
}
