import AVFoundation
import MLX
import ParakeetMLX
import SwiftUI

// MARK: - Performance Metrics
struct PerformanceMetrics {
    let processingTime: TimeInterval
    let audioLengthSeconds: Double
    let tokenCount: Int
    let realTimeRatio: Double  // How many times faster than real-time
    let tokensPerSecond: Double

    var formattedProcessingTime: String {
        String(format: "%.2f", processingTime)
    }

    var formattedAudioLength: String {
        String(format: "%.2f", audioLengthSeconds)
    }

    var formattedRealTimeRatio: String {
        String(format: "%.1fx", realTimeRatio)
    }

    var formattedTokensPerSecond: String {
        String(format: "%.1f", tokensPerSecond)
    }
}

// MARK: - Transcription Service
@MainActor
class TranscriptionService: ObservableObject {
    @Published var isModelLoading = false
    @Published var isTranscribing = false
    @Published var downloadProgress: Double = 0
    @Published var transcriptionProgress: Double = 0
    @Published var transcriptionResult = ""
    @Published var timeAlignedResults: [AlignedSentence] = []
    @Published var performanceMetrics: PerformanceMetrics?
    @Published var errorMessage: String?

    private var model: ParakeetTDT?

    func loadModel() async {
        isModelLoading = true
        errorMessage = nil

        do {
            // Add a timeout to prevent hanging indefinitely
            model = try await withThrowingTaskGroup(of: ParakeetTDT.self) { group in
                group.addTask {
                    try await loadParakeetModel(
                        from: "mlx-community/parakeet-tdt-0.6b-v2",
                        dtype: .float32,
                        cacheDirectory: getParakeetCacheDirectory(),
                        progressHandler: { @Sendable [weak self] progress in
                            Task { @MainActor in
                                self?.downloadProgress = progress.fractionCompleted
                            }
                        }
                    )
                }

                // Add timeout task
                group.addTask {
                    try await Task.sleep(nanoseconds: 300_000_000_000)  // 5 minutes timeout
                    throw NSError(
                        domain: "TimeoutError",
                        code: 1,
                        userInfo: [
                            NSLocalizedDescriptionKey: "Model loading timed out after 5 minutes"
                        ]
                    )
                }

                // Return the first completed task (either model load or timeout)
                let result = try await group.next()!
                group.cancelAll()
                return result
            }
        } catch {
            let detailedError =
                "Failed to load model: \(error.localizedDescription)\nError details: \(error)"
            errorMessage = detailedError
        }

        isModelLoading = false
    }

    func transcribeFile(at url: URL) {
        guard let model = model else {
            errorMessage = "Model not loaded"
            return
        }

        Task {
            // Clear previous results
            clearResults()

            isTranscribing = true
            transcriptionProgress = 0
            errorMessage = nil

            do {
                // Load audio file and measure timing
                let audioLoadStart = CFAbsoluteTimeGetCurrent()
                let audioData = try await loadAudioFile(from: url)
                let audioLoadTime = CFAbsoluteTimeGetCurrent() - audioLoadStart

                // Calculate audio duration
                let audioLengthSeconds = Double(audioData.shape[0]) / 16000.0  // 16kHz sample rate

                // Transcribe with timing
                let transcriptionStart = CFAbsoluteTimeGetCurrent()
                let result = try model.transcribe(
                    audioData: audioData,
                    chunkDuration: 30.0,
                    overlapDuration: 5.0,
                    chunkCallback: { @Sendable [weak self] current, total in
                        Task { @MainActor in
                            self?.transcriptionProgress = Double(current) / Double(total)
                        }
                    }
                )
                let transcriptionTime = CFAbsoluteTimeGetCurrent() - transcriptionStart

                transcriptionResult = result.text
                timeAlignedResults = result.sentences

                // Calculate performance metrics
                let tokenCount = result.sentences.flatMap { $0.tokens }.count
                let realTimeRatio = audioLengthSeconds / transcriptionTime
                let tokensPerSecond = Double(tokenCount) / transcriptionTime

                performanceMetrics = PerformanceMetrics(
                    processingTime: transcriptionTime,
                    audioLengthSeconds: audioLengthSeconds,
                    tokenCount: tokenCount,
                    realTimeRatio: realTimeRatio,
                    tokensPerSecond: tokensPerSecond
                )

            } catch {
                errorMessage = "Transcription failed: \(error.localizedDescription)"
            }

            isTranscribing = false
            transcriptionProgress = 0
        }
    }

    func clearResults() {
        transcriptionResult = ""
        timeAlignedResults = []
        performanceMetrics = nil
        errorMessage = nil
    }

    private func loadAudioFile(from url: URL) async throws -> MLXArray {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // Access security-scoped resource
                    guard url.startAccessingSecurityScopedResource() else {
                        continuation.resume(
                            throwing: NSError(
                                domain: "SecurityError", code: 1,
                                userInfo: [NSLocalizedDescriptionKey: "Cannot access file"]))
                        return
                    }
                    defer { url.stopAccessingSecurityScopedResource() }

                    let audioFile = try AVAudioFile(forReading: url)

                    // Convert to 16kHz mono
                    guard
                        let format = AVAudioFormat(
                            commonFormat: .pcmFormatFloat32,
                            sampleRate: 16000,
                            channels: 1,
                            interleaved: false
                        )
                    else {
                        continuation.resume(
                            throwing: NSError(
                                domain: "AudioError", code: 1,
                                userInfo: [
                                    NSLocalizedDescriptionKey: "Failed to create audio format"
                                ]))
                        return
                    }

                    let frameCount = AVAudioFrameCount(audioFile.length)
                    guard
                        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
                    else {
                        continuation.resume(
                            throwing: NSError(
                                domain: "AudioError", code: 2,
                                userInfo: [
                                    NSLocalizedDescriptionKey: "Failed to create audio buffer"
                                ]))
                        return
                    }

                    // Convert format if needed
                    if audioFile.processingFormat.sampleRate != format.sampleRate
                        || audioFile.processingFormat.channelCount != format.channelCount
                    {

                        guard
                            let converter = AVAudioConverter(
                                from: audioFile.processingFormat, to: format)
                        else {
                            continuation.resume(
                                throwing: NSError(
                                    domain: "AudioError", code: 3,
                                    userInfo: [
                                        NSLocalizedDescriptionKey:
                                            "Failed to create audio converter"
                                    ]))
                            return
                        }

                        guard
                            let inputBuffer = AVAudioPCMBuffer(
                                pcmFormat: audioFile.processingFormat,
                                frameCapacity: frameCount
                            )
                        else {
                            continuation.resume(
                                throwing: NSError(
                                    domain: "AudioError", code: 4,
                                    userInfo: [
                                        NSLocalizedDescriptionKey: "Failed to create input buffer"
                                    ]))
                            return
                        }

                        try audioFile.read(into: inputBuffer)

                        var error: NSError?
                        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                            outStatus.pointee = .haveData
                            return inputBuffer
                        }

                        converter.convert(to: buffer, error: &error, withInputFrom: inputBlock)

                        if let error = error {
                            continuation.resume(throwing: error)
                            return
                        }
                    } else {
                        try audioFile.read(into: buffer)
                    }

                    // Convert to MLXArray
                    guard let floatData = buffer.floatChannelData?[0] else {
                        continuation.resume(
                            throwing: NSError(
                                domain: "AudioError", code: 5,
                                userInfo: [NSLocalizedDescriptionKey: "Failed to get audio data"]))
                        return
                    }

                    let samples = Array(
                        UnsafeBufferPointer(start: floatData, count: Int(buffer.frameLength)))
                    let audioArray = MLXArray(samples)

                    continuation.resume(returning: audioArray)

                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
