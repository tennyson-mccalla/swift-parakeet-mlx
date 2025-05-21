# Cross-Platform Support - iOS & macOS

Parakeet MLX Swift is designed from the ground up to provide a seamless experience across both iOS and macOS platforms. This document outlines the specific optimizations and features that ensure the best possible user experience on each platform.

## Platform Requirements

| Platform | Minimum Version | Recommended | Architecture |
|----------|----------------|-------------|--------------|
| **iOS** | 16.0+ | 17.0+ | arm64 (Apple Silicon) |
| **macOS** | 14.0+ | 14.0+ | arm64 (Apple Silicon) |

> **Note**: While Intel Macs may work, we recommend Apple Silicon for optimal MLX performance.

## Platform-Specific Optimizations

### 🍎 iOS Optimizations

#### Navigation
- **iOS 16+**: Uses modern `NavigationStack` for optimal navigation experience
- **iOS 15**: Graceful fallback to `NavigationView` with hidden navigation bar
- **Adaptive layouts**: Automatically adjusts to portrait/landscape orientations

#### Touch Interface
- **Touch-optimized buttons**: Larger touch targets (40pt icons, 120pt button height)
- **Gesture-friendly spacing**: 20pt padding optimized for finger navigation
- **Dynamic layouts**: Responsive to different screen sizes (iPhone, iPad)

#### File Access
- **Documents app integration**: Seamless file picker with iOS document system
- **Security-scoped resources**: Proper handling of sandboxed file access
- **File sharing support**: Import audio files via AirDrop, Files app, or other apps

#### UI Elements
- **Progress indicators**: 1.5x scale for better visibility on mobile
- **Text selection**: Enabled for easy copying on touch devices
- **Clipboard integration**: Native UIPasteboard support

### 🖥️ macOS Optimizations

#### Navigation
- **macOS 13+**: Uses `NavigationSplitView` for native desktop experience
- **Sidebar integration**: Clean sidebar with app branding
- **Window management**: Resizable windows with sensible default size (800x600)

#### Desktop Interface
- **Precision controls**: Smaller, more precise UI elements (32pt icons, 100pt button height)
- **Desktop spacing**: 30pt padding for larger screen real estate
- **Window constraints**: Minimum window size (600x400) for optimal viewing

#### File Access
- **Finder integration**: Native macOS file picker with full Finder features
- **File type associations**: Proper audio file type declarations
- **Drag & drop**: Ready for future drag-and-drop file support

#### Desktop UX
- **Hover tooltips**: Contextual help text on buttons and controls
- **Keyboard shortcuts**: Ready for future keyboard shortcut implementation
- **Menu integration**: Prepared for native macOS menu bar integration
- **NSPasteboard**: Native macOS clipboard functionality

## Shared Features

### Universal Components
- **File picker**: Works seamlessly on both platforms with appropriate native UI
- **Progress tracking**: Consistent progress indication with platform-appropriate sizing
- **Error handling**: Platform-aware error messages and recovery suggestions
- **Model loading**: Identical async/await model loading experience
- **Audio processing**: Unified AVFoundation-based audio processing pipeline

### Responsive Design
- **Adaptive layouts**: UI automatically adjusts to screen size and platform
- **Dynamic scaling**: Icons, buttons, and spacing scale appropriately
- **Consistent branding**: Unified visual identity across platforms
- **Accessibility ready**: Foundation for future accessibility improvements

## Implementation Details

### Navigation Architecture
```swift
#if os(iOS)
// Modern iOS navigation
if #available(iOS 16.0, *) {
    NavigationStack { content }
} else {
    NavigationView { content } // Fallback
}
#else
// macOS desktop experience
if #available(macOS 13.0, *) {
    NavigationSplitView {
        sidebar
    } detail: {
        content
    }
} else {
    NavigationView { content } // Fallback
}
#endif
```

### Platform-Specific UI Properties
```swift
private var platformSpecificIconSize: CGFloat {
    #if os(iOS)
    return 60  // Larger for touch
    #else
    return 48  // Smaller for desktop
    #endif
}

private var platformSpecificPadding: CGFloat {
    #if os(iOS)
    return 20  // Compact for mobile
    #else
    return 30  // Spacious for desktop
    #endif
}
```

### Clipboard Integration
```swift
private func copyToClipboard(_ text: String) {
    #if os(iOS)
    UIPasteboard.general.string = text
    #elseif os(macOS)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
    #endif
}
```

## Getting Started

### For iOS Projects
1. Add ParakeetMLX to your iOS project
2. Copy `examples/TranscriptionUI.swift`
3. Optionally use `examples/Info.plist` for enhanced file integration
4. Build and run on iPhone or iPad

### For macOS Projects
1. Add ParakeetMLX to your macOS project
2. Copy `examples/TranscriptionUI.swift`
3. Build and run - the app will automatically use macOS-optimized UI

### Universal Projects
The same code works on both platforms! Just add ParakeetMLX to a multiplatform target and the UI will automatically adapt.

## Future Enhancements

### Planned iOS Features
- Siri Shortcuts integration
- Background audio processing
- Share extension for importing audio from other apps
- iOS-specific accessibility features

### Planned macOS Features
- Menu bar integration
- Keyboard shortcuts
- Drag & drop file support
- Services menu integration
- Touch Bar support (if applicable)

### Universal Improvements
- Accessibility improvements (VoiceOver, Dynamic Type)
- Localization support
- Advanced audio format support
- Real-time microphone transcription

## Best Practices

### Performance
- Always build with Xcode (not `swift build`) for proper Metal shader compilation
- Use `.bfloat16` data type for optimal performance on Apple Silicon
- Consider device capabilities when setting chunk sizes

### User Experience
- Provide clear feedback during model download (2.5GB file)
- Handle file access permissions gracefully
- Show appropriate loading states for each platform
- Test on both iPhone/iPad and Mac for consistent experience

### Development
- Use platform-specific UI guidelines (HIG for iOS, macOS Human Interface Guidelines)
- Test file picker integration on both platforms
- Verify clipboard functionality works correctly
- Ensure proper window sizing and constraints

---

> **Ready to get started?** Check out the [examples](examples/) directory for complete working implementations, or follow the [Quick Start guide](README.md#quick-start-ios--macos) to add speech recognition to your existing iOS or macOS app! 