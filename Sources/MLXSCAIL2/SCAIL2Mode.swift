import MLXToolKit

// The animation/replacement task tags are canonical `Mode`s on the contract (1.6.0); they select
// SCAIL's `replaceFlag`. Quality/fast are the speed tags (denoise-step presets), private to this
// package — same pattern as the other Wan wrappers.
extension Mode {
    /// Fewer denoising steps — the quicker path.
    public static let fast: Mode = "fast"
    /// The reference quality path (config-default steps); the package default.
    public static let quality: Mode = "quality"
}

/// Map `mode` (+ any explicit `steps`) to a denoise step count. Explicit `steps` wins; `.fast` → 8
/// (SCAIL's distilled fast tier), otherwise the SCAIL default (16).
func resolveSteps(mode: Mode?, steps: Int?) -> Int {
    if let steps { return steps }
    switch mode {
    case .fast: return 8
    default: return 16  // nil / .quality / .animation / .replacement / unknown → default
    }
}

/// SCAIL's two task semantics ride the canonical `.animation` / `.replacement` modes; everything
/// else (incl. the speed tags and `nil`) defaults to animation (`replaceFlag == false`).
func resolveReplaceFlag(mode: Mode?) -> Bool {
    mode == .replacement
}
