// CancellationTests.swift — SCAIL-2 through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before
// the notLoaded guard, envelope validation, or weights); CAN-3 is the document of record
// for the checkpoint cadence: the segmented DPM++ animation sampler threads a throwing
// `onStep` closure — `try Task.checkCancellation()` once per denoising step
// (MLXSCAIL2Package.runAnimation → SCAILRuntime.generate) — and the wan-core streaming
// VAE decode bails per temporal chunk (`Task.isCancelled` in decodeStreaming), with the
// wrapper's post-core `try Task.checkCancellation()` discarding a truncated result.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXSCAIL2

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config + stub media; construction is cheap (C13) and the entry checkpoint
        // throws before validation or weights are touched, so this is offline-safe.
        let package = MLXSCAIL2Package(configuration: SCAIL2Configuration())
        let request = CharacterAnimationRequest(
            referenceImage: Image(format: .png, data: Data()),
            drivingVideo: Video(format: .mp4, data: Data()))
        let report = await CancellationConformance.checkRun(
            package: package, request: request)
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // characterAnimation is a long-run capability — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MLXSCAIL2Package.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: MLXSCAIL2Package.manifest,
            posture: .cadence([
                // Per denoising step: the throwing onStep closure threaded through
                // SCAILRuntime.generate into the segmented DPM++ sampler.
                .init(phase: .denoise, unit: .step),
                // Per VAE-decode temporal chunk: wan-core decodeStreaming bails on
                // Task.isCancelled; the wrapper's post-core checkpoint rethrows.
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
