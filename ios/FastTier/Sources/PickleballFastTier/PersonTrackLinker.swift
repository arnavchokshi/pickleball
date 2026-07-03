import Foundation

public struct OnDevicePersonObservation: Equatable, Sendable {
    public var bboxXYWH: [Double]
    public var confidence: Double
    public var source: String
    public var role: String?

    public init(bboxXYWH: [Double], confidence: Double, source: String, role: String? = nil) {
        self.bboxXYWH = bboxXYWH
        self.confidence = confidence
        self.source = source
        self.role = role
    }
}

public struct PersonTrackLinker: Sendable {
    private struct ActiveTrack: Sendable {
        var id: Int
        var bboxXYWH: [Double]
        var lastFrameIndex: Int
    }

    public var iouThreshold: Double
    public var maxTrackAgeFrames: Int
    public var maxTracks: Int
    public var highConfidenceThreshold: Double
    public var lowConfidenceThreshold: Double
    private var nextTrackID: Int
    private var activeTracks: [ActiveTrack]

    public init(
        iouThreshold: Double = 0.3,
        maxTrackAgeFrames: Int = 8,
        maxTracks: Int = 4,
        highConfidenceThreshold: Double = 0.5,
        lowConfidenceThreshold: Double = 0.1
    ) {
        self.iouThreshold = iouThreshold
        self.maxTrackAgeFrames = maxTrackAgeFrames
        self.maxTracks = maxTracks
        self.highConfidenceThreshold = highConfidenceThreshold
        self.lowConfidenceThreshold = lowConfidenceThreshold
        self.nextTrackID = 1
        self.activeTracks = []
    }

    public mutating func update(frameIndex: Int, observations: [OnDevicePersonObservation]) -> [OnDevicePersonDetection] {
        pruneTracks(olderThan: frameIndex)
        let eligibleObservations = observations
            .filter { observation in observation.confidence >= lowConfidenceThreshold }
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    return lhs.bboxXYWH.lexicographicallyPrecedes(rhs.bboxXYWH)
                }
                return lhs.confidence > rhs.confidence
            }
        let highConfidenceObservations = eligibleObservations.filter { observation in
            observation.confidence >= highConfidenceThreshold
        }
        let lowConfidenceObservations = eligibleObservations.filter { observation in
            observation.confidence < highConfidenceThreshold
        }

        var usedTrackIDs = Set<Int>()
        var detections: [OnDevicePersonDetection] = []
        link(
            observations: highConfidenceObservations,
            frameIndex: frameIndex,
            allowNewTracks: true,
            usedTrackIDs: &usedTrackIDs,
            detections: &detections
        )
        link(
            observations: lowConfidenceObservations,
            frameIndex: frameIndex,
            allowNewTracks: false,
            usedTrackIDs: &usedTrackIDs,
            detections: &detections
        )
        return detections
            .sorted { $0.trackID < $1.trackID }
            .prefix(maxTracks)
            .map { $0 }
    }

    private mutating func link(
        observations: [OnDevicePersonObservation],
        frameIndex: Int,
        allowNewTracks: Bool,
        usedTrackIDs: inout Set<Int>,
        detections: inout [OnDevicePersonDetection]
    ) {
        for observation in observations {
            guard detections.count < maxTracks else {
                return
            }
            let matchedIndex = bestTrackIndex(for: observation, excluding: usedTrackIDs)
            let trackID: Int
            if let matchedIndex {
                activeTracks[matchedIndex].bboxXYWH = observation.bboxXYWH
                activeTracks[matchedIndex].lastFrameIndex = frameIndex
                trackID = activeTracks[matchedIndex].id
            } else {
                guard allowNewTracks, activeTracks.count < maxTracks else {
                    continue
                }
                trackID = nextTrackID
                nextTrackID += 1
                activeTracks.append(ActiveTrack(id: trackID, bboxXYWH: observation.bboxXYWH, lastFrameIndex: frameIndex))
            }
            usedTrackIDs.insert(trackID)
            detections.append(
                OnDevicePersonDetection(
                    trackID: trackID,
                    bboxXYWH: observation.bboxXYWH,
                    confidence: observation.confidence,
                    source: observation.source,
                    role: observation.role
                )
            )
        }
    }

    private mutating func pruneTracks(olderThan frameIndex: Int) {
        activeTracks.removeAll { track in
            frameIndex - track.lastFrameIndex > maxTrackAgeFrames
        }
    }

    private func bestTrackIndex(for observation: OnDevicePersonObservation, excluding usedTrackIDs: Set<Int>) -> Int? {
        var bestIndex: Int?
        var bestIOU = iouThreshold
        for (index, track) in activeTracks.enumerated() where !usedTrackIDs.contains(track.id) {
            let score = bboxIOU(observation.bboxXYWH, track.bboxXYWH)
            if score >= bestIOU {
                bestIndex = index
                bestIOU = score
            }
        }
        return bestIndex
    }
}

private func bboxIOU(_ lhs: [Double], _ rhs: [Double]) -> Double {
    guard lhs.count == 4, rhs.count == 4 else {
        return 0
    }
    let lx1 = lhs[0]
    let ly1 = lhs[1]
    let lx2 = lhs[0] + lhs[2]
    let ly2 = lhs[1] + lhs[3]
    let rx1 = rhs[0]
    let ry1 = rhs[1]
    let rx2 = rhs[0] + rhs[2]
    let ry2 = rhs[1] + rhs[3]
    let ix1 = max(lx1, rx1)
    let iy1 = max(ly1, ry1)
    let ix2 = min(lx2, rx2)
    let iy2 = min(ly2, ry2)
    let intersectionWidth = max(0, ix2 - ix1)
    let intersectionHeight = max(0, iy2 - iy1)
    let intersection = intersectionWidth * intersectionHeight
    guard intersection > 0 else {
        return 0
    }
    let union = lhs[2] * lhs[3] + rhs[2] * rhs[3] - intersection
    return union > 0 ? intersection / union : 0
}
