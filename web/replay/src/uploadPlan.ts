/**
 * Presigned S3 multipart upload part math (INFRA-3). The JS analogue of the
 * iOS `ResumableChunkPlan` (`ios/Upload/Sources/PickleballUpload/UploadManifest.swift`)
 * and of the server's `presign_multipart_put` (`server/s3.py`): part count
 * uses ceiling division so a size that lands exactly on a part boundary
 * (`size % partSize == 0`) does not get a dangling trailing empty part.
 */

export type UploadPartRange = {
  partNumber: number;
  offset: number;
  length: number;
};

export type UploadPlan = {
  partCount: number;
  ranges: UploadPartRange[];
};

export function planParts(sizeBytes: number, partSizeBytes: number): UploadPlan {
  if (!Number.isFinite(partSizeBytes) || partSizeBytes <= 0) {
    throw new Error("partSizeBytes must be positive");
  }
  if (!Number.isFinite(sizeBytes) || sizeBytes < 0) {
    throw new Error("sizeBytes must not be negative");
  }
  if (sizeBytes === 0) {
    return { partCount: 0, ranges: [] };
  }

  const partCount = Math.ceil(sizeBytes / partSizeBytes);
  const ranges: UploadPartRange[] = [];
  for (let partNumber = 1; partNumber <= partCount; partNumber += 1) {
    const offset = (partNumber - 1) * partSizeBytes;
    const length = Math.min(partSizeBytes, sizeBytes - offset);
    ranges.push({ partNumber, offset, length });
  }
  return { partCount, ranges };
}
