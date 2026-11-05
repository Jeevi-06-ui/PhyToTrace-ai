# PhytoTrace Evidence Map repair

## Exact scan and cause

I matched the saved inspection `INSP_090c51bc32b3423d95ef7d880abfcb27` to its recorded input, `reports/debug_segmentation/benchmark_whatsapp_orig.jpg`. The original benchmark image is 3024 × 4032 with no EXIF rotation tag. Replaying the old dilated-mask path on that file exactly reproduced the stored 2,235,217 px at 26.20% and 1,037,240 px at 25.57%. Those stored figures differ slightly from the approximate screenshot figures. A separately encoded inspection copy produced slightly different areas.

I inspected the individual binary masks. U-Net sigmoid values were concentrated around 0.474–0.507. At 0.5 the U-Net assigned much of the left background to foreground. The LAB/HSV fallback grouped the central leaf with adjacent foliage and much of the lower-right frame. The two raw YOLO masks independently cover lower-right foliage. The old pipeline dilated the heuristic mask, clipped the YOLO masks against it, and accepted the clipped results before checking leaf-mask reliability. The frontend drew the polygons the server supplied. The wrong location already appears in the native YOLO masks. I found no axis swap or polarity inversion in the YOLO mask mapping for this file.

| Original YOLO candidate | Confidence | Raw mask pixels | Original box | Overlap with undilated heuristic |
|---|---:|---:|---|---:|
| RAW_01 | 26.20% | 2,270,734 | [1197, 2376, 3018, 4024] | 88.83% |
| RAW_02 | 25.57% | 1,065,008 | [1279, 3299, 3007, 4027] | 79.49% |

High containment in that heuristic shows why it cannot establish that either mask belongs to the target leaflet.

## Polygon focus and result

The scanner accepts click or tap vertices, shows a closed outline with visible points, and offers Undo, Clear and Use outline before inference. FastAPI rejects crossing polygons, duplicate vertices, degenerate shapes and out-of-frame coordinates. The upload remains full resolution. The backend extracts the original-pixel bounding crop, runs the existing checkpoint, compares native masks with the rasterized user outline, and translates candidate geometry and masks back by the crop origin. The outline is labeled **User-selected leaf boundary** and rendered separately from model evidence.

The Chromium run drew 28 points around the central leaflet. Its bounding crop was [886, 1210, 2226, 2874]. Full-frame inference proposed two candidates at the existing 0.20 cutoff. Crop inference proposed zero. A separate diagnostic-only crop run at 0.05 yielded 16 candidates on the original benchmark image. Fifteen failed containment. The remaining proposal covered 1,455,268 raw pixels with 93.12% containment and failed the inherited maximum bounding-box/frame ratio. The audit verified that reprojection preserved each candidate's pixel count and intersection with the selected outline. Cropping removed the reported false highlights. It did not establish detection of the visible winding marks.

The current checkpoint has no validated acceptance operating point and previously failed a reviewed-negative check. It is now review-only. Geometry and confidence remain available in the candidate table. There are no lettered regions or damage percentages from this checkpoint, including with manual focus. The polygon supplies the selected leaf area. It cannot turn a model miss into a trusted zero or validate a lesion. Older saved scans are kept, but the scanner asks users to re-upload them before viewing evidence under the new workflow.

## Verification

- Chromium through the real Next.js scanner and FastAPI, using the original WhatsApp benchmark, reviewed development positive `5558326.jpg` and reviewed negative `BLK_3_1002_PL010_H.JPG`.
- Real full-image candidate counts were 2, 1 and 2 respectively. The WhatsApp crop returned 0 at the scanner cutoff. Each browser result displayed unavailable measurement with zero accepted regions. The final run recorded zero browser page errors.
- Pytest **97 passed, 3 skipped**. TypeScript type checking passed. Cloud workflows were outside this repair.
- No checkpoints or annotations changed. No frozen test image was used for tuning.

## Reproduce

Run the existing services with FastAPI on port 8007 and Next.js on port 3007, with `PHYTOTRACE_API_ORIGIN=http://127.0.0.1:8007` for Next.js.

```sh
node scripts/verify_evidence_polygon.cjs
.venv/bin/python scripts/audit_evidence_polygon.py
.venv/bin/python -m pytest -q
```

Local artifacts are in `reports/evidence_repair`. They contain pre-inference and final browser screenshots, fixed-scale 16-bit U-Net probability maps, binary U-Net and colour masks, the full-resolution crop, user boundary, separate native and reprojected YOLO masks, decision CSVs with original boxes and overlap, and the final overlays.
