# PhytoTrace AI master build and repair report

**5 November 2026 · Local verification · Experimental model status**

The existing Next.js scanner now runs the original PyTorch models through FastAPI v1. Invalid leaf boundaries yield an unavailable measurement and zero accepted evidence regions. The experimental U-Net improved on the small validation set. YOLO remains too weak for a scientifically defensible farmer-facing diagnosis, so the experimental checkpoints are opt-in and the application has no verified public deployment.

## Repository audit and failure reproduction

The working directory was `/Users/jeevithp/Desktop/PhyToTrace`. The root and frontend contained existing working-tree changes. No reset, remote push, data rewrite or checkpoint replacement was performed. Overlapping edits affected the scanner during the work and twice restored a heuristic-mask measurement path. That path was reconciled, and a real-image regression test now checks it in automatic and manual modes.

| Finding | Reproduced evidence | Change |
|---|---|---|
| U-Net checkpoint compatible but nearly flat | The one-channel LightweightUNet loaded strictly and entered eval mode. All 14 BatchNorm counters recorded nine training forwards. The original runtime reproduced probability ranges of roughly 0.473–0.508, with per-image standard deviations 0.0065–0.0094. The bare state dict has no optimizer step history. | Matched training's 256 × 256 RGB [0, 1] resize. Probability spread remains an abstention heuristic. It provides no calibration guarantee. |
| U-Net training and validation partitioning | The training CLI selected frozen train IDs and then the trainer split that short list again. Its validation metric therefore did not use the frozen validation set. Missing masks could become all-background targets. | The trainer consumes explicit reviewed train and validation pairs, rejects missing/misaligned/nonbinary masks, applies paired spatial and image-only intensity augmentation to training, and logs optimizer steps, combined BCE/Dice, Dice and IoU. |
| Colour fallback overclaimed foliage | An Otsu fallback could create foreground on non-green backgrounds. The scanner used heuristic pixels as the denominator. The reviewed negative PL010_H showed 25.85% damage before repairs. The smartphone scan reported 0.00% after rejecting all candidates. | Removed the Otsu fallback. Colour masks are labelled previews. Neither full-frame nor manually cropped heuristic output can produce accepted evidence or an area percentage. |
| YOLO mask coordinate mismatch | Boxes were in original coordinates while letterboxed masks were stretched directly to original dimensions. Fixed 50 px and 300,000 px gates also depended on upload resolution. | Requested native masks with padding reversal from Ultralytics. Computes containment and mask area in the same canvas, uses a model-resolution size floor, and retains the inherited 0.05 confidence, 58% containment and 35% bounding-box/frame heuristics. These are experimental operating settings. |
| Candidate and result state ambiguity | Missing weights, an empty candidate set, rejected candidates and unusable masks could produce equivalent output. | Distinct result states are returned. Current user-facing text avoids healthy, disease, pest species and infestation stage conclusions. |
| API and browser geometry | Legacy Pydantic response dropped evidence labels and details. The v1 path multiplied 0/255 masks by 255 again. Manual crop used duplicate branches and silently ignored invalid input. | Both paths share format, EXIF and crop validation. Inference runs in the original-pixel rectangle and translates geometry back to the original canvas. Masks use their original 0/255 values. Browser uses native-image SVG coordinates. |

The first test runs inherited a repository hook that automatically loaded the local environment and exercised the existing live MongoDB/Cloudinary test. Those calls cannot establish that rotated credentials were in use. The automatic environment loading was removed, the live test is opt-in, and the repaired cloud paths require explicit confirmation of credential rotation. Cloud verification in this report remains pending.

## Data and model recovery

The catalog contains 95 photos with a frozen 65/15/15 train/validation/test split. The repository records 18 completed leaf masks, of which 16 are marked ready with matching reviewer decisions. Training consumed 10 eligible samples, validation consumed 3, and 3 ready records belong to the excluded frozen test set. Verified train negatives are IMG_0010, IMG_0015 and IMG_0022. Validation currently has three positives and zero verified negatives. IMG_0020 and IMG_0024 are marked `needs_correction` and were excluded. No new masks or biological labels were created.

- Training IDs: IMG_0001, IMG_0004, IMG_0006, IMG_0008, IMG_0009, IMG_0010, IMG_0012, IMG_0014, IMG_0015, IMG_0022.
- Validation IDs: IMG_0002, IMG_0028, IMG_0095.
- YOLO export: 82 train polygons across 7 positives and 3 verified negatives. Validation has 15 polygons. The exporter rejects overlap across frozen splits, contradictory negatives and COCO image dimensions that disagree with the oriented photograph.
- Separate CVAT train and validation uploads and manifests are in [data/astra_annotation_batches](../data/astra_annotation_batches/REVIEW.md). They contain 55 and 11 unannotated images respectively. Metadata labels in the manifests are triage hints. Prioritize reviewed negatives, other damage and difficult smartphone views.

### U-Net

The BCEWithLogits and soft Dice loss both consume raw logits correctly. RGB [0, 1] is the existing checkpoint's scale. Foreground was encoded as nonzero in the binary masks. The recovered run used Adam at 0.0005, batch size 8, 40 epochs and 80 optimizer updates on MPS. The best validation Dice at training resolution was **0.7896**, IoU **0.6534**, at epoch 34. The checkpoint is in `models/checkpoints/astra_leaf_experiment/leaf_unet_best.pt`.

| Method | Mean original-resolution Dice | Mean original-resolution IoU |
|---|---:|---:|
| Legacy raw U-Net threshold | 0.440 | 0.339 |
| Recovered U-Net | 0.790 | 0.654 |
| LAB/HSV preview | 0.749 | 0.605 |

The recovered model loses to the heuristic on one of the three images. No independent deployment accuracy estimate follows from these three selected validation photographs.

### YOLOv8n-Seg

The existing checkpoint and the separate 25-epoch run loaded as one-class YOLOv8n segmentation models with class `mine_damage`. The new run used MPS, image size 640, mask downsample ratio 2, independent instance masks and mosaic disabled. The saved best experiment checkpoint contains the same 417 tensor values as the legacy checkpoint despite distinct file hashes and metadata. The final-epoch checkpoint contains changed weights. Independent CPU validation used the same three images and 15 polygons.

| Weights | Mask precision | Mask recall | Mask mAP50 | Mask mAP50–95 |
|---|---:|---:|---:|---:|
| Original best | 0.0621 | 0.0667 | 0.0096 | 0.0030 |
| New best | 0.0621 | 0.0667 | 0.0096 | 0.0030 |
| New last | 0.0414 | 0.0667 | 0.0122 | 0.0018 |

The old detector accepted one region on the known negative with the recovered U-Net. The last experimental weights accepted 31 on that negative. These are individual-image false-positive observations, not population estimates. The browser-positive accepted mask overlaps the reviewed training image’s mine mask at IoU **0.2953**. Leaf focus on the greenhouse image helped change the candidate set, but the old model still produced no accepted regions on that crop. Twelve experimental overlapping tiles produced 280 pre-deduplication regions and remain outside the production scanner. No threshold was optimized on the frozen test images. Representative reviewed positive and negative validation examples are needed before tuning a threshold.

## Real inference and browser evidence

| Case | Checkpoints | Observed output |
|---|---|---|
| Reviewer-confirmed negative, default HTTP API and Chromium upload | Original U-Net and YOLO | HTTP 200, `LEAF_MASK_UNRELIABLE`, 20 raw candidates, zero accepted, damage `null`. |
| Reviewed positive 5558326.jpg, Chromium upload and polygon click | Recovered U-Net and original YOLO | HTTP 200, `DAMAGE_DETECTED`, region A (model confidence 6.9%). Estimated 19.19% of the predicted leaf mask. This verifies rendering of a real prediction. |
| Complex greenhouse phone photograph, Chromium manual focus | Recovered U-Net and original YOLO | HTTP 200, `MINE_CANDIDATES_REJECTED`. The selected crop [816, 1169, 2268, 2961] was returned in the 3024 × 4032 original coordinate system. No accepted polygon was available on this crop. |
| Corrupt image and invalid crop | Default API | HTTP 400 and 422 respectively. No model result was fabricated. |
| Guest history and claimed guest UID | Default API | Empty history and HTTP 401 for access to a saved record. |
| Missing model and injected execution failure | Local failure smoke | `MODEL_UNAVAILABLE` and `INFERENCE_ERROR`, both with damage `null`. |

Screenshots are [default abstention](astra_cv/browser/default-unreliable.png), [reviewed positive](astra_cv/browser/reviewed-positive.png), [greenhouse focus](astra_cv/browser/greenhouse-manual.png) and [live weather](astra_cv/browser/live-weather.png). Mobile screenshots cover [English](astra_cv/browser/mobile-en.png), [Kannada](astra_cv/browser/mobile-kn.png) and [Hindi](astra_cv/browser/mobile-hi.png).

Raw visualizations include the source photograph, fixed-scale 16-bit probability map, thresholded U-Net mask, LAB/HSV preview, every native raw YOLO mask, accepted mask where available, overlay and CSV/JSON candidate decisions. See [original positive artifacts](astra_cv/original/reviewed_positive/08_candidates.csv), [recovered positive artifacts](astra_cv/new_leaf_original_yolo/reviewed_positive/08_candidates.csv), [greenhouse crop artifacts](astra_cv/new_leaf_original_yolo/greenhouse_manual/08_candidates.csv), [YOLO validation metrics](astra_cv/yolo_validation.json), [U-Net run log](astra_cv/unet_training.log) and [HTTP checks](astra_cv/http_smoke.txt). Manual diagnostic intermediate masks use crop-local coordinates and final overlays use original coordinates. The candidate tables use original coordinates.

## Integration and acceptance status

| Area | Implemented and verified | Remaining |
|---|---|---|
| Scanner and Evidence Map | Browser upload, v1 inference, pointer-based rectangle, evidence click, original/mask/overlay layers, opacity, scrolling pan at zoom, original-pixel outlines and null measurement handling. | Independent model validation, field negatives, wider phone coverage and any polygon focus beyond the rectangle option. |
| Firebase | Real Google popup, Firebase SDK session restoration and backend ID-token verification. Removed mock sign-in and caller-provided guest identity. | Google login with the authorized Firebase project and allowed domains. |
| Guest | Scan response is marked no-store and stays in the browser view for up to 30 minutes. No cloud or local inspection record is written. Per-process limits protect scan, assistant and weather routes. | Shared/distributed rate limits and a production load test. |
| MongoDB | Owner-filtered v1 saves, history, retrieval, preferences and aggregate farm/plant views. Evidence is embedded in the inspection document. Local legacy SQLite is separate. | Rotated Atlas credentials and authenticated browser round trip. Farm/plant CRUD, persisted weather/model/audit records and cloud migrations remain incomplete. |
| Cloudinary | Account scans use authenticated asset uploads, owner-gated signed download links and deletion with retryable asset keys. R2 is retained as legacy code. | Rotated Cloudinary credentials and a live upload, delivery and deletion check. Legacy public URLs require a separate authorized migration. |
| Languages | Shared persisted dictionary locale, localized scanner core, settings and portions of the existing navigation, landing, assistant and weather. Three scanner viewport checks passed at mobile width. | Full-page translation coverage, complete dynamic result translation and native-speaker review. |
| Assistant | Server-side Gemini Flash-Lite integration, output budget, owner-scoped inspection context, general-answer cache and explicit offline guidance. Removed pesticide doses. Offline response verified through HTTP. | Live Gemini call with an approved key and authenticated inspection context. |
| Weather | Real Open-Meteo conditions and forecast verified in Chromium after simulated device permission and coordinates. Manual coordinate entry and the six-field irrigation context form passed a Chromium HTTP round trip. Google provider code is selected when configured. No fabricated provider fallback. | Live Google Weather key, user device GPS check and weather/irrigation persistence. Current guidance stays at scouting and moisture checks. |
| Dashboard and settings | Account-backed endpoints, actual Recharts time series, empty states and Firebase settings page. | Authenticated records and cloud UI workflows remain unverified. |
| Deployment | CPU Dockerfile, narrow Docker context and configurable HTTPS backend origin prepared. Next.js production build succeeds. | Container image build and load test, authorized Cloud Run/Vercel projects, production secret provisioning, dependency audit remediation and external-browser verification. No public URL is claimed. |

### Sources used for provider and agricultural behavior

- [Google Weather API overview](https://developers.google.com/maps/documentation/weather/overview)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Gemini 3.1 Flash-Lite model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
- [ICAR-IIHR historical crop weather advisories](https://www.iihr.res.in/sites/default/files/weather/Crop%20weather%20situation%20-%202023.pdf)

## Code changed

- Inference and vision: `src/phytotrace/inference/pipeline.py`, `src/phytotrace/models/{leaf_segmenter,mine_segmenter}.py`, `src/phytotrace/api/{main,schemas,scan_input}.py`.
- Training and diagnosis: `src/phytotrace/data/dataset.py`, `src/phytotrace/training/train_{leaf,mine}_segmenter.py`, `src/phytotrace/segmentation/yolo_exporter.py`, `scripts/{train_segmentation,audit_astra_models}.py`.
- Services: `src/phytotrace/api/v1/router.py`, `auth/firebase.py`, `db/{database,mongo}.py`, `storage/cloudinary_storage.py`, `services/{assistant,weather}.py`, all under `src/phytotrace`.
- Browser: existing scanner, dashboard, history, farm, timeline, research, weather and assistant routes, new settings route, locale dictionaries, Firebase context and existing evidence/gauge/chart components. `scripts/verify_astra_scanner.cjs` captures real browser results.
- Deployment and isolation: `Dockerfile`, `.dockerignore`, `.env.example`, `frontend/next.config.ts`, `frontend/vercel.json`, `src/phytotrace/__init__.py` and test fixtures.

## Run commands

From the repository root, start the default backend and the frontend in separate shells.

```sh
cd /Users/jeevithp/Desktop/PhyToTrace
.venv/bin/python -m uvicorn phytotrace.api.main:app --port 8007
```

```sh
cd /Users/jeevithp/Desktop/PhyToTrace/frontend
npm ci
PHYTOTRACE_API_ORIGIN=http://127.0.0.1:8007 PHYTOTRACE_NEXT_DIST_DIR=.next-astra npm run dev -- --port 3007
```

For the experimental browser proof, restart the backend with the separate U-Net checkpoint and run the Chromium verifier. Port 3007 was used because an existing frontend dev server was already running.

```sh
PHYTOTRACE_LEAF_CHECKPOINT=models/checkpoints/astra_leaf_experiment/leaf_unet_best.pt \
  .venv/bin/python -m uvicorn phytotrace.api.main:app --port 8007

node scripts/verify_astra_scanner.cjs
.venv/bin/python -m pytest -q
PHYTOTRACE_API_ORIGIN=http://127.0.0.1:8007 PHYTOTRACE_NEXT_DIST_DIR=.next-astra-build npm --prefix frontend run build
```

Training reproduction. Use fresh output paths for another run. The audit script targets the completed ASTRA checkpoints.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train_segmentation.py \
  --task leaf --epochs 40 --save-dir models/checkpoints/astra_leaf_experiment
.venv/bin/python scripts/prepare_yolo_dataset.py \
  --output_dir data/astra_yolo_experiment --yaml_out data/astra_yolo_experiment/dataset.yaml
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train_segmentation.py \
  --task mine --epochs 25 --data_yaml data/astra_yolo_experiment/dataset.yaml --run-name astra_mine_experiment
.venv/bin/python scripts/audit_astra_models.py
```

Pytest finished with **96 passed and 3 cloud tests skipped**. The Next.js production build succeeded. See [pytest output](astra_cv/pytest.log) and [Next.js build output](astra_cv/next-build.log). Local cloud tests require explicit opt-in, fresh credentials and a dedicated target. Supply environment variables through a trusted local environment file or a cloud secret manager. Application imports do not implicitly read a local secret file.
