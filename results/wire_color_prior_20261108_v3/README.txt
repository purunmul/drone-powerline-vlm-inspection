Three development frames. First LangSAM masks only; no zoom added.
Palette sRGB values are illustrative engineering choices, not KEC numeric swatches.
Qwen reviews identical original/marked crop pairs with and without KEC color guidance.
At least two Qwen wire anchors and a passing unmarked control are needed for Qwen profiles.
Automatic anchors use narrow segments with consistent contrast on both sides, from at least two masks.
Combined acceptance is the union of expanded standard and photo-specific Lab ranges.
Mask pixels are removed only outside all applicable ranges with >=80% agreement in a 5x5 neighborhood.
The refined masks are re-skeletonized. Magenta marks removed centerline pixels and orange marks new ones.
Annotation-derived raster used for evaluation only. No training or manual point prompts.
