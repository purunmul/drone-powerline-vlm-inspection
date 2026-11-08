Original-resolution model predictions on the three development images.
Cyan lines are visualized at 5 px. The PNG centerline is the exact evaluation input.
Baseline is the original LangSAM prediction. Round 1 fixed adds every zoom centerline.
Agreement checks existing lines inside the 4 fixed-selected regions per image.
Agreement uses a 13 px orientation window, 5 px distance and a 25 degree angle limit.
Novel checks zoom-only components at >5 px from baseline, with span >=48 px,
eigenvalue ratio <=0.12, density <=1.6, and at least 80% within 6 px mask radius.
Reference annotations are used only for evaluation and comparing the methods.
Chosen output is a development result and is not an independent test score.
