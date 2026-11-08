# First-image development diagnostic

Tolerance=5 pixels for this summary; 3/5/10 all saved. Cyan=predicted skeleton within 5px of reference; magenta=predicted skeleton outside reference band; yellow=reference not recovered within 5px. These are annotation-agreement indicators, not proof of real-world wire identity.

|Condition|Precision|Recall|F1|Mask pixels|Mask fraction in reference band|
|---|---|---|---|---|---|
|A_all_candidates|0.375123|0.399641|0.386994|515044|0.260683|
|B_vlm_keep_only|0.578277|0.097074|0.166241|27122|0.536539|
|C_remove_vehicle_only|0.381781|0.399641|0.390507|474393|0.283021|
|D_visual_segments_only|0.989816|0.090065|0.165106|16050|0.977009|
|E_hypothetical_manual_partial_repair|0.469766|0.482346|0.475973|420172|0.266969|