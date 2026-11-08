Development only. Same frozen first LangSAM masks. Zoom addition is outside this test.
Fragments follow observed centerline pixels; short bridges are bounded by direction and offset.
Color can add at most 0.06 to ranking and does not change eligibility.
Pole contact zones use nearby fragment endpoints near the upper part of a pole candidate box.
These zones block proposed bridges but preserve already observed line pixels.
Each VLM can accept borderline bridges, veto reviewed ones or change pole state only if the
corresponding unmarked control passes on the same image. Unreviewed cases use the rules.
XML bridge support requires one annotated polyline near both ends and at least 70 percent of the bridge.
An unsupported bridge is not definitive evidence of a physical misconnection. No pole ground truth.
Original overlay selects the best mean F1 on the same development set.
