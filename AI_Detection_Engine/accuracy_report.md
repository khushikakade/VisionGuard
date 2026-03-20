# VisionGuard AI Accuracy Report
Date: 2026-03-20 20:08:10
Model: openai/clip-vit-large-patch14-336
Technique: Advanced Max-Pooling Ensembling
Total Snapshots Tested: 59

## Summary
- **Overall Accuracy (Top-3 + Threshold)**: 93.22%
- **Correct Classifications**: 55

## Detailed Results
| Filename | Ground Truth | Top Prediction | Top-3 Matches | Similarity | Threshold | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| alert_20260306_202750_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.7089 | 0.1 | ✅ PASS |
| alert_20260306_202750_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7771 | 0.15 | ✅ PASS |
| alert_20260306_202750_suspicious_pacing.jpg | suspicious_pacing | suspicious_pacing | suspicious_pacing, physical_altercation, crowd_gathering | 0.6913 | 0.1 | ✅ PASS |
| alert_20260306_202755_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6986 | 0.1 | ✅ PASS |
| alert_20260306_202755_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6819 | 0.1 | ✅ PASS |
| alert_20260306_202755_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, abnormal_movement_running | 0.7719 | 0.15 | ✅ PASS |
| alert_20260306_202802_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6879 | 0.1 | ✅ PASS |
| alert_20260306_202802_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, masked_person | 0.7778 | 0.15 | ✅ PASS |
| alert_20260306_204456_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, person_falling | 0.7139 | 0.1 | ✅ PASS |
| alert_20260306_204456_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, unattended_baggage | 0.7372 | 0.1 | ✅ PASS |
| alert_20260306_204456_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.7752 | 0.15 | ✅ PASS |
| alert_20260306_204500_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.7062 | 0.1 | ✅ PASS |
| alert_20260306_204500_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6960 | 0.1 | ✅ PASS |
| alert_20260306_204500_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7784 | 0.15 | ✅ PASS |
| alert_20260306_204504_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.7072 | 0.1 | ✅ PASS |
| alert_20260306_204504_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.7089 | 0.1 | ✅ PASS |
| alert_20260306_204504_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7771 | 0.15 | ✅ PASS |
| alert_20260306_204504_suspicious_pacing.jpg | suspicious_pacing | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.1913 | 0.1 | ❌ FAIL |
| alert_20260306_204510_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6986 | 0.1 | ✅ PASS |
| alert_20260306_204510_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6819 | 0.1 | ✅ PASS |
| alert_20260306_204510_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, abnormal_movement_running | 0.7719 | 0.15 | ✅ PASS |
| alert_20260306_204630_abnormal_movement_running.jpg | abnormal_movement_running | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.2139 | 0.1 | ❌ FAIL |
| alert_20260306_204630_person_falling.jpg | person_falling | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.2372 | 0.1 | ✅ PASS |
| alert_20260306_204630_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.7752 | 0.15 | ✅ PASS |
| alert_20260306_204637_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.7062 | 0.1 | ✅ PASS |
| alert_20260306_204637_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6960 | 0.1 | ✅ PASS |
| alert_20260306_204637_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.2784 | 0.15 | ✅ PASS |
| alert_20260306_204644_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.7072 | 0.1 | ✅ PASS |
| alert_20260306_204644_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7771 | 0.15 | ✅ PASS |
| alert_20260306_204644_suspicious_pacing.jpg | suspicious_pacing | suspicious_pacing | suspicious_pacing, physical_altercation, crowd_gathering | 0.6913 | 0.1 | ✅ PASS |
| alert_20260306_204645_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.7089 | 0.1 | ✅ PASS |
| alert_20260306_204653_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6986 | 0.1 | ✅ PASS |
| alert_20260306_204653_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6819 | 0.1 | ✅ PASS |
| alert_20260306_204653_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, abnormal_movement_running | 0.7719 | 0.15 | ✅ PASS |
| alert_20260306_204702_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6879 | 0.1 | ✅ PASS |
| alert_20260306_204702_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, masked_person | 0.7778 | 0.15 | ✅ PASS |
| alert_20260306_204710_abnormal_movement_running.jpg | abnormal_movement_running | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.2139 | 0.1 | ❌ FAIL |
| alert_20260306_204710_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, unattended_baggage | 0.7372 | 0.1 | ✅ PASS |
| alert_20260306_204710_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.7752 | 0.15 | ✅ PASS |
| alert_20260306_204732_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6833 | 0.1 | ✅ PASS |
| alert_20260306_204732_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7658 | 0.15 | ✅ PASS |
| alert_20260306_204743_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, person_falling | 0.7139 | 0.1 | ✅ PASS |
| alert_20260306_204743_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, unattended_baggage | 0.7372 | 0.1 | ✅ PASS |
| alert_20260306_204743_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.7752 | 0.15 | ✅ PASS |
| alert_20260306_214354_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, person_falling | 0.7139 | 0.1 | ✅ PASS |
| alert_20260306_214354_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, unattended_baggage | 0.7372 | 0.1 | ✅ PASS |
| alert_20260306_214354_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, person_falling, unattended_baggage | 0.7752 | 0.15 | ✅ PASS |
| alert_20260306_214355_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.7062 | 0.1 | ✅ PASS |
| alert_20260306_214355_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6960 | 0.1 | ✅ PASS |
| alert_20260306_214355_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7784 | 0.15 | ✅ PASS |
| alert_20260306_214356_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.7072 | 0.1 | ✅ PASS |
| alert_20260306_214356_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.7089 | 0.1 | ✅ PASS |
| alert_20260306_214356_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.7771 | 0.15 | ✅ PASS |
| alert_20260306_214356_suspicious_pacing.jpg | suspicious_pacing | physical_altercation | physical_altercation, crowd_gathering, unattended_baggage | 0.1913 | 0.1 | ❌ FAIL |
| alert_20260306_214358_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6986 | 0.1 | ✅ PASS |
| alert_20260306_214358_person_falling.jpg | person_falling | person_falling | person_falling, physical_altercation, crowd_gathering | 0.6819 | 0.1 | ✅ PASS |
| alert_20260306_214358_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, abnormal_movement_running | 0.7719 | 0.15 | ✅ PASS |
| alert_20260306_214359_abnormal_movement_running.jpg | abnormal_movement_running | abnormal_movement_running | abnormal_movement_running, physical_altercation, crowd_gathering | 0.6879 | 0.1 | ✅ PASS |
| alert_20260306_214359_physical_altercation.jpg | physical_altercation | physical_altercation | physical_altercation, crowd_gathering, masked_person | 0.7778 | 0.15 | ✅ PASS |
