# Archived: earlier RA-PDS-KD segment-level proposal

**This document is superseded and is not the methodology behind the current v3 results.** It described an earlier proposed pipeline using aligned Wav2Vec2/MiniLM segments, segment-level teachers, and clean/missing/noisy robustness conditions. That pipeline is not the frozen student-v3 experiment.

For the current implementation, results, and test protocol, use:

- [Current repository status and DEV results](../README.md)
- [Student v3 design](student_v3_design.md)
- [Student v3 DEV freeze](student_v3_dev_freeze.md)
- [Student v3 final TEST procedure](student_v3_final_test.md)
- Code: scripts/students/participant_student_v3/ and scripts/teachers/

The historical proposal is retained here for context only. Do not cite its segment-level setup or its old active-notebook path as the current method. The v3 controlled comparison found that Reliability-Aware KD did not outperform Standard KD on DEV-34; see the freeze record and README for the actual metrics.
