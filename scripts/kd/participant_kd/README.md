# Participant-level KD

Uses the exact `ParticipantReLiMPNet` architecture used by the No-KD baseline.

## Standard KD

- TRAIN-107 locked participant targets only
- audio teacher: frozen USSD run #4 TRAIN crop export
- text teacher: frozen Idiap checkpoint TRAIN graph-node export
- equal teacher contribution
- temperature: 2.0
- KD weight: 0.5
- hard loss: class-weighted BCE
- KD loss: average of the two softened teacher BCE losses, equivalently BCE to their mean softened target, multiplied by T^2
- threshold fixed at 0.5
- checkpoint selected on DEV-34 participant macro-F1
- DEV teacher targets are not used for student fitting
- TEST is not prepared or opened

Reliability-aware KD is added only after the Standard-KD result is frozen.
