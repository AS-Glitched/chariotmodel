# CHARIOT prototype map

| Layer | Module |
|-------|--------|
| Site ratings | columns on the sealed extract (`rated_pv_w`, battery Wh) |
| Admissibility | `chariot/admissibility.py` |
| Seal + split | `chariot/extract.py` |
| Baselines | `chariot/baselines.py` |
| HC-BSSWM | `chariot/hcbsswm.py` |
| Tribunal | `chariot/tribunal.py` |
| Three modes | `chariot/advisory.py` |
| Spine | `chariot/pipeline.py` |

Law: physics admits data, the world model proposes futures, the Tribunal certifies them, advisories speak or abstain.
