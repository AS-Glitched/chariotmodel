DKA Alice Springs — real solar-only board v0 (LOCKED)
- Model: hcbsswm_v1.7_residual_ar, use_weather=false
- Data: alice_springs clean + bridge (load/SOC placeholders only)
- Weather: concurrent ERA5, NOT day-ahead forecast; weather path not used for ops
- Primary metric: daylight DSM proxy (~8% mean on this run)
- Secondary: multi-horizon energy vs clear-sky (H2/H3 model better than clear-sky)
- Forbidden claims: DSM-ready, beats persistence in general, weather skill, load/SOC skill
- Do not retune v1.7 on this file
