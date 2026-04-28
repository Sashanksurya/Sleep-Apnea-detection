1. Multimodal Technical Overview
This dashboard translates the model&#39;s technical backend into a clean, digestible format. A
horizontal bar chart highlights the feature importance, with the Oxygen Desaturation Index
(ODI) taking the lead. Below, a performance table cleanly displays the 92.4% accuracy of the
active Random Forest model.
[ Placeholder: Insert corresponding image here ]
2. Clinical Patient Overview &amp; Trends
This view focuses on actionable patient data. A patient profile displays a large gauge
indicating risk, supported by a historical chart tracking metrics over time. The clean white
background helps prioritize the automated recommendations, ensuring the clinician&#39;s eye is
drawn to the most critical health indicators derived from the physiological signals.
[ Placeholder: Insert corresponding image here ]
3. Multimodal Signal Analysis
This interface shows the exact time-series data the model is analyzing. The screen is
dominated by stacked, time-synced waveforms of the three key inputs: filtered EEG, ECG,
and continuous SpO₂ saturation. When an apnea event is detected, a transparent overlay
highlights the specific signals across all channels simultaneously.
[ Placeholder: Insert corresponding image here ]
4. Automated Scoring &amp; Report Generation
Following the analysis of the 30-second epochs, the system synthesizes findings into a clear
output view. A central visual compares the total epochs classified as Apnea versus Normal.
A primary action button allows the clinician to instantly generate a comprehensive
diagnostic report.
[ Placeholder: Insert corresponding image here ]
5. Backend Dataset &amp; Training
This administrative view focuses on data integrity. A central pie chart visualizes the precise
dataset balance from the Physionet Apnea-ECG dataset used for training: 33,600 normal

epochs and 16,440 apnea epochs. Adjacent curves track the model&#39;s training and validation
accuracy to ensure it reaches the necessary clinical thresholds.
