# Result Table Manifest

These are historical original-submission results. For the revised
manuscript use `../revision_20260923/` and release v1.1.0.

These CSV files are compact, immutable source data for the manuscript's
reported tables and figures. They contain aggregate results only, not
raw images, annotations, model features, or participant data.

- `fig2_main_results_source.csv`: main DOTA-v1.5 and SODA-A comparison.
- `fig3_component_ablation_source.csv`: component and mechanism controls.
- `fig4_risk_coverage_source.csv`: selective-review risk/coverage curves.
- `fig5_hard_queries_source.csv`: hard-query subset results.
- `fig6_*_class_gains_source.csv`: class-level gains.
- `encoder_transfer.csv`: OpenCLIP, GeoRSCLIP, and RemoteCLIP results.
- `query_template_robustness.csv`: individual-template robustness.
- `matched_controls.csv`: reciprocal-neighbor and degree controls.
- `scale_sensitivity.csv`: fixed single-scale and multi-scale comparison.
- `relevance_threshold_sensitivity.csv`: polygon-coverage sensitivity.

Values must not be edited manually. Regenerate them from frozen outputs
or document any correction in the release notes.
