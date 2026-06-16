# Data Dictionary

Mean/std values are seed-level unless noted.

## `data/processed/altitude_selection_counts.csv`

- Source: `results_r1/altitude_sensitivity/altitude_selection_counts.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `method, method_label, altitude_m, count, fraction`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/altitude_summary.csv`

- Source: `results_r1/altitude_sensitivity/altitude_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `method, method_label, seeds, slots_per_seed, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, mean_virtual_aoi_queue_mean, mean_virtual_aoi_queue_std, mean_virtual_unc_queue_mean, mean_virtual_unc_queue_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, horizontal_travel_m_mean, horizontal_travel_m_std, vertical_travel_m_mean, vertical_travel_m_std, travel_3d_m_mean, travel_3d_m_std, runtime_ms_mean, runtime_ms_std` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/best_dcee_config_formal.json`

- Source: `results_r1/formal_drl_tuning/best_dcee_config_formal.json`
- Meaning: JSON configuration or selected checkpoint metadata.

## `data/processed/channel_sensitivity_summary.csv`

- Source: `results_r1/physical_robustness/channel_sensitivity_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `case_type, case_name, channel_mode, road_variant, primitive_noise, method, method_label, num_seeds, slots_per_seed, mean_penalty, mean_backlog, mean_virtual_backlog, mean_aoi_excess, mean_uncertainty_excess, mean_travel, mean_runtime_ms, solver_success_rate, mean_inner_iterations, mean_candidate_family_size`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/dcee_tuning_stage1.csv`

- Source: `results_r1/formal_drl_tuning/dcee_tuning_stage1.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `variant, method, method_label, num_seeds, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, mean_virtual_aoi_queue_mean, mean_virtual_aoi_queue_std, mean_virtual_unc_queue_mean, mean_virtual_unc_queue_std, travel_distance_mean, travel_distance_std, solver_success_mean, solver_success_std, solver_nit_mean, solver_nit_std, runtime_ms_mean, runtime_ms_std` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/dcee_tuning_stage2.csv`

- Source: `results_r1/formal_drl_tuning/dcee_tuning_stage2.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `variant, method, method_label, num_seeds, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, mean_virtual_aoi_queue_mean, mean_virtual_aoi_queue_std, mean_virtual_unc_queue_mean, mean_virtual_unc_queue_std, travel_distance_mean, travel_distance_std, solver_success_mean, solver_success_std, solver_nit_mean, solver_nit_std, runtime_ms_mean, runtime_ms_std` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/drl_checkpoint_validation_summary.csv`

- Source: `results_r1/formal_drl_tuning/drl_checkpoint_validation_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `algorithm, method, train_seed, checkpoint_step, model_path, validation_episodes, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, travel_distance_mean, travel_distance_std, solver_success_mean, solver_success_std, inference_runtime_ms_mean, inference_runtime_ms_std`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/drl_training_summary.csv`

- Source: `results_r1/formal_drl_tuning/drl_training_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `algorithm, method, train_seed, target_train_steps, completed_steps, status, message, wall_clock_sec, saved_models, python_version, stable_baselines3_version, gymnasium_version, torch_version, numpy_version`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/fc_union_best_source_counts.csv`

- Source: `results_r1/benchmark_full_balanced_rescue_10seed/merged/union_best_source_counts.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, union_best_source_method, union_best_source_label, union_best_is_executed_point, count`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/fc_union_candidate_family_coverage_summary.csv`

- Source: `results_r1/benchmark_full_balanced_rescue_10seed/merged/candidate_family_coverage_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, rows, min_candidate_count, max_candidate_count, executed_point_found_rate, has_full_family_rate, has_shortlist_rate, notes`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/fc_union_rescue_diagnostics_summary.csv`

- Source: `results_r1/benchmark_full_balanced_rescue_10seed/merged/rescue_diagnostics_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, candidate_evaluations, primary_solver_success_rate, final_solver_success_rate, primary_solver_fail_count, final_solver_fail_count, rescue_rate, mean_rescue_attempts, rescue_success_count, value_changed_by_rescue_rate, eligible_for_reference_rate, union_best_final_success_rate, union_best_eligible_for_reference_rate`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/fc_union_solver_diagnostics_summary.csv`

- Source: `results_r1/benchmark_full_balanced_rescue_10seed/merged/solver_diagnostics_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, candidate_evaluations, solver_success_rate, solver_fail_count, primary_solver_success_rate, primary_solver_fail_count, rescue_rate, rescue_success_count, mean_nit, p95_nit, max_solver_residual, mean_eval_runtime_ms`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/fc_union_table5_summary.csv`

- Source: `results_r1/benchmark_full_balanced_rescue_10seed/merged/table5_seed_level_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, finite_gap_mean, finite_gap_std, p95_gap_mean, signed_residual_mean, signed_residual_std, solver_success_rate_mean, rescue_rate_mean, mean_nit, union_size_mean, num_seeds`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/finite_gap_running.csv`

- Source: `results_r1/theory_diagnostics/finite_gap_running.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `slot, seed, profile, method, finite_gap, method_label, running_mean_finite_gap`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/formal_drl_comparison_summary.csv`

- Source: `results_r1/formal_drl_tuning/formal_comparison_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `method, num_test_seeds, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, mean_virtual_aoi_queue_mean, mean_virtual_aoi_queue_std, mean_virtual_unc_queue_mean, mean_virtual_unc_queue_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, travel_distance_mean, travel_distance_std, solver_success_mean, solver_success_std, solver_nit_mean, solver_nit_std, solver_runtime_ms_mean, solver_runtime_ms_std, inference_runtime_ms_mean, inference_runtime_ms_std` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/formal_pairwise_statistics.csv`

- Source: `results_r1/formal_drl_tuning/formal_pairwise_statistics.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `baseline, comparison, metric, paired_diff_mean_dcee_minus_other, paired_diff_std, bootstrap_ci_low, bootstrap_ci_high, dcee_mean, other_mean, relative_reduction_when_positive, num_paired_seeds`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/inference_runtime_summary.csv`

- Source: `results_r1/formal_drl_tuning/inference_runtime_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `method, inference_runtime_ms, inference_runtime_ms, solver_runtime_ms, solver_runtime_ms`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/main_performance_summary.csv`

- Source: `results_r1/final_main/main_performance_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `method, method_label, num_seeds, slots_per_seed, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, mean_virtual_aoi_queue_mean, mean_virtual_aoi_queue_std, mean_virtual_unc_queue_mean, mean_virtual_unc_queue_std, travel_distance_mean, travel_distance_std, solver_success_mean, solver_success_std, solver_nit_mean, solver_nit_std, runtime_ms_mean, runtime_ms_std` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/mechanism_ablation_summary.csv`

- Source: `results_r1/final_main/mechanism_ablation_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `method, method_label, num_seeds, slots_per_seed, total_penalty_mean, total_penalty_std, queue_backlog_mean, queue_backlog_std, aoi_violation_mean, aoi_violation_std, uncertainty_violation_mean, uncertainty_violation_std, mean_virtual_aoi_queue_mean, mean_virtual_aoi_queue_std, mean_virtual_unc_queue_mean, mean_virtual_unc_queue_std, travel_distance_mean, travel_distance_std, solver_success_mean, solver_success_std, solver_nit_mean, solver_nit_std, runtime_ms_mean, runtime_ms_std` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/physical_parameter_table.csv`

- Source: `results_r1/physical_robustness/physical_parameter_table.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `parameter, value, unit, interpretation`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/primitive_noise_summary.csv`

- Source: `results_r1/physical_robustness/primitive_noise_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `case_type, case_name, channel_mode, road_variant, primitive_noise, method, method_label, num_seeds, slots_per_seed, mean_penalty, mean_backlog, mean_virtual_backlog, mean_aoi_excess, mean_uncertainty_excess, mean_travel, mean_runtime_ms, solver_success_rate, mean_inner_iterations, mean_candidate_family_size`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/queue_summary.csv`

- Source: `results_r1/theory_diagnostics/queue_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, method_label, num_seeds, slots_per_seed, mean_real_backlog, final_real_backlog_mean, mean_virtual_backlog, final_virtual_backlog_mean, mean_penalty, mean_running_penalty_final, mean_aoi_excess, mean_uncertainty_excess, mean_travel, solver_success_rate, mean_runtime_ms`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/queue_trajectories.csv`

- Source: `results_r1/theory_diagnostics/queue_trajectories.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, seed, slot, method, method_label, V, sum_Q, mean_Q, sum_Y, sum_Z, sum_virtual_queue, mean_virtual_aoi_queue, mean_virtual_unc_queue, mean_AoI, mean_uncertainty, aoi_excess, uncertainty_excess, data_service_slack, data_service_slack_min, data_service_violation, aoi_slack, aoi_slack_min, aoi_violation, uncertainty_slack` ...
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/road_risk_variant_summary.csv`

- Source: `results_r1/physical_robustness/road_risk_variant_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `case_type, case_name, channel_mode, road_variant, primitive_noise, method, method_label, num_seeds, slots_per_seed, mean_penalty, mean_backlog, mean_virtual_backlog, mean_aoi_excess, mean_uncertainty_excess, mean_travel, mean_runtime_ms, solver_success_rate, mean_inner_iterations, mean_candidate_family_size`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/runtime_summary.csv`

- Source: `results_r1/runtime_scalability/runtime_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `scenario, scale_axis, scale_value, method, method_label, records, mean_ms_per_slot, p95_ms_per_slot, mean_solver_ms_per_slot, p95_solver_ms_per_slot, inner_solves_per_slot, serial_solve_count_per_slot, parallelizable_solve_count_per_slot, mean_solver_iterations_per_slot, p95_solver_iterations_per_slot, solver_success_rate, mean_candidate_family_size`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/selected_drl_checkpoints.json`

- Source: `results_r1/formal_drl_tuning/selected_drl_checkpoints.json`
- Meaning: JSON configuration or selected checkpoint metadata.

## `data/processed/slater_slack_summary_combined.csv`

- Source: `results_r1/theory_diagnostics/slater_slack_summary_combined.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, method, method_label, component, mean_slack, p5_slack, violation_ratio, records`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.

## `data/processed/v_tradeoff_summary.csv`

- Source: `results_r1/theory_diagnostics/v_tradeoff_summary.csv`
- Row meaning: processed experiment summary or trajectory record.
- Key columns: `profile, V, method, method_label, num_seeds, mean_penalty, mean_real_backlog, mean_virtual_backlog, mean_aoi_excess, mean_uncertainty_excess, mean_travel, mean_runtime_ms, solver_success_rate`
- Corresponding table/figure: see `ARTIFACTS_MANIFEST.csv`.
