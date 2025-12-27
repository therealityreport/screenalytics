# Run-Scoped Endpoint Inventory (Auto-Generated)

Generated: 2025-12-27T18:57:12Z

This table is generated from FastAPI routes at runtime. Columns `reads/writes/artifact_pointers` are heuristic.

## Canonical vs Deprecated

Canonical (run-scoped):
- POST /episodes/{ep_id}/runs/{run_id}/jobs/{stage}
- GET /episodes/{ep_id}/runs/{run_id}/state
- GET /episodes/{ep_id}/runs/{run_id}/integrity
- GET /episodes/{ep_id}/runs/{run_id}/screentime
- GET /episodes/{ep_id}/faces_review_bundle

Deprecated wrappers (use canonical paths):
- POST /jobs/detect_track
- POST /jobs/detect_track_async
- POST /jobs/cluster
- POST /jobs/cluster_async
- POST /jobs/faces_embed
- POST /celery_jobs/detect_track
- POST /celery_jobs/cluster
- POST /celery_jobs/faces_embed

| Method | Route | Handler | Required IDs | Required Query Params | Reads | Writes | Artifact Pointers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GET | /celery_jobs | apps.api.routers.celery_jobs.list_active_celery_jobs | - | - | yes | no | job-trigger |
| GET | /celery_jobs/active | apps.api.routers.celery_jobs.get_active_jobs_endpoint | - | - | yes | no | job-trigger |
| POST | /celery_jobs/cluster | apps.api.routers.celery_jobs.start_cluster_celery | - | - | no | yes | job-trigger |
| POST | /celery_jobs/detect_track | apps.api.routers.celery_jobs.start_detect_track_celery | - | - | no | yes | job-trigger |
| POST | /celery_jobs/faces_embed | apps.api.routers.celery_jobs.start_faces_embed_celery | - | - | no | yes | job-trigger |
| GET | /celery_jobs/history | apps.api.routers.celery_jobs.get_job_history_endpoint | - | - | yes | no | job-trigger |
| POST | /celery_jobs/kill_all_local | apps.api.routers.celery_jobs.kill_all_local_jobs | - | - | no | yes | job-trigger |
| GET | /celery_jobs/local | apps.api.routers.celery_jobs.list_local_jobs | - | - | yes | no | job-trigger |
| GET | /celery_jobs/logs/{ep_id}/{operation} | apps.api.routers.celery_jobs.get_operation_logs | ep_id | - | yes | no | job-trigger |
| POST | /celery_jobs/parallel | apps.api.routers.celery_jobs.start_parallel_jobs | - | req | no | yes | job-trigger |
| GET | /celery_jobs/parallel/{group_id} | apps.api.routers.celery_jobs.get_parallel_job_status | - | - | yes | no | job-trigger |
| GET | /celery_jobs/stream/{job_id} | apps.api.routers.celery_jobs.stream_celery_job | job_id | - | yes | no | job-trigger |
| GET | /celery_jobs/{job_id} | apps.api.routers.celery_jobs.get_celery_job_status | job_id | - | yes | no | job-trigger |
| POST | /celery_jobs/{job_id}/cancel | apps.api.routers.celery_jobs.cancel_celery_job | job_id | - | no | yes | job-trigger |
| GET | /episodes | apps.api.routers.episodes.list_episodes | - | - | yes | no | unknown |
| POST | /episodes | apps.api.routers.episodes.create_episode | - | - | no | yes | unknown |
| POST | /episodes/delete_all | apps.api.routers.episodes.delete_all | - | - | no | yes | unknown |
| POST | /episodes/purge_all | apps.api.routers.episodes.purge_all | - | - | no | yes | unknown |
| GET | /episodes/s3_shows | apps.api.routers.episodes.list_s3_shows | - | - | yes | no | unknown |
| GET | /episodes/s3_shows/{show}/episodes | apps.api.routers.episodes.list_s3_episodes_for_show | - | - | yes | no | unknown |
| GET | /episodes/s3_videos | apps.api.routers.episodes.list_s3_videos | - | - | yes | no | unknown |
| POST | /episodes/upsert_by_id | apps.api.routers.episodes.upsert_by_id | - | - | no | yes | unknown |
| DELETE | /episodes/{ep_id} | apps.api.routers.episodes.delete_episode | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id} | apps.api.routers.episodes.episode_details | ep_id | - | yes | no | unknown |
| PATCH | /episodes/{ep_id} | apps.api.routers.episodes.update_episode | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/analyze_unassigned | apps.api.routers.grouping.analyze_unassigned_clusters | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/artifact_status | apps.api.routers.episodes.get_artifact_status | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/assets | apps.api.routers.episodes.presign_episode_assets | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/assignments | apps.api.routers.episodes.get_episode_assignments | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/assignments/cluster | apps.api.routers.episodes.set_cluster_assignment | ep_id, run_id | run_id | no | yes | unknown |
| PUT | /episodes/{ep_id}/assignments/cluster | apps.api.routers.episodes.set_cluster_assignment | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/assignments/face_exclusion | apps.api.routers.episodes.set_face_exclusion | ep_id, run_id | run_id | no | yes | unknown |
| PUT | /episodes/{ep_id}/assignments/face_exclusion | apps.api.routers.episodes.set_face_exclusion | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/assignments/track | apps.api.routers.episodes.set_track_assignment | ep_id, run_id | run_id | no | yes | unknown |
| PUT | /episodes/{ep_id}/assignments/track | apps.api.routers.episodes.set_track_assignment | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/audio/smart_split | apps.api.routers.audio.smart_split_segment | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/auto_assign_high_confidence | apps.api.routers.grouping.auto_assign_high_confidence | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/auto_link_cast | apps.api.routers.grouping.auto_link_cast | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/backup | apps.api.routers.grouping.create_backup | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/backups | apps.api.routers.grouping.list_backups | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/cast_suggestions | apps.api.routers.grouping.get_cast_suggestions | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/cleanup_empty_clusters | apps.api.routers.episodes.cleanup_empty_clusters_endpoint | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/cleanup_preview | apps.api.routers.grouping.cleanup_preview | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/cluster_centroids | apps.api.routers.grouping.get_cluster_centroids | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/cluster_centroids/compute | apps.api.routers.grouping.compute_cluster_centroids | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/cluster_suggestions | apps.api.routers.grouping.get_cluster_suggestions | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/cluster_suggestions_from_assigned | apps.api.routers.grouping.get_cluster_suggestions_from_assigned | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/cluster_tracks | apps.api.routers.episodes.list_cluster_tracks | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/clusters/batch_assign | apps.api.routers.grouping.batch_assign_clusters | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/clusters/batch_assign_async | apps.api.routers.grouping.batch_assign_clusters_async | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/clusters/group | apps.api.routers.grouping.group_clusters | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/clusters/group/progress | apps.api.routers.grouping.group_clusters_progress | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/clusters/group_async | apps.api.routers.grouping.group_clusters_async | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/clusters/{cluster_id}/metrics | apps.api.routers.episodes.get_cluster_metrics | ep_id, cluster_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/clusters/{cluster_id}/suggest_cast | apps.api.routers.grouping.suggest_cast_for_cluster | ep_id, cluster_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/clusters/{cluster_id}/track_reps | apps.api.routers.episodes.get_cluster_track_reps | ep_id, run_id, cluster_id | run_id | yes | no | unknown |
| GET | /episodes/{ep_id}/consistency_check | apps.api.routers.grouping.cross_episode_consistency | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/delete | apps.api.routers.episodes.delete_episode_new | ep_id | - | no | yes | unknown |
| DELETE | /episodes/{ep_id}/dismissed_suggestions | apps.api.routers.grouping.clear_dismissed_suggestions | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/dismissed_suggestions | apps.api.routers.grouping.get_dismissed_suggestions | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/dismissed_suggestions | apps.api.routers.grouping.dismiss_suggestions | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/dismissed_suggestions/reset_state | apps.api.routers.grouping.reset_dismissed_suggestions_state | ep_id, run_id | run_id | no | yes | unknown |
| DELETE | /episodes/{ep_id}/dismissed_suggestions/{suggestion_id} | apps.api.routers.grouping.restore_suggestion | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/events | apps.api.routers.episodes.episode_events | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/face_review/decision/start | apps.api.routers.face_review.process_decision | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/face_review/improve_faces_queue | apps.api.routers.face_review.get_improve_faces_queue | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/face_review/initial_unassigned_suggestions | apps.api.routers.face_review.get_initial_unassigned_suggestions | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/face_review/mark_initial_pass_done | apps.api.routers.face_review.mark_initial_pass_done | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/face_review/reset_initial_pass | apps.api.routers.face_review.reset_initial_pass | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/face_review/reset_state | apps.api.routers.face_review.reset_face_review_state | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/faces/move_frames | apps.api.routers.episodes.move_faces | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/faces/{face_id}/unskip | apps.api.routers.episodes.unskip_face | ep_id, face_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/faces_grid | apps.api.routers.episodes.faces_grid | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/faces_review_bundle | apps.api.routers.episodes.get_faces_review_bundle | ep_id, run_id | run_id | yes | no | unknown |
| GET | /episodes/{ep_id}/find_similar_unassigned | apps.api.routers.grouping.find_similar_unassigned | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/frame/{frame_idx}/preview | apps.api.routers.episodes.generate_frame_preview | ep_id | - | yes | no | unknown |
| DELETE | /episodes/{ep_id}/frames | apps.api.routers.episodes.delete_frame | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/frames/{frame_idx}/overlay | apps.api.routers.episodes.generate_frame_overlay | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/frames_with_faces | apps.api.routers.episodes.get_frames_with_faces | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/generate_singleton_suggestions | apps.api.routers.episodes.generate_singleton_suggestions | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/hydrate | apps.api.routers.episodes.hydrate_episode_video | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/identities | apps.api.routers.episodes.list_identities | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/identities/merge | apps.api.routers.episodes.merge_identities | ep_id, run_id | run_id | no | yes | unknown |
| DELETE | /episodes/{ep_id}/identities/{identity_id} | apps.api.routers.episodes.delete_identity | ep_id, run_id, identity_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/identities/{identity_id} | apps.api.routers.episodes.identity_detail | ep_id, identity_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/identities/{identity_id}/export_seeds | apps.api.routers.episodes.export_facebank_seeds | ep_id, run_id, identity_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/identities/{identity_id}/lock | apps.api.routers.episodes.lock_identity | ep_id, run_id, identity_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/identities/{identity_id}/name | apps.api.routers.episodes.assign_identity_name | ep_id, run_id, identity_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/identities/{identity_id}/rename | apps.api.routers.episodes.rename_identity | ep_id, run_id, identity_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/identities/{identity_id}/unlock | apps.api.routers.episodes.unlock_identity | ep_id, run_id, identity_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/identities_with_metrics | apps.api.routers.episodes.list_identities_with_metrics | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/identity_locks | apps.api.routers.episodes.list_identity_locks | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/merge_all_duplicates | apps.api.routers.grouping.merge_all_duplicates | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/merge_clusters | apps.api.routers.grouping.merge_clusters | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/mirror | apps.api.routers.episodes.mirror_episode_video | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/mirror_artifacts | apps.api.routers.episodes.mirror_episode_artifacts | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/outlier_audit | apps.api.routers.grouping.get_outlier_audit | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/people/{person_id}/clusters_summary | apps.api.routers.episodes.get_person_clusters_summary | ep_id, run_id | run_id | yes | no | unknown |
| GET | /episodes/{ep_id}/potential_duplicates | apps.api.routers.grouping.get_potential_duplicates | ep_id | - | yes | no | unknown |
| DELETE | /episodes/{ep_id}/presence | apps.api.routers.episodes.leave_presence | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/presence | apps.api.routers.episodes.get_presence | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/presence | apps.api.routers.episodes.update_presence | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/progress | apps.api.routers.episodes.episode_progress | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/recover_noise_tracks | apps.api.routers.episodes.recover_noise_tracks | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/recover_noise_tracks/preview | apps.api.routers.episodes.recover_noise_tracks_preview | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/refresh_similarity | apps.api.routers.episodes.refresh_similarity_values | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/refresh_similarity_async | apps.api.routers.episodes.refresh_similarity_async | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/report.pdf | apps.api.routers.episodes.get_episode_report_pdf | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/rescue_quality_clusters | apps.api.routers.episodes.rescue_quality_clusters | ep_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/restore/{backup_id} | apps.api.routers.grouping.restore_backup | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/runs | apps.api.routers.episodes.list_episode_runs | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/runs/{run_id}/export | apps.api.routers.episodes.export_run_debug_bundle | ep_id, run_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/runs/{run_id}/integrity | apps.api.routers.episodes.get_run_integrity | ep_id, run_id | - | yes | no | run_state |
| POST | /episodes/{ep_id}/runs/{run_id}/jobs/{stage} | apps.api.routers.episodes.trigger_run_stage_job | ep_id, run_id | - | no | yes | job-trigger |
| GET | /episodes/{ep_id}/runs/{run_id}/screentime | apps.api.routers.episodes.get_run_screentime | ep_id, run_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/runs/{run_id}/state | apps.api.routers.episodes.get_run_processing_state | ep_id, run_id | - | yes | no | run_state |
| POST | /episodes/{ep_id}/save_assignments | apps.api.routers.grouping.save_assignments | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/singleton_analysis | apps.api.routers.episodes.analyze_singletons_batch | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/singletons/{track_id}/nearby_suggestions | apps.api.routers.episodes.get_singleton_nearby_suggestions | ep_id, track_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/smart_suggestions | apps.api.routers.grouping.list_smart_suggestions | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/smart_suggestions/apply | apps.api.routers.grouping.apply_smart_suggestion | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/smart_suggestions/apply_all | apps.api.routers.grouping.apply_all_smart_suggestions | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/smart_suggestions/batches | apps.api.routers.grouping.list_smart_suggestion_batches | ep_id, run_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/smart_suggestions/dismiss | apps.api.routers.grouping.dismiss_smart_suggestions | ep_id, run_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/smart_suggestions/generate | apps.api.routers.grouping.generate_smart_suggestions | ep_id, run_id | run_id | no | yes | unknown |
| GET | /episodes/{ep_id}/snapshots | apps.api.routers.episodes.list_episode_snapshots | ep_id | - | yes | no | unknown |
| DELETE | /episodes/{ep_id}/snapshots/{snapshot_id} | apps.api.routers.episodes.delete_episode_snapshot | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/status | apps.api.routers.episodes.episode_run_status | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/sync_thumbnails_to_s3 | apps.api.routers.episodes.sync_thumbnails_to_s3 | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/tiered_suggestions | apps.api.routers.grouping.get_tiered_suggestions | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/timeline_export | apps.api.routers.episodes.export_timeline_data | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/timestamp/{timestamp_s}/preview | apps.api.routers.episodes.generate_timestamp_preview | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/tracks/bulk_assign | apps.api.routers.episodes.bulk_assign_tracks | ep_id, run_id | run_id | no | yes | unknown |
| DELETE | /episodes/{ep_id}/tracks/{track_id} | apps.api.routers.episodes.delete_track | ep_id, track_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/tracks/{track_id} | apps.api.routers.episodes.track_detail | ep_id, run_id, track_id | run_id | yes | no | unknown |
| GET | /episodes/{ep_id}/tracks/{track_id}/crops | apps.api.routers.episodes.list_track_crops | ep_id, run_id, track_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/tracks/{track_id}/force_embed | apps.api.routers.episodes.force_embed_track | ep_id, track_id | - | no | yes | unknown |
| DELETE | /episodes/{ep_id}/tracks/{track_id}/frames | apps.api.routers.episodes.delete_track_frames | ep_id, track_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/tracks/{track_id}/frames | apps.api.routers.episodes.list_track_frames | ep_id, run_id, track_id | run_id | yes | no | unknown |
| POST | /episodes/{ep_id}/tracks/{track_id}/frames/move | apps.api.routers.episodes.move_track_frames | ep_id, track_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/tracks/{track_id}/integrity | apps.api.routers.episodes.track_integrity | ep_id, run_id, track_id | run_id | yes | no | unknown |
| GET | /episodes/{ep_id}/tracks/{track_id}/metrics | apps.api.routers.episodes.get_track_metrics | ep_id, track_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/tracks/{track_id}/move | apps.api.routers.episodes.move_track | ep_id, track_id | - | no | yes | unknown |
| POST | /episodes/{ep_id}/tracks/{track_id}/name | apps.api.routers.episodes.assign_track_name | ep_id, run_id, track_id | run_id | no | yes | unknown |
| POST | /episodes/{ep_id}/tracks/{track_id}/unskip_all | apps.api.routers.episodes.unskip_all_track_faces | ep_id, track_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/unclustered_tracks | apps.api.routers.episodes.list_unclustered_tracks | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/undo | apps.api.routers.grouping.undo_last_operation | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/undo_stack | apps.api.routers.grouping.get_undo_stack | ep_id | - | yes | no | unknown |
| GET | /episodes/{ep_id}/unlinked_entities | apps.api.routers.grouping.list_unlinked_entities | ep_id | - | yes | no | unknown |
| POST | /episodes/{ep_id}/video_clip | apps.api.routers.episodes.generate_video_clip | ep_id | - | no | yes | unknown |
| GET | /episodes/{ep_id}/video_meta | apps.api.routers.episodes.episode_video_meta | ep_id | - | yes | no | unknown |
| GET | /jobs | apps.api.routers.jobs.list_jobs | - | - | yes | no | job-trigger |
| GET | /jobs/audio/prerequisites | apps.api.routers.audio.check_audio_prerequisites | - | - | yes | no | job-trigger |
| POST | /jobs/body_tracking/fusion | apps.api.routers.jobs.run_body_tracking_fusion | - | - | no | yes | job-trigger |
| POST | /jobs/body_tracking/run | apps.api.routers.jobs.run_body_tracking | - | - | no | yes | job-trigger |
| POST | /jobs/cluster | apps.api.routers.jobs.run_cluster | - | - | no | yes | job-trigger |
| POST | /jobs/cluster_async | apps.api.routers.jobs.enqueue_cluster_async | - | - | no | yes | job-trigger |
| POST | /jobs/detect | apps.api.routers.jobs.enqueue_detect | - | - | no | yes | job-trigger |
| POST | /jobs/detect_track | apps.api.routers.jobs.run_detect_track | - | - | no | yes | job-trigger |
| POST | /jobs/detect_track_async | apps.api.routers.jobs.enqueue_detect_track_async | - | - | no | yes | job-trigger |
| POST | /jobs/episode_audio_diarize_transcribe | apps.api.routers.audio.start_diarize_transcribe | - | - | no | yes | job-trigger |
| POST | /jobs/episode_audio_files | apps.api.routers.audio.start_audio_files | - | - | no | yes | job-trigger |
| POST | /jobs/episode_audio_finalize | apps.api.routers.audio.start_finalize_transcript | - | - | no | yes | job-trigger |
| POST | /jobs/episode_audio_pipeline | apps.api.routers.audio.start_audio_pipeline | - | - | no | yes | job-trigger |
| GET | /jobs/episode_audio_status | apps.api.routers.audio.get_audio_status | ep_id | ep_id | yes | no | job-trigger |
| POST | /jobs/episode_cleanup_async | apps.api.routers.jobs.enqueue_episode_cleanup_async | - | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/clusters/bulk_merge | apps.api.routers.audio.bulk_merge_clusters | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/clusters/merge | apps.api.routers.audio.merge_clusters | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/clusters/preview | apps.api.routers.audio.preview_clustering | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/clusters/recluster | apps.api.routers.audio.recluster_voices | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/clusters/similarity_matrix | apps.api.routers.audio.get_cluster_similarity_matrix | ep_id | - | yes | no | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/clusters/suggest_merges | apps.api.routers.audio.suggest_cluster_merges | ep_id | - | yes | no | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/diarization/comparison | apps.api.routers.audio.get_diarization_comparison | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/diarize_only | apps.api.routers.audio.diarize_only | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/qc.json | apps.api.routers.audio.download_audio_qc | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/archive | apps.api.routers.audio.archive_transcript_segment | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/segments/archived | apps.api.routers.audio.list_archived_segments | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/assign_cast | apps.api.routers.audio.assign_segment_to_cast | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/move | apps.api.routers.audio.move_segment | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/segments/quality | apps.api.routers.audio.get_segment_quality_scores | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/restore | apps.api.routers.audio.restore_archived_segment | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/smart_split | apps.api.routers.audio.smart_split_segment | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/split | apps.api.routers.audio.split_segment | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/segments/word-split | apps.api.routers.audio.word_level_smart_split | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/segments/{segment_id}/words | apps.api.routers.audio.get_segment_words | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/speaker-assignment | apps.api.routers.audio.upsert_single_speaker_assignment | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/speaker-assignments | apps.api.routers.audio.get_speaker_assignments | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/speaker-assignments | apps.api.routers.audio.set_speaker_assignments | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/speaker-groups/split-at-utterance | apps.api.routers.audio.split_segment_at_utterance | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/transcribe_only | apps.api.routers.audio.transcribe_only | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/transcript.jsonl | apps.api.routers.audio.download_transcript_jsonl | ep_id | - | yes | no | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/transcript.vtt | apps.api.routers.audio.download_transcript_vtt | ep_id | - | yes | no | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/voiceprint-overrides | apps.api.routers.audio.set_voiceprint_override | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/voiceprint_refresh | apps.api.routers.audio.refresh_voiceprint_identification | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/voices/assign | apps.api.routers.audio.assign_voice_cluster | ep_id | - | no | yes | job-trigger |
| POST | /jobs/episodes/{ep_id}/audio/voices_only | apps.api.routers.audio.voices_only | ep_id | - | no | yes | job-trigger |
| GET | /jobs/episodes/{ep_id}/audio/waveform | apps.api.routers.audio.get_audio_waveform | ep_id | - | yes | no | job-trigger |
| POST | /jobs/faces_embed | apps.api.routers.jobs.run_faces_embed | - | - | no | yes | job-trigger |
| POST | /jobs/faces_embed_async | apps.api.routers.jobs.enqueue_faces_embed_async | - | - | no | yes | job-trigger |
| POST | /jobs/jobs/episode-run | apps.api.routers.jobs.start_episode_run | - | - | no | yes | job-trigger |
| POST | /jobs/jobs/facebank/backfill_display | apps.api.routers.jobs.backfill_facebank_display | - | - | no | yes | job-trigger |
| POST | /jobs/screen_time/analyze | apps.api.routers.jobs.analyze_screen_time | - | - | no | yes | job-trigger |
| DELETE | /jobs/shows/{show_id}/cast/{cast_id}/voice_reference | apps.api.routers.audio.delete_voice_reference | cast_id | - | no | yes | job-trigger |
| GET | /jobs/shows/{show_id}/cast/{cast_id}/voice_reference | apps.api.routers.audio.get_voice_reference_status | cast_id | - | yes | no | job-trigger |
| POST | /jobs/shows/{show_id}/cast/{cast_id}/voice_reference | apps.api.routers.audio.upload_voice_reference | cast_id | - | no | yes | job-trigger |
| GET | /jobs/shows/{show_id}/voice_analytics | apps.api.routers.audio.get_voice_analytics | - | - | yes | no | job-trigger |
| GET | /jobs/shows/{show_id}/voice_references | apps.api.routers.audio.list_voice_references | - | - | yes | no | job-trigger |
| POST | /jobs/track | apps.api.routers.jobs.enqueue_track | - | - | no | yes | job-trigger |
| POST | /jobs/video_export | apps.api.routers.jobs.export_video_with_overlays | - | - | no | yes | job-trigger |
| POST | /jobs/webhooks/pyannote/diarization | apps.api.routers.audio.receive_pyannote_webhook | - | - | no | yes | job-trigger |
| GET | /jobs/{job_id} | apps.api.routers.jobs.job_details | job_id | - | yes | no | job-trigger |
| POST | /jobs/{job_id}/cancel | apps.api.routers.jobs.cancel_job | job_id | - | no | yes | job-trigger |
| GET | /jobs/{job_id}/progress | apps.api.routers.jobs.get_job_progress | job_id | - | yes | no | job-trigger |
