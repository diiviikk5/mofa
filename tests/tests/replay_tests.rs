use mofa_foundation::agent::components::tool::SimpleTool;
use mofa_kernel::agent::components::tool::ToolInput;
use mofa_kernel::bus::CommunicationMode;
use mofa_kernel::message::AgentMessage;
use mofa_testing::{ReplayArtifact, ReplayRunner, ScenarioRunner, ScenarioSpec, TestStatus};
use serde_json::json;

fn sample_spec() -> ScenarioSpec {
    ScenarioSpec::from_yaml_str(
        r#"
suite_name: replay-scenario
case_name: happy-path
kind: regression
target: tool-flow
clock:
  start_ms: 500
llm:
  model_name: replay-model
  responses:
    - prompt_substring: summarize
      response: "Short summary"
tools:
  - name: search
    description: Search docs
    schema:
      type: object
    stubbed_result:
      text: "Found result"
expectations:
  infer_total: 1
  tool_calls:
    - tool_name: search
      expected: 1
  bus_messages_from:
    - sender_id: evaluator
      expected: 1
"#,
    )
    .unwrap()
}

#[tokio::test]
async fn replay_artifact_round_trips_via_json() {
    let run = ScenarioRunner::new(sample_spec())
        .run_with_replay(|ctx| async move {
            let _ = ctx.infer("summarize this").await.unwrap();

            let tool = ctx.tool("search").unwrap();
            let _ = tool
                .execute(ToolInput::from_json(json!({"query": "rust"})))
                .await;

            let _ = ctx
                .bus
                .send_and_capture(
                    "evaluator",
                    CommunicationMode::Broadcast,
                    AgentMessage::TaskRequest {
                        task_id: "t1".into(),
                        content: "done".into(),
                    },
                )
                .await;

            Ok(())
        })
        .await
        .unwrap();

    let json = serde_json::to_string_pretty(&run.artifact).unwrap();
    let restored: ReplayArtifact = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, run.artifact);
    assert_eq!(restored.scenario_status, TestStatus::Passed);
}

#[tokio::test]
async fn replay_runner_matches_same_scenario_behavior() {
    let baseline = ScenarioRunner::new(sample_spec())
        .run_with_replay(|ctx| async move {
            let _ = ctx.infer("summarize this").await.unwrap();

            let tool = ctx.tool("search").unwrap();
            let _ = tool
                .execute(ToolInput::from_json(json!({"query": "rust"})))
                .await;

            let _ = ctx
                .bus
                .send_and_capture(
                    "evaluator",
                    CommunicationMode::Broadcast,
                    AgentMessage::TaskRequest {
                        task_id: "t1".into(),
                        content: "done".into(),
                    },
                )
                .await;

            Ok(())
        })
        .await
        .unwrap();

    let replay = ReplayRunner::new(baseline.artifact.clone())
        .run(sample_spec(), |ctx| async move {
            let _ = ctx.infer("summarize this").await.unwrap();

            let tool = ctx.tool("search").unwrap();
            let _ = tool
                .execute(ToolInput::from_json(json!({"query": "rust"})))
                .await;

            let _ = ctx
                .bus
                .send_and_capture(
                    "evaluator",
                    CommunicationMode::Broadcast,
                    AgentMessage::TaskRequest {
                        task_id: "t1".into(),
                        content: "done".into(),
                    },
                )
                .await;

            Ok(())
        })
        .await
        .unwrap();

    assert!(replay.comparison.matches);
    assert!(replay.comparison.drifts.is_empty());
    assert_eq!(replay.current.scenario_status, TestStatus::Passed);
}

#[tokio::test]
async fn replay_runner_reports_drift_for_changed_tool_flow() {
    let baseline = ScenarioRunner::new(sample_spec())
        .run_with_replay(|ctx| async move {
            let _ = ctx.infer("summarize this").await.unwrap();

            let tool = ctx.tool("search").unwrap();
            let _ = tool
                .execute(ToolInput::from_json(json!({"query": "rust"})))
                .await;

            let _ = ctx
                .bus
                .send_and_capture(
                    "evaluator",
                    CommunicationMode::Broadcast,
                    AgentMessage::TaskRequest {
                        task_id: "t1".into(),
                        content: "done".into(),
                    },
                )
                .await;

            Ok(())
        })
        .await
        .unwrap();

    let replay = ReplayRunner::new(baseline.artifact.clone())
        .run(sample_spec(), |ctx| async move {
            let _ = ctx.infer("summarize this").await.unwrap();

            let _ = ctx
                .bus
                .send_and_capture(
                    "evaluator",
                    CommunicationMode::Broadcast,
                    AgentMessage::TaskRequest {
                        task_id: "t1".into(),
                        content: "done".into(),
                    },
                )
                .await;

            Ok(())
        })
        .await
        .unwrap();

    assert!(!replay.comparison.matches);
    assert!(
        replay
            .comparison
            .drifts
            .iter()
            .any(|drift| drift.field == "tool_calls")
    );
}
