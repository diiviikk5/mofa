//! Replay artifacts and deterministic replay helpers for scenario regression.

use crate::clock::Clock;
use crate::report::{TestReport, TestStatus};
use crate::scenario::{ScenarioContext, ScenarioRunner, ScenarioSpec};
use mofa_foundation::agent::components::tool::SimpleTool;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::future::Future;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayArtifact {
    pub suite_name: String,
    pub case_name: Option<String>,
    pub kind: Option<String>,
    pub target: Option<String>,
    pub model_name: String,
    pub timestamp_ms: u64,
    pub final_clock_ms: u64,
    pub scenario_status: TestStatus,
    pub infer_history: Vec<String>,
    pub tool_calls: Vec<ReplayToolCall>,
    pub bus_messages: Vec<ReplayBusMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayToolCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub raw_input: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayBusMessage {
    pub sender_id: String,
    pub mode: String,
    pub message: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRun {
    pub report: TestReport,
    pub artifact: ReplayArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayRunner {
    baseline: ReplayArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayComparisonResult {
    pub matches: bool,
    pub drifts: Vec<ReplayDrift>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayDrift {
    pub field: String,
    pub expected: String,
    pub actual: String,
}

impl ReplayArtifact {
    pub async fn capture(
        spec: &ScenarioSpec,
        context: &ScenarioContext,
        report: &TestReport,
    ) -> Self {
        let mut tool_calls = Vec::new();
        let mut tool_names = context.tool_names();
        tool_names.sort();

        for tool_name in tool_names {
            if let Some(tool) = context.tool(&tool_name) {
                for call in tool.history().await {
                    tool_calls.push(ReplayToolCall {
                        tool_name: tool.name().to_string(),
                        arguments: call.arguments,
                        raw_input: call.raw_input,
                    });
                }
            }
        }

        let bus_messages = context
            .bus
            .captured_messages
            .read()
            .await
            .iter()
            .map(|(sender_id, mode, message)| ReplayBusMessage {
                sender_id: sender_id.clone(),
                mode: format!("{mode:?}"),
                message: serde_json::to_value(message)
                    .unwrap_or_else(|_| serde_json::Value::String(format!("{message:?}"))),
            })
            .collect();

        let scenario_status = report
            .results
            .iter()
            .find(|case| case.name == "scenario_execution")
            .map(|case| case.status.clone())
            .unwrap_or(TestStatus::Skipped);

        Self {
            suite_name: spec.suite_name.clone(),
            case_name: spec.case_name.clone(),
            kind: spec.kind.clone(),
            target: spec.target.clone(),
            model_name: context.model_name.clone(),
            timestamp_ms: report.timestamp,
            final_clock_ms: context.clock.now_millis(),
            scenario_status,
            infer_history: context.backend.infer_history(),
            tool_calls,
            bus_messages,
        }
    }

    pub fn compare(&self, current: &ReplayArtifact) -> ReplayComparisonResult {
        let mut drifts = Vec::new();

        compare_field(
            &mut drifts,
            "scenario_status",
            format!("{:?}", self.scenario_status),
            format!("{:?}", current.scenario_status),
        );
        compare_field(
            &mut drifts,
            "infer_history",
            stable_json(&self.infer_history),
            stable_json(&current.infer_history),
        );
        compare_field(
            &mut drifts,
            "tool_calls",
            stable_json(&self.tool_calls),
            stable_json(&current.tool_calls),
        );
        compare_field(
            &mut drifts,
            "bus_messages",
            stable_json(&self.bus_messages),
            stable_json(&current.bus_messages),
        );
        compare_field(
            &mut drifts,
            "final_clock_ms",
            self.final_clock_ms.to_string(),
            current.final_clock_ms.to_string(),
        );

        ReplayComparisonResult {
            matches: drifts.is_empty(),
            drifts,
        }
    }
}

impl ReplayRunner {
    pub fn new(baseline: ReplayArtifact) -> Self {
        Self { baseline }
    }

    pub fn baseline(&self) -> &ReplayArtifact {
        &self.baseline
    }

    pub async fn run<F, Fut>(
        &self,
        spec: ScenarioSpec,
        scenario: F,
    ) -> anyhow::Result<ReplayExecution>
    where
        F: FnOnce(ScenarioContext) -> Fut,
        Fut: Future<Output = Result<(), String>>,
    {
        let run = ScenarioRunner::new(spec).run_with_replay(scenario).await?;
        let comparison = self.baseline.compare(&run.artifact);

        Ok(ReplayExecution {
            baseline: self.baseline.clone(),
            current: run.artifact,
            comparison,
            report: run.report,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayExecution {
    pub baseline: ReplayArtifact,
    pub current: ReplayArtifact,
    pub comparison: ReplayComparisonResult,
    pub report: TestReport,
}

fn compare_field(drifts: &mut Vec<ReplayDrift>, field: &str, expected: String, actual: String) {
    if expected != actual {
        drifts.push(ReplayDrift {
            field: field.to_string(),
            expected,
            actual,
        });
    }
}

fn stable_json<T: Serialize>(value: &T) -> String {
    let mut normalized = serde_json::to_value(value).unwrap_or(serde_json::Value::Null);
    sort_json_value(&mut normalized);
    serde_json::to_string(&normalized).unwrap_or_else(|_| "null".to_string())
}

fn sort_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (key, mut nested) in std::mem::take(map) {
                sort_json_value(&mut nested);
                sorted.insert(key, nested);
            }
            *map = sorted.into_iter().collect();
        }
        serde_json::Value::Array(items) => {
            for item in items {
                sort_json_value(item);
            }
        }
        _ => {}
    }
}
