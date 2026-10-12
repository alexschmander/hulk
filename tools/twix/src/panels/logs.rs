use crate::{
    backend::RobotBackend,
    panel::{Panel, PanelCreationContext, PanelUiContext},
    repaint::{ObservationRepaint, RepaintOnUpdates},
};
use eframe::egui::{self, RichText, Ui};
use ros_z::time::Time;
use ros_z_debug::{DynamicTopicObservation, ObservationPolicy, TopicObservationUpdateReceiver};
use serde_json::{Value, json};
use std::{num::NonZeroUsize, sync::Arc, time::Duration};

pub struct LogsPanel {
    backend: Arc<RobotBackend>,
    namespace: String,
    observation: Option<DynamicTopicObservation>,
    repaint: Option<ObservationRepaint>,
    updates: Option<TopicObservationUpdateReceiver>,
    records: Vec<Value>,
    error: Option<String>,
    minimum: u8,
    search: String,
    follow: bool,
}
impl LogsPanel {
    fn connect(&mut self, context: PanelUiContext<'_>) {
        self.observation = None;
        self.repaint = None;
        self.updates = None;
        self.records.clear();
        self.backend = context.backend.clone();
        self.namespace = context.backend.namespace();
        let _runtime = context.backend.runtime_handle().enter();
        match context.backend.observer().observe_dynamic("logs") {
            Ok(builder) => {
                let observation = builder
                    .policy(
                        ObservationPolicy::time_window(Duration::from_secs(86400))
                            .unwrap()
                            .with_subscriber_qos(ros_z::qos::QosProfile {
                                durability: ros_z::qos::QosDurability::TransientLocal,
                                history: ros_z::qos::QosHistory::KeepLast(
                                    NonZeroUsize::new(4096).unwrap(),
                                ),
                                ..Default::default()
                            }),
                    )
                    .spawn();
                self.updates = observation.subscribe_updates().ok();
                self.repaint = Some(observation.repaint_on_updates(&context));
                self.observation = Some(observation);
                self.error = None;
            }
            Err(error) => self.error = Some(error.to_string()),
        }
    }
}
fn timestamp(nanoseconds: i64) -> String {
    let milliseconds = nanoseconds.div_euclid(1_000_000);
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        milliseconds.div_euclid(3_600_000).rem_euclid(24),
        milliseconds.div_euclid(60_000).rem_euclid(60),
        milliseconds.div_euclid(1000).rem_euclid(60),
        milliseconds.rem_euclid(1000)
    )
}
fn message_with_fields(record: &Value) -> String {
    let mut message = record["message"].as_str().unwrap_or_default().to_owned();
    let fields = record["fields"].as_str().unwrap_or_default();
    if let Ok(Value::Object(fields)) = serde_json::from_str(fields) {
        for (name, value) in fields {
            message.push_str("  ");
            message.push_str(&name);
            message.push('=');
            if let Some(value) = value.as_str() {
                message.push_str(value);
            } else {
                message.push_str(&value.to_string());
            }
        }
    } else if !fields.is_empty() {
        message.push_str("  ");
        message.push_str(fields);
    }
    message
}
fn level_name(level: u8) -> &'static str {
    match level {
        1 => "ERROR",
        2 => "WARN",
        3 => "INFO",
        4 => "DEBUG",
        _ => "TRACE",
    }
}
fn matches(record: &Value, minimum: u8, search: &str) -> bool {
    record["level"]
        .as_u64()
        .is_some_and(|level| level <= u64::from(minimum))
        && ["target", "message", "fields"].iter().any(|field| {
            record[field]
                .as_str()
                .unwrap_or_default()
                .to_lowercase()
                .contains(search)
        })
}
impl Panel for LogsPanel {
    const STORAGE_ID: &'static str = "logs";
    const DISPLAY_NAME: &'static str = "Logs";
    const ICON: &'static str = egui_material_icons::icons::ICON_SUBJECT.codepoint;
    fn new(context: PanelCreationContext<'_>) -> Self {
        let saved = context.value.cloned().unwrap_or_default();
        let mut panel = Self {
            backend: context.backend.clone(),
            namespace: String::new(),
            observation: None,
            repaint: None,
            updates: None,
            records: Vec::new(),
            error: None,
            minimum: saved["minimum"]
                .as_u64()
                .filter(|n| (1..=5).contains(n))
                .unwrap_or(3) as u8,
            search: saved["search"].as_str().unwrap_or_default().to_owned(),
            follow: true,
        };
        panel.connect(PanelUiContext {
            backend: &context.backend,
            egui_context: &context.egui_context,
        });
        panel
    }
    fn header_ui(&mut self, ui: &mut Ui, _: PanelUiContext<'_>) {
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("log-level")
                .selected_text(level_name(self.minimum))
                .show_ui(ui, |ui| {
                    for level in (1..=5).rev() {
                        ui.selectable_value(&mut self.minimum, level, level_name(level));
                    }
                })
                .response
                .on_hover_text("Minimum displayed severity; recording is unchanged");
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Filter logs")
                    .desired_width(160.0),
            );
            ui.checkbox(&mut self.follow, "Follow")
                .on_hover_text("Scroll to new messages; collection always continues");
        });
    }
    fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
        if self.namespace != context.backend.namespace()
            || !Arc::ptr_eq(&self.backend, context.backend)
        {
            self.connect(context);
        }
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let Some(observation) = &self.observation else {
            return;
        };
        // The observer also supplies MCAP replay data and clears history when its target changes.
        let mut changed = self.records.is_empty();
        if let Some(updates) = &mut self.updates {
            while let Ok(Some(_)) = updates.try_recv() {
                changed = true;
            }
        }
        if changed {
            self.records = observation.window_json(Time::zero(), Time::from_nanos(i64::MAX));
        }
        if let ros_z_debug::TopicObservationStatus::Retrying { error, .. } = observation.status() {
            ui.colored_label(ui.visuals().warn_fg_color, error);
        }
        if self.records.is_empty() {
            ui.weak("Waiting for logs from the connected robot…");
            return;
        }
        let search = self.search.to_lowercase();
        let records: Vec<_> = self
            .records
            .iter()
            .filter(|r| matches(r, self.minimum, &search))
            .collect();
        egui::ScrollArea::both()
            .stick_to_bottom(self.follow)
            .auto_shrink([false; 2])
            .show_rows(
                ui,
                ui.text_style_height(&egui::TextStyle::Monospace),
                records.len(),
                |ui, range| {
                    for record in &records[range] {
                        let level = record["level"].as_u64().unwrap_or(3) as u8;
                        let color = match level {
                            1 => ui.visuals().error_fg_color,
                            2 => ui.visuals().warn_fg_color,
                            4 | 5 => ui.visuals().weak_text_color(),
                            _ => ui.visuals().text_color(),
                        };
                        let time =
                            timestamp(record["simulation_time"].as_i64().unwrap_or_default());
                        let text = format!(
                            "{time} {:5} {}  {}",
                            level_name(level),
                            record["target"].as_str().unwrap_or_default(),
                            message_with_fields(record).replace('\n', " ⏎ ")
                        );
                        ui.add(
                            egui::Label::new(RichText::new(text).monospace().color(color))
                                .selectable(true)
                                .extend(),
                        )
                        .on_hover_text(format!(
                            "Simulation time: {} ns\nWall time: {} ({} ns)\n{}",
                            record["simulation_time"],
                            timestamp(record["wall_time"].as_i64().unwrap_or_default()),
                            record["wall_time"],
                            message_with_fields(record)
                        ));
                    }
                },
            );
    }
    fn save(&self) -> Value {
        json!({"minimum": self.minimum, "search": self.search})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn severity_and_search_only_filter_displayed_records() {
        let error =
            json!({"level":1, "target":"motion", "message":"Sensor timeout", "fields":"{}"});
        let debug = json!({"level":4, "target":"motion", "message":"New sensor", "fields":"{}"});
        assert!(matches(&error, 2, "sensor"));
        assert!(!matches(&debug, 2, "sensor"));
        assert!(matches(&debug, 5, "motion"));
        assert!(!matches(&error, 5, "localization"));
    }
    #[test]
    fn displayed_message_includes_structured_fields_without_truncation() {
        let record = json!({
            "message": "motion command changed",
            "fields": json!({
                "motion_command": "Walk { forward: 0.25, left: 0.0 }",
                "motion_type": "Walk",
                "previous_motion_type": "Stand"
            }).to_string()
        });
        assert_eq!(
            message_with_fields(&record),
            "motion command changed  motion_command=Walk { forward: 0.25, left: 0.0 }  motion_type=Walk  previous_motion_type=Stand"
        );
        assert_eq!(
            message_with_fields(&json!({"message": "plain", "fields": "{}"})),
            "plain"
        );
        assert_eq!(
            message_with_fields(&json!({"message": "plain", "fields": "legacy details"})),
            "plain  legacy details"
        );
    }
    #[derive(Clone, serde::Serialize, serde::Deserialize, ros_z::Message)]
    struct TestLog {
        sequence: u64,
        level: u8,
        target: String,
        message: String,
        fields: String,
        simulation_time: i64,
        wall_time: i64,
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn viewer_replays_local_history_and_clears_on_namespace_switch() {
        check_history_and_namespace(false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn viewer_replays_remote_history_and_clears_on_namespace_switch() {
        check_history_and_namespace(true).await;
    }

    async fn check_history_and_namespace(remote: bool) {
        let scope = format!("logs_test_{}", uuid::Uuid::new_v4().simple());
        let robot_context = if remote {
            Some(
                ros_z::context::ContextBuilder::default()
                    .with_json("namespace", &scope)
                    .with_mode("router")
                    .disable_multicast_scouting()
                    .with_connect_endpoints(std::iter::empty::<&str>())
                    .with_listen_endpoints(["tcp/127.0.0.1:0"])
                    .build()
                    .await
                    .unwrap(),
            )
        } else {
            None
        };
        let router = if let Some(context) = &robot_context {
            Some(context.session().info().locators().await[0].to_string())
        } else {
            None
        };
        let backend = Arc::new(
            RobotBackend::new_scoped(
                tokio::runtime::Handle::current(),
                router,
                "/log_test_a".into(),
                Some(scope),
            )
            .await
            .unwrap(),
        );
        let node = if let Some(context) = &robot_context {
            Arc::new(context.create_node("robot_logs").build().await.unwrap())
        } else {
            backend.node()
        };
        let a = node
            .publisher::<TestLog>("/log_test_a/logs")
            .qos(ros_z::qos::QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                history: ros_z::qos::QosHistory::KeepLast(
                    std::num::NonZeroUsize::new(512).unwrap(),
                ),
                ..Default::default()
            })
            .build()
            .await
            .unwrap();
        let b = node
            .publisher::<TestLog>("/log_test_b/logs")
            .qos(ros_z::qos::QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                history: ros_z::qos::QosHistory::KeepLast(
                    std::num::NonZeroUsize::new(512).unwrap(),
                ),
                ..Default::default()
            })
            .build()
            .await
            .unwrap();
        for (publisher, message) in [(&a, "robot a"), (&b, "robot b")] {
            for sequence in 0..512 {
                publisher
                    .publish_with_source_time(
                        &TestLog {
                            sequence,
                            level: 3,
                            target: "motion".into(),
                            message: message.into(),
                            fields: "{}".into(),
                            simulation_time: 123,
                            wall_time: 456,
                        },
                        Time::from_nanos(123),
                    )
                    .await
                    .unwrap();
            }
        }
        let egui = egui::Context::default();
        let mut panel = LogsPanel::new(PanelCreationContext {
            backend: backend.clone(),
            value: None,
            egui_context: egui.clone(),
            render_state: None,
        });
        for (namespace, expected) in [("/log_test_a", "robot a"), ("/log_test_b", "robot b")] {
            backend.set_namespace(namespace.into()).unwrap();
            let started = std::time::Instant::now();
            loop {
                let _ = egui.run_ui(Default::default(), |ui| {
                    panel.ui(
                        ui,
                        PanelUiContext {
                            backend: &backend,
                            egui_context: &egui,
                        },
                    );
                });
                assert!(
                    panel
                        .records
                        .iter()
                        .all(|record| record["message"] == expected),
                    "old namespace leaked into viewer"
                );
                if panel.records.len() == 512 {
                    assert_eq!(panel.records.first().unwrap()["sequence"], 0);
                    assert_eq!(panel.records.last().unwrap()["sequence"], 511);
                    break;
                }
                assert!(
                    started.elapsed() < Duration::from_secs(5),
                    "only {} retained logs for {namespace}: {:?}",
                    panel.records.len(),
                    panel.observation.as_ref().unwrap().status()
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}
