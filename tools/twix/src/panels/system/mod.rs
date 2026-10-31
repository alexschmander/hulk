mod paint;
#[cfg(test)]
mod tests;

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use eframe::egui::{ComboBox, ScrollArea, Ui};
use ros_z::{Message, pubsub::PublicationId, time::Time};
use ros_z_debug::{
    CachedSubscriptionStatus, ObservationPolicy, SampleRecord, TopicObservation,
    TopicObservationStatus, query::Cursor,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use types::system_info::{DiskIo, NetworkIo, Tegrastats};

use crate::{
    panel::{Panel, PanelCreationContext, PanelUiContext},
    repaint::{ObservationContext, ObservationRepaint, RepaintOnUpdates},
};

const HISTORY: Duration = Duration::from_secs(60);
const STALE_AFTER: Duration = Duration::from_secs(2);
type Records<T> = Vec<Arc<SampleRecord<T>>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum View {
    #[default]
    System,
    Network,
    Disks,
}

const VIEWS: [View; 3] = [View::System, View::Network, View::Disks];

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    view: View,
    show_idle_devices: bool,
}

pub struct SystemPanel {
    settings: Settings,
    source: Result<Source, String>,
    /// Namespace, backend presentation and view that `source` observes.
    built_for: Option<(String, (u64, u64), View)>,
}

enum Source {
    Stats(Observed<Tegrastats>),
    Network(Observed<BTreeMap<String, NetworkIo>>),
    Disks(Observed<BTreeMap<String, DiskIo>>),
}

struct Observed<T> {
    topic: &'static str,
    observation: TopicObservation<T>,
    _repaint: ObservationRepaint,
    last_publication: Option<PublicationId>,
    last_update: Instant,
}

impl<T: Message + Send + Sync + 'static> Observed<T>
where
    T::Codec: Send + Sync,
{
    fn new(context: &impl ObservationContext, topic: &'static str) -> Result<Self, String> {
        let _runtime = context.backend().runtime_handle().enter();
        let observation = ObservationPolicy::time_window(HISTORY)
            .and_then(|policy| {
                let observer = context.backend().observer();
                Ok(observer.observe_typed(topic)?.policy(policy).spawn())
            })
            .map_err(|error| format!("{topic}: {error:#}"))?;
        let repaint = observation.repaint_on_updates(context);
        Ok(Self {
            topic,
            observation,
            _repaint: repaint,
            last_publication: None,
            last_update: Instant::now(),
        })
    }

    /// Shows the observation state, then paints the latest sample with its retained history.
    fn show(&mut self, ui: &mut Ui, live: bool, paint: impl FnOnce(&mut Ui, &T, &Records<T>)) {
        let records = self
            .observation
            .window(Time::zero(), Time::from_nanos(i64::MAX));
        let latest = records.last().map(|record| record.publication_id);
        if latest != self.last_publication {
            self.last_publication = latest;
            self.last_update = Instant::now();
        }
        let status = self.observation.status();
        let ready = matches!(&status, TopicObservationStatus::Observing { cache } if cache.status() == &CachedSubscriptionStatus::Ready);
        if !ready {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                super::observation_status::format_topic_observation_status(status),
            );
        }
        if records.is_empty() {
            ui.label(format!("Waiting for {}…", self.topic));
        } else if live {
            // Use receipt progress, not clock synchronization with the robot.
            let age = self.last_update.elapsed();
            if age >= STALE_AFTER {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("Stale: no new sample for {:.0} s", age.as_secs_f64()),
                );
            }
            ui.ctx().request_repaint_after(Duration::from_secs(1));
        }
        if let Some(latest) = records.last() {
            paint(ui, &latest.value, &records);
        }
    }
}

impl Panel for SystemPanel {
    const STORAGE_ID: &'static str = "system";
    const DISPLAY_NAME: &'static str = "System metrics";
    const ICON: &'static str = egui_material_icons::icons::ICON_MONITOR_HEART.codepoint;

    fn new(context: PanelCreationContext<'_>) -> Self {
        Self {
            settings: context
                .value
                .and_then(|value| Settings::deserialize(value).ok())
                .unwrap_or_default(),
            source: Err("Not observing yet".into()),
            built_for: None,
        }
    }

    fn header_ui(&mut self, ui: &mut Ui, _context: PanelUiContext<'_>) {
        ui.horizontal_wrapped(|ui| {
            ComboBox::from_id_salt("system-view")
                .selected_text(format!("{:?}", self.settings.view))
                .show_ui(ui, |ui| {
                    for view in VIEWS {
                        ui.selectable_value(&mut self.settings.view, view, format!("{view:?}"));
                    }
                });
            if self.settings.view != View::System {
                ui.checkbox(&mut self.settings.show_idle_devices, "Show idle devices")
                    .on_hover_text("Devices active in the last 60 s are always shown");
            }
        });
    }

    fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
        // Rebuild synchronously before painting a different view, namespace, backend or seek.
        let backend = context.backend;
        let key = (
            backend.namespace(),
            backend.presentation_identity(),
            self.settings.view,
        );
        if self.built_for.as_ref() != Some(&key) {
            self.source = Source::new(&context, self.settings.view);
            self.built_for = Some(key);
        }
        let live = context.backend.query_connection().cursor.position().cursor == Cursor::Live;
        let show_idle = self.settings.show_idle_devices;
        ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| match &mut self.source {
                Err(error) => {
                    ui.colored_label(ui.visuals().error_fg_color, error);
                }
                Ok(Source::Stats(source)) => source.show(ui, live, paint::system),
                Ok(Source::Network(source)) => source.show(ui, live, |ui, latest, records| {
                    paint::devices(ui, latest, records, show_idle);
                }),
                Ok(Source::Disks(source)) => source.show(ui, live, |ui, latest, records| {
                    paint::devices(ui, latest, records, show_idle);
                }),
            });
    }

    fn save(&self) -> Value {
        serde_json::to_value(&self.settings).expect("system settings serialize")
    }
}

impl Source {
    fn new(context: &impl ObservationContext, view: View) -> Result<Self, String> {
        match view {
            View::System => Observed::new(context, "system/tegrastats").map(Self::Stats),
            View::Network => Observed::new(context, "system/network_io").map(Self::Network),
            View::Disks => Observed::new(context, "system/disk_io").map(Self::Disks),
        }
    }
}
