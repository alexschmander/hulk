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

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
enum View {
    #[default]
    Compute,
    Memory,
    Thermals,
    Network,
    Disks,
}

impl View {
    const ALL: [Self; 5] = [
        Self::Compute,
        Self::Memory,
        Self::Thermals,
        Self::Network,
        Self::Disks,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Compute => "CPU & GPU",
            Self::Memory => "Memory",
            Self::Thermals => "Temperatures & power",
            Self::Network => "Network I/O",
            Self::Disks => "Disk I/O",
        }
    }

    fn topic(self) -> &'static str {
        match self {
            Self::Compute | Self::Memory | Self::Thermals => "system/tegrastats",
            Self::Network => "system/network_io",
            Self::Disks => "system/disk_io",
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    view: View,
    show_idle_devices: bool,
}

pub struct SystemPanel {
    settings: Settings,
    source: Result<Source, String>,
    namespace: String,
    presentation: (u64, u64),
}

enum Source {
    Stats(Observed<Tegrastats>),
    Network(Observed<BTreeMap<String, NetworkIo>>),
    Disks(Observed<BTreeMap<String, DiskIo>>),
}

struct Observed<T> {
    observation: TopicObservation<T>,
    _repaint: ObservationRepaint,
    last_publication: Option<PublicationId>,
    last_update: Instant,
}

impl<T: Message + Send + Sync + 'static> Observed<T>
where
    T::Codec: Send + Sync,
{
    fn new(context: &impl ObservationContext, topic: &str) -> ros_z_debug::Result<Self> {
        let _runtime = context.backend().runtime_handle().enter();
        let observation = context
            .backend()
            .observer()
            .observe_typed(topic)?
            .policy(ObservationPolicy::time_window(HISTORY)?)
            .spawn();
        let repaint = observation.repaint_on_updates(context);
        Ok(Self {
            observation,
            _repaint: repaint,
            last_publication: None,
            last_update: Instant::now(),
        })
    }

    fn records(&mut self, ui: &mut Ui, live: bool) -> Records<T> {
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
            ui.label("Waiting for system metrics...");
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
        records
    }
}

impl Panel for SystemPanel {
    const STORAGE_ID: &'static str = "system";
    const DISPLAY_NAME: &'static str = "System";
    const ICON: &'static str = egui_material_icons::icons::ICON_MONITOR_HEART.codepoint;

    fn new(context: PanelCreationContext<'_>) -> Self {
        let settings = context
            .value
            .and_then(|value| Settings::deserialize(value).ok())
            .unwrap_or_default();
        let source = Source::new(&context, settings.view);
        Self {
            settings,
            source,
            namespace: context.backend.namespace(),
            presentation: context.backend.presentation_identity(),
        }
    }

    fn header_ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
        ui.horizontal_wrapped(|ui| {
            let previous = self.settings.view;
            ComboBox::from_id_salt("system-view")
                .selected_text(self.settings.view.label())
                .show_ui(ui, |ui| {
                    for view in View::ALL {
                        ui.selectable_value(&mut self.settings.view, view, view.label());
                    }
                });
            if matches!(self.settings.view, View::Network | View::Disks) {
                ui.checkbox(&mut self.settings.show_idle_devices, "Show idle devices");
            }
            ui.weak("60 s history")
                .on_hover_text(self.settings.view.topic());
            if previous != self.settings.view || ui.button("Reconnect").clicked() {
                self.source = Source::new(&context, self.settings.view);
            }
        });
    }

    fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
        // Rebuild synchronously before painting a different namespace, backend or seek.
        let namespace = context.backend.namespace();
        let presentation = context.backend.presentation_identity();
        if namespace != self.namespace || presentation != self.presentation {
            self.source = Source::new(&context, self.settings.view);
            self.namespace = namespace;
            self.presentation = presentation;
        }
        let live = context.backend.query_connection().cursor.position().cursor == Cursor::Live;
        ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| match &mut self.source {
                Err(error) => {
                    ui.colored_label(ui.visuals().error_fg_color, error);
                }
                Ok(Source::Stats(source)) => {
                    let records = source.records(ui, live);
                    if let Some(latest) = records.last() {
                        match self.settings.view {
                            View::Compute => paint::compute(ui, &latest.value, &records),
                            View::Memory => paint::memory(ui, &latest.value, &records),
                            View::Thermals => paint::thermals(ui, &latest.value, &records),
                            _ => unreachable!("view and source are created together"),
                        }
                    }
                }
                Ok(Source::Network(source)) => {
                    let records = source.records(ui, live);
                    if let Some(latest) = records.last() {
                        paint::network(
                            ui,
                            &latest.value,
                            &records,
                            self.settings.show_idle_devices,
                        );
                    }
                }
                Ok(Source::Disks(source)) => {
                    let records = source.records(ui, live);
                    if let Some(latest) = records.last() {
                        paint::disks(ui, &latest.value, &records, self.settings.show_idle_devices);
                    }
                }
            });
    }

    fn save(&self) -> Value {
        serde_json::to_value(&self.settings).expect("system settings serialize")
    }
}

impl Source {
    fn new(context: &impl ObservationContext, view: View) -> Result<Self, String> {
        let topic = view.topic();
        match view {
            View::Compute | View::Memory | View::Thermals => {
                Observed::new(context, topic).map(Self::Stats)
            }
            View::Network => Observed::new(context, topic).map(Self::Network),
            View::Disks => Observed::new(context, topic).map(Self::Disks),
        }
        .map_err(|error| format!("{topic}: {error:#}"))
    }
}
