mod history;

use crate::{
    backend::RobotBackend,
    panel::{Panel, PanelCreationContext, PanelUiContext, empty_state},
    repaint::{ObservationRepaint, RepaintOnUpdates},
};
use eframe::egui::{
    self, Button, Color32, FontId, Label, Popup, PopupCloseBehavior, Rect, RichText, ScrollArea,
    TextEdit, TextFormat, TextStyle, Ui, Visuals, text::LayoutJob, vec2,
};
use egui_material_icons::icons;
use ros_z::time::Time;
use ros_z_debug::{DynamicTopicObservation, ObservationPolicy, TopicObservationUpdateReceiver};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroUsize,
    sync::Arc,
    time::Duration,
};

const LEVELS: [&str; 5] = ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"];
const DEFAULT_LEVELS: [bool; 5] = [true, true, true, false, false];
/// Longer targets are abbreviated so messages stay aligned; the tooltip shows them in full.
const TARGET_WIDTH: usize = 28;

/// One log record, parsed once whenever the observation changes.
struct Entry {
    sequence: u64,
    /// Severity from 1 (error) to 5 (trace).
    level: u8,
    target: String,
    message: String,
    /// Structured fields; legacy plain-text fields have an empty name.
    fields: Vec<(String, String)>,
    simulation_time: i64,
    wall_time: i64,
    /// Lowercase target, message and fields for case-insensitive search.
    haystack: String,
}

impl Entry {
    fn parse(record: &Value) -> Self {
        let text = |name: &str| record[name].as_str().unwrap_or_default();
        let raw_fields = text("fields");
        let fields = match serde_json::from_str(raw_fields) {
            Ok(Value::Object(fields)) => fields
                .into_iter()
                .map(|(name, value)| match value {
                    Value::String(value) => (name, value),
                    value => (name, value.to_string()),
                })
                .collect(),
            _ if raw_fields.is_empty() => Vec::new(),
            _ => vec![(String::new(), raw_fields.to_owned())],
        };
        let mut entry = Self {
            sequence: record["sequence"].as_u64().unwrap_or_default(),
            level: record["level"]
                .as_u64()
                .map_or(3, |level| level.clamp(1, 5) as u8),
            target: text("target").to_owned(),
            message: text("message").to_owned(),
            fields,
            simulation_time: record["simulation_time"].as_i64().unwrap_or_default(),
            wall_time: record["wall_time"].as_i64().unwrap_or_default(),
            haystack: String::new(),
        };
        entry.haystack = format!("{} {}", entry.target, entry.text()).to_lowercase();
        entry
    }

    /// The message followed by its fields on one line, as shown in the list.
    fn text(&self) -> String {
        let mut text = single_line(&self.message);
        for (name, value) in &self.fields {
            text.push_str("  ");
            if !name.is_empty() {
                text.push_str(name);
                text.push('=');
            }
            text.push_str(&single_line(value));
        }
        text
    }

    fn line(&self) -> String {
        format!(
            "{} {:5} {}  {}",
            timestamp(self.simulation_time),
            level_name(self.level),
            self.target,
            self.text()
        )
    }

    fn repeats(&self, other: &Self) -> bool {
        self.level == other.level && self.target == other.target && self.message == other.message
    }
}

#[derive(Clone, PartialEq)]
struct Filter {
    levels: [bool; 5],
    search: String,
    hidden_targets: BTreeSet<String>,
    /// Merge consecutive entries with the same level, target and message.
    collapse: bool,
    /// Hide entries up to this wall time, set by Clear.
    cleared_until: Option<i64>,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            levels: DEFAULT_LEVELS,
            search: String::new(),
            hidden_targets: BTreeSet::new(),
            collapse: true,
            cleared_until: None,
        }
    }
}

impl Filter {
    /// Everything except severity, so level buttons can show what they would add.
    fn matches_content(&self, entry: &Entry, search: &str) -> bool {
        self.cleared_until
            .is_none_or(|cleared| entry.wall_time > cleared)
            && !self.hidden_targets.contains(&entry.target)
            && entry.haystack.contains(search)
    }

    fn matches(&self, entry: &Entry, search: &str) -> bool {
        self.levels[usize::from(entry.level - 1)] && self.matches_content(entry, search)
    }

    fn is_narrowed(&self) -> bool {
        self.levels != DEFAULT_LEVELS
            || !self.search.is_empty()
            || !self.hidden_targets.is_empty()
            || self.cleared_until.is_some()
    }
}

struct Row {
    /// The latest entry of a run of repeats.
    index: usize,
    repeats: usize,
    first: usize,
}

/// Filtered rows, recomputed only when entries or the filter change.
struct View {
    filter: Filter,
    rows: Vec<Row>,
    /// Entries per severity that pass every other filter.
    counts: [usize; 5],
    shown: usize,
}

impl View {
    fn new(entries: &[Entry], filter: &Filter) -> Self {
        let search = filter.search.to_lowercase();
        let mut counts = [0; 5];
        let mut rows: Vec<Row> = Vec::new();
        let mut shown = 0;
        for (index, entry) in entries.iter().enumerate() {
            if !filter.matches_content(entry, &search) {
                continue;
            }
            counts[usize::from(entry.level - 1)] += 1;
            if !filter.matches(entry, &search) {
                continue;
            }
            shown += 1;
            match rows.last_mut() {
                Some(row) if filter.collapse && entries[row.index].repeats(entry) => {
                    row.index = index;
                    row.repeats += 1;
                }
                _ => rows.push(Row {
                    index,
                    repeats: 1,
                    first: index,
                }),
            }
        }
        Self {
            filter: filter.clone(),
            rows,
            counts,
            shown,
        }
    }
}

enum RowAction {
    HideTarget(String),
    OnlyTarget(String),
    Search(String),
}

pub struct LogsPanel {
    backend: Arc<RobotBackend>,
    namespace: String,
    observation: Option<DynamicTopicObservation>,
    history: Option<history::History>,
    history_session: String,
    history_error: Option<String>,
    history_loading: bool,
    presentation_identity: (u64, u64),
    repaint: Option<ObservationRepaint>,
    updates: Option<TopicObservationUpdateReceiver>,
    entries: Vec<Entry>,
    /// Targets by descending entry count.
    targets: Vec<(String, usize)>,
    target_width: usize,
    error: Option<String>,
    filter: Filter,
    view: Option<View>,
    jump_to_latest: bool,
}
impl LogsPanel {
    fn connect(&mut self, context: PanelUiContext<'_>) {
        self.history = None;
        self.history_session.clear();
        self.history_error = None;
        self.history_loading = false;
        self.filter.cleared_until = None;
        self.presentation_identity = context.backend.presentation_identity();
        self.observation = None;
        self.repaint = None;
        self.updates = None;
        self.set_entries(Vec::new());
        self.backend = context.backend.clone();
        self.namespace = context.backend.namespace();
        let _runtime = context.backend.runtime_handle().enter();
        if !context.backend.is_replaying() {
            self.history = Some(history::History::start(
                context.backend.node(),
                &self.namespace,
                context.egui_context.clone(),
            ));
        }
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

    fn set_entries(&mut self, entries: Vec<Entry>) {
        let mut targets: HashMap<&str, usize> = HashMap::new();
        for entry in &entries {
            *targets.entry(&entry.target).or_default() += 1;
        }
        let mut targets: Vec<_> = targets
            .into_iter()
            .map(|(target, count)| (target.to_owned(), count))
            .collect();
        targets.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        self.target_width = targets
            .iter()
            .map(|(target, _)| target.chars().count())
            .max()
            .unwrap_or_default()
            .min(TARGET_WIDTH);
        self.targets = targets;
        self.entries = entries;
        self.view = None;
    }

    fn append_entries(&mut self, entries: impl Iterator<Item = Entry>) {
        let mut added: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            *added.entry(entry.target.clone()).or_default() += 1;
            self.entries.push(entry);
        }
        if added.is_empty() {
            return;
        }
        for (target, count) in added {
            self.target_width = self
                .target_width
                .max(target.chars().count().min(TARGET_WIDTH));
            if let Some((_, total)) = self.targets.iter_mut().find(|(name, _)| *name == target) {
                *total += count;
            } else {
                self.targets.push((target, count));
            }
        }
        self.targets
            .sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        self.view = None;
    }

    fn receive_history(&mut self) {
        // Bound work per frame even when the complete session has many pages.
        for _ in 0..4 {
            let Some(update) = self
                .history
                .as_mut()
                .and_then(|history| history.updates.try_recv().ok())
            else {
                break;
            };
            match update {
                history::Update::Page(page) => {
                    self.history_error = None;
                    self.history_loading = page.next_offset < page.end;
                    if self.history_session != page.session {
                        self.history_session = page.session;
                        self.set_entries(Vec::new());
                        self.filter.cleared_until = None;
                        // The archive is authoritative and continues from its exact cursor.
                        // Switching from the live tail cannot duplicate or omit entries.
                        self.observation = None;
                        self.repaint = None;
                        self.updates = None;
                    }
                    self.append_entries(page.records.into_iter().map(|record| {
                        Entry::parse(
                            &serde_json::to_value(record).expect("log record is JSON serializable"),
                        )
                    }));
                }
                history::Update::Error(error) => {
                    self.history_error = Some(error);
                }
            }
        }
    }

    fn view(&mut self) -> &View {
        if self
            .view
            .as_ref()
            .is_none_or(|view| view.filter != self.filter)
        {
            self.view = Some(View::new(&self.entries, &self.filter));
        }
        self.view.as_ref().unwrap()
    }

    fn level_buttons(&mut self, ui: &mut Ui) {
        let counts = self.view().counts;
        ui.spacing_mut().item_spacing.x = 2.0;
        for (index, name) in LEVELS.iter().enumerate() {
            let level = index as u8 + 1;
            let enabled = self.filter.levels[index];
            let mut job = LayoutJob::default();
            let font = TextStyle::Button.resolve(ui.style());
            let visuals = ui.visuals();
            // Room for the severity dot, painted after the button.
            job.append(
                &title_case(name),
                12.0,
                TextFormat::simple(
                    font.clone(),
                    if enabled {
                        visuals.strong_text_color()
                    } else {
                        visuals.weak_text_color()
                    },
                ),
            );
            job.append(
                &compact_count(counts[index]),
                6.0,
                TextFormat::simple(font, ui.visuals().weak_text_color()),
            );
            let response = ui
                .add(Button::selectable(self.filter.levels[index], job).frame_when_inactive(true))
                .on_hover_text(format!(
                    "{} {} entries\nClick to toggle, {}-click to show only {}",
                    counts[index],
                    name.to_lowercase(),
                    ui.ctx().format_modifiers(egui::Modifiers::COMMAND),
                    name.to_lowercase()
                ));
            let dot = egui::pos2(
                response.rect.left() + ui.spacing().button_padding.x + 4.0,
                response.rect.center().y,
            );
            let color = level_color(ui.visuals(), level);
            if enabled {
                ui.painter().circle_filled(dot, 3.5, color);
            } else {
                ui.painter().circle_stroke(dot, 3.0, (1.0, color));
            }
            if response.clicked() {
                if ui.input(|input| input.modifiers.command) {
                    self.filter.levels = [false; 5];
                    self.filter.levels[index] = true;
                } else {
                    self.filter.levels[index] ^= true;
                }
            }
        }
    }

    fn targets_button(&mut self, ui: &mut Ui) {
        let hidden = self.filter.hidden_targets.len();
        let label = if hidden == 0 {
            format!("{}  Targets", icons::ICON_FILTER_LIST.codepoint)
        } else {
            format!("{}  {hidden} hidden", icons::ICON_FILTER_LIST.codepoint)
        };
        let response = ui
            .add(Button::selectable(hidden > 0, label).frame_when_inactive(true))
            .on_hover_text("Choose which log targets to show");
        Popup::from_toggle_button_response(&response)
            .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
            .width(320.0)
            .show(|ui| {
                ui.horizontal(|ui| {
                    ui.strong("Targets");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(hidden > 0, Button::new("Show all").small())
                            .clicked()
                        {
                            self.filter.hidden_targets.clear();
                        }
                    });
                });
                ui.separator();
                let mut targets = self.targets.clone();
                // Keep hidden targets listed after a reconnect so they can be shown again.
                for target in &self.filter.hidden_targets {
                    if !targets.iter().any(|(known, _)| known == target) {
                        targets.push((target.clone(), 0));
                    }
                }
                if targets.is_empty() {
                    ui.weak("No targets yet");
                }
                // Popups size themselves to their content, so request the list's height up front.
                let row = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
                let height = (targets.len() as f32 * row).min(320.0);
                ScrollArea::vertical()
                    .min_scrolled_height(height)
                    .max_height(height)
                    .show(ui, |ui| {
                        for (target, count) in targets {
                            ui.horizontal(|ui| {
                                let mut shown = !self.filter.hidden_targets.contains(&target);
                                if ui
                                    .checkbox(&mut shown, RichText::new(&target).monospace())
                                    .changed()
                                {
                                    if shown {
                                        self.filter.hidden_targets.remove(&target);
                                    } else {
                                        self.filter.hidden_targets.insert(target.clone());
                                    }
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| ui.weak(compact_count(count)),
                                );
                            });
                        }
                    });
            });
    }

    fn copy_visible(&mut self, ui: &Ui) {
        let view = self.view.as_ref().unwrap();
        let text: Vec<String> = view
            .rows
            .iter()
            .map(|row| {
                let line = self.entries[row.index].line();
                if row.repeats > 1 {
                    format!("{line}  (×{})", row.repeats)
                } else {
                    line
                }
            })
            .collect();
        ui.ctx().copy_text(text.join("\n"));
    }

    fn list_ui(&mut self, ui: &mut Ui) {
        self.view();
        let view = self.view.as_ref().unwrap();
        let entries = &self.entries;
        let row_height = ui.text_style_height(&TextStyle::Monospace) + 6.0;
        let font = TextStyle::Monospace.resolve(ui.style());
        let mut action = None;
        let mut scroll = ScrollArea::both()
            .id_salt("log-rows")
            .stick_to_bottom(true)
            .auto_shrink([false; 2]);
        if std::mem::take(&mut self.jump_to_latest) {
            scroll = scroll.vertical_scroll_offset(view.rows.len() as f32 * row_height);
        }
        let output = ui
            .scope(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                scroll.show_rows(ui, row_height, view.rows.len(), |ui, range| {
                    for row in &view.rows[range] {
                        let entry = &entries[row.index];
                        let job =
                            row_job(entry, row.repeats, self.target_width, ui.visuals(), &font);
                        let galley = ui.painter().layout_job(job);
                        let width = ui.available_width().max(galley.size().x + 12.0);
                        let (rect, _) =
                            ui.allocate_exact_size(vec2(width, row_height), egui::Sense::hover());
                        let visuals = ui.visuals();
                        if entry.level == 1 {
                            ui.painter().rect_filled(
                                rect,
                                0,
                                visuals.error_fg_color.gamma_multiply(0.08),
                            );
                        }
                        if ui.rect_contains_pointer(rect) {
                            ui.painter()
                                .rect_filled(rect, 0, visuals.widgets.hovered.weak_bg_fill);
                        }
                        let text = Rect::from_min_size(
                            rect.min + vec2(6.0, 3.0),
                            vec2(galley.size().x, row_height - 6.0),
                        );
                        let response = ui
                            .place(text, Label::new(galley).selectable(true).extend())
                            .on_hover_ui(|ui| details_ui(ui, entries, row));
                        response.context_menu(|ui| {
                            if ui
                                .button(format!(
                                    "{}  Copy line",
                                    icons::ICON_CONTENT_COPY.codepoint
                                ))
                                .clicked()
                            {
                                ui.ctx().copy_text(entry.line());
                            }
                            if ui.button("Copy message").clicked() {
                                ui.ctx().copy_text(entry.message.clone());
                            }
                            ui.separator();
                            if ui
                                .button(format!(
                                    "Search for “{}”",
                                    truncate(&single_line(&entry.message), 32)
                                ))
                                .clicked()
                            {
                                action = Some(RowAction::Search(single_line(&entry.message)));
                            }
                            if ui.button(format!("Hide {}", entry.target)).clicked() {
                                action = Some(RowAction::HideTarget(entry.target.clone()));
                            }
                            if ui.button(format!("Show only {}", entry.target)).clicked() {
                                action = Some(RowAction::OnlyTarget(entry.target.clone()));
                            }
                        });
                    }
                })
            })
            .inner;
        let at_bottom = output.state.offset.y + output.inner_rect.height()
            >= output.content_size.y - row_height;
        if !at_bottom {
            let size = vec2(150.0, 28.0);
            let rect = Rect::from_min_size(
                output.inner_rect.right_bottom() - size - vec2(18.0, 14.0),
                size,
            );
            if ui
                .place(
                    rect,
                    Button::new(format!(
                        "{}  Jump to latest",
                        icons::ICON_ARROW_DOWNWARD.codepoint
                    ))
                    .fill(ui.visuals().selection.bg_fill),
                )
                .on_hover_text("Scroll to the newest entry and keep following")
                .clicked()
            {
                self.jump_to_latest = true;
            }
        }
        match action {
            Some(RowAction::HideTarget(target)) => {
                self.filter.hidden_targets.insert(target);
            }
            Some(RowAction::OnlyTarget(target)) => {
                self.filter.hidden_targets = self
                    .targets
                    .iter()
                    .map(|(known, _)| known.clone())
                    .filter(|known| *known != target)
                    .collect();
            }
            Some(RowAction::Search(message)) => self.filter.search = message,
            None => {}
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

fn single_line(text: &str) -> String {
    text.replace('\n', " ⏎ ")
}

fn level_name(level: u8) -> &'static str {
    LEVELS[usize::from(level.clamp(1, 5) - 1)]
}

fn level_color(visuals: &Visuals, level: u8) -> Color32 {
    match (level, visuals.dark_mode) {
        (1, _) => visuals.error_fg_color,
        (2, _) => visuals.warn_fg_color,
        (3, true) => Color32::from_rgb(100, 165, 240),
        (3, false) => Color32::from_rgb(25, 95, 190),
        (4, _) => visuals.weak_text_color(),
        _ => visuals.weak_text_color().gamma_multiply(0.7),
    }
}

fn title_case(name: &str) -> String {
    let mut characters = name.chars();
    characters.next().map_or_else(String::new, |first| {
        first.to_string() + characters.as_str().to_lowercase().as_str()
    })
}

fn compact_count(count: usize) -> String {
    match count {
        0..1000 => count.to_string(),
        1000..10_000 => format!("{:.1}k", count as f32 / 1000.0),
        _ => format!("{}k", count / 1000),
    }
}

fn truncate(text: &str, length: usize) -> String {
    if text.chars().count() <= length {
        text.to_owned()
    } else {
        text.chars().take(length - 1).chain(['…']).collect()
    }
}

/// Shortens leading module segments first, e.g. `ros_z::pubsub::subscriber` to
/// `r::pubsub::subscriber`, then truncates from the left.
fn abbreviate_target(target: &str, width: usize) -> String {
    if target.chars().count() <= width {
        return target.to_owned();
    }
    let mut segments: Vec<&str> = target.split("::").collect();
    for index in 0..segments.len().saturating_sub(1) {
        let segment = segments[index];
        segments[index] = &segment[..segment.chars().next().map_or(0, char::len_utf8)];
        let joined = segments.join("::");
        if joined.chars().count() <= width {
            return joined;
        }
    }
    let joined: Vec<char> = segments.join("::").chars().collect();
    std::iter::once('…')
        .chain(joined[joined.len() + 1 - width..].iter().copied())
        .collect()
}

fn row_job(
    entry: &Entry,
    repeats: usize,
    target_width: usize,
    visuals: &Visuals,
    font: &FontId,
) -> LayoutJob {
    let format = |color| TextFormat::simple(font.clone(), color);
    let weak = visuals.weak_text_color();
    let level = level_color(visuals, entry.level);
    let message = match entry.level {
        1 => visuals.error_fg_color,
        2 | 3 => visuals.strong_text_color(),
        _ => visuals.text_color(),
    };
    let mut job = LayoutJob::default();
    job.append(&timestamp(entry.simulation_time), 0.0, format(weak));
    job.append(
        &format!(" {:<5} ", level_name(entry.level)),
        10.0,
        TextFormat {
            background: level.gamma_multiply(0.16),
            ..format(level)
        },
    );
    job.append(
        &format!(
            "{:<target_width$}",
            abbreviate_target(&entry.target, target_width)
        ),
        10.0,
        format(weak),
    );
    job.append(&single_line(&entry.message), 10.0, format(message));
    // After the message, so messages stay aligned whether or not they repeat.
    if repeats > 1 {
        job.append(
            &format!(" ×{repeats} "),
            10.0,
            TextFormat {
                background: visuals.widgets.inactive.weak_bg_fill,
                ..format(visuals.strong_text_color())
            },
        );
    }
    for (name, value) in &entry.fields {
        if !name.is_empty() {
            job.append(name, 18.0, format(weak));
            job.append("=", 0.0, format(weak));
        }
        job.append(
            &single_line(value),
            if name.is_empty() { 18.0 } else { 0.0 },
            format(visuals.text_color()),
        );
    }
    job
}

fn details_ui(ui: &mut Ui, entries: &[Entry], row: &Row) {
    let entry = &entries[row.index];
    ui.set_max_width(520.0);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(level_name(entry.level))
                .monospace()
                .color(level_color(ui.visuals(), entry.level)),
        );
        ui.label(RichText::new(&entry.target).monospace().weak());
    });
    ui.add(Label::new(RichText::new(&entry.message).strong()).wrap());
    if !entry.fields.is_empty() {
        ui.add_space(4.0);
        egui::Grid::new("log-fields")
            .num_columns(2)
            .spacing(vec2(12.0, 2.0))
            .show(ui, |ui| {
                for (name, value) in &entry.fields {
                    ui.label(RichText::new(name).monospace().weak());
                    ui.add(Label::new(RichText::new(value).monospace()).wrap());
                    ui.end_row();
                }
            });
    }
    ui.add_space(4.0);
    if row.repeats > 1 {
        ui.weak(format!(
            "Repeated {} times since {}",
            row.repeats,
            timestamp(entries[row.first].simulation_time)
        ));
    }
    ui.weak(format!(
        "Simulation {} · wall {} · #{}",
        timestamp(entry.simulation_time),
        timestamp(entry.wall_time),
        entry.sequence
    ));
}

fn saved_filter(saved: Value) -> Filter {
    let mut filter = Filter::default();
    if let Some(levels) = saved["levels"].as_array() {
        filter.levels = [false; 5];
        for level in levels.iter().filter_map(Value::as_u64) {
            if let Some(enabled) = filter.levels.get_mut((level as usize).wrapping_sub(1)) {
                *enabled = true;
            }
        }
    } else if let Some(minimum) = saved["minimum"].as_u64().filter(|n| (1..=5).contains(n)) {
        filter.levels = std::array::from_fn(|index| index < minimum as usize);
    }
    filter.search = saved["search"].as_str().unwrap_or_default().to_owned();
    filter.hidden_targets = saved["hidden_targets"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|target| Some(target.as_str()?.to_owned()))
        .collect();
    filter.collapse = saved["collapse"].as_bool().unwrap_or(true);
    filter
}

impl Panel for LogsPanel {
    const STORAGE_ID: &'static str = "logs";
    const DISPLAY_NAME: &'static str = "Logs";
    const ICON: &'static str = egui_material_icons::icons::ICON_SUBJECT.codepoint;
    fn new(context: PanelCreationContext<'_>) -> Self {
        let filter = saved_filter(context.value.cloned().unwrap_or_default());
        let mut panel = Self {
            backend: context.backend.clone(),
            namespace: String::new(),
            observation: None,
            history: None,
            history_session: String::new(),
            history_error: None,
            history_loading: false,
            presentation_identity: (0, 0),
            repaint: None,
            updates: None,
            entries: Vec::new(),
            targets: Vec::new(),
            target_width: 0,
            error: None,
            filter,
            view: None,
            jump_to_latest: false,
        };
        panel.connect(PanelUiContext {
            backend: &context.backend,
            egui_context: &context.egui_context,
        });
        panel
    }
    fn header_ui(&mut self, ui: &mut Ui, _: PanelUiContext<'_>) {
        ui.horizontal_wrapped(|ui| {
            ui.scope(|ui| self.level_buttons(ui));
            ui.add_space(6.0);
            let search = ui.add(
                TextEdit::singleline(&mut self.filter.search)
                    .hint_text(format!("{}  Search logs", icons::ICON_SEARCH.codepoint))
                    .desired_width(180.0),
            );
            if !self.filter.search.is_empty()
                && ui
                    .add(Button::new(icons::ICON_CLOSE.codepoint).frame(false))
                    .on_hover_text("Clear search")
                    .clicked()
            {
                self.filter.search.clear();
                search.request_focus();
            }
            self.targets_button(ui);
            ui.add(
                Button::selectable(
                    self.filter.collapse,
                    format!("{}  Group repeats", icons::ICON_UNFOLD_LESS.codepoint),
                )
                .frame_when_inactive(true),
            )
            .on_hover_text("Merge consecutive entries with the same level, target and message")
            .clicked()
            .then(|| self.filter.collapse ^= true);

            if ui.available_width() < 220.0 {
                ui.end_row();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.add_space(2.0);
                let view = self.view();
                let has_rows = !view.rows.is_empty();
                let (shown, total) = (view.shown, self.entries.len());
                if ui
                    .add_enabled(
                        has_rows,
                        Button::new(icons::ICON_DELETE_SWEEP.codepoint).frame(false),
                    )
                    .on_hover_text("Clear: hide everything received so far")
                    .clicked()
                {
                    self.filter.cleared_until =
                        self.entries.iter().map(|entry| entry.wall_time).max();
                }
                if ui
                    .add_enabled(
                        has_rows,
                        Button::new(icons::ICON_CONTENT_COPY.codepoint).frame(false),
                    )
                    .on_hover_text("Copy visible entries")
                    .clicked()
                {
                    self.copy_visible(ui);
                }
                ui.add_space(6.0);
                ui.weak(if shown == total {
                    format!("{total} entries")
                } else {
                    format!("{shown} of {total}")
                });
            });
        });
    }
    fn ui(&mut self, ui: &mut Ui, context: PanelUiContext<'_>) {
        if self.namespace != context.backend.namespace()
            || !Arc::ptr_eq(&self.backend, context.backend)
            || self.presentation_identity != context.backend.presentation_identity()
        {
            self.connect(context);
        }
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        self.receive_history();
        if self.history_loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.weak("Loading full log…");
            });
        }
        if let Some(error) = &self.history_error {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Full log history unavailable; retrying…",
            )
            .on_hover_text(error);
        }
        if let Some(observation) = &self.observation {
            // Preserve the normal topic observer for MCAP replay and older publishers.
            let mut changed = self.entries.is_empty();
            if let Some(updates) = &mut self.updates {
                while let Ok(Some(_)) = updates.try_recv() {
                    changed = true;
                }
            }
            if let ros_z_debug::TopicObservationStatus::Retrying { error, .. } =
                observation.status()
            {
                ui.colored_label(ui.visuals().warn_fg_color, error);
            }
            if changed {
                let records = observation.window_json(Time::zero(), Time::from_nanos(i64::MAX));
                self.set_entries(records.iter().map(Entry::parse).collect());
            }
        }
        if self.entries.is_empty() {
            let detail = format!(
                "Nothing has been published on {}/logs yet. Retained history appears as soon as a robot connects.",
                self.namespace.trim_end_matches('/')
            );
            empty_state(
                ui,
                icons::ICON_HOURGLASS_EMPTY.codepoint,
                "Waiting for logs",
                &detail,
                |ui| {
                    ui.spinner();
                },
            );
            return;
        }
        if self.view().rows.is_empty() {
            let hidden = self.entries.len();
            if let Some(cleared) = self.filter.cleared_until {
                let mut restore = false;
                empty_state(
                    ui,
                    icons::ICON_DELETE_SWEEP.codepoint,
                    "Cleared",
                    &format!("New entries after {} appear here.", timestamp(cleared)),
                    |ui| restore = ui.button("Show cleared entries").clicked(),
                );
                if restore {
                    self.filter.cleared_until = None;
                }
            } else {
                let mut reset = false;
                empty_state(
                    ui,
                    icons::ICON_SEARCH_OFF.codepoint,
                    "No logs match",
                    &format!(
                        "All {hidden} entries are hidden by severity, search or target filters."
                    ),
                    |ui| {
                        reset = ui
                            .add_enabled(self.filter.is_narrowed(), Button::new("Reset filters"))
                            .clicked();
                    },
                );
                if reset {
                    self.filter = Filter {
                        collapse: self.filter.collapse,
                        ..Filter::default()
                    };
                }
            }
            return;
        }
        self.list_ui(ui);
    }
    fn save(&self) -> Value {
        let levels: Vec<_> = (1..=5)
            .filter(|level| self.filter.levels[level - 1])
            .collect();
        json!({
            "levels": levels,
            "search": self.filter.search,
            "hidden_targets": self.filter.hidden_targets,
            "collapse": self.filter.collapse,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn entry(level: u8, target: &str, message: &str, fields: &str) -> Entry {
        Entry::parse(
            &json!({"level": level, "target": target, "message": message, "fields": fields}),
        )
    }
    #[test]
    fn severity_search_and_targets_only_filter_displayed_records() {
        let entries = [
            entry(1, "motion", "Sensor timeout", "{}"),
            entry(4, "motion", "New sensor", "{}"),
            entry(3, "localization", "Pose", r#"{"source":"Sensor"}"#),
        ];
        let mut filter = Filter {
            search: "SENSOR".into(),
            ..Filter::default()
        };
        let view = View::new(&entries, &filter);
        assert_eq!(view.counts, [1, 0, 1, 1, 0]);
        assert_eq!(
            view.shown, 2,
            "debug is hidden by default but still counted"
        );
        filter.hidden_targets.insert("localization".into());
        filter.levels = [false, false, false, true, false];
        let view = View::new(&entries, &filter);
        assert_eq!(view.counts, [1, 0, 0, 1, 0]);
        assert_eq!(
            view.rows.iter().map(|row| row.index).collect::<Vec<_>>(),
            [1]
        );
    }
    #[test]
    fn consecutive_repeats_collapse_to_their_latest_entry() {
        let entries = [
            entry(2, "sub", "queue full", r#"{"dropped":"1"}"#),
            entry(2, "sub", "queue full", r#"{"dropped":"2"}"#),
            entry(3, "motion", "changed", "{}"),
            entry(2, "sub", "queue full", r#"{"dropped":"3"}"#),
        ];
        let view = View::new(&entries, &Filter::default());
        let rows: Vec<_> = view
            .rows
            .iter()
            .map(|row| (row.first, row.index, row.repeats))
            .collect();
        assert_eq!(rows, [(0, 1, 2), (2, 2, 1), (3, 3, 1)]);
        let expanded = View::new(
            &entries,
            &Filter {
                collapse: false,
                ..Filter::default()
            },
        );
        assert_eq!(expanded.rows.len(), 4);
    }
    #[test]
    fn long_targets_abbreviate_leading_modules_first() {
        assert_eq!(abbreviate_target("motion", 24), "motion");
        assert_eq!(
            abbreviate_target("ros_z::pubsub::subscriber", 24),
            "r::pubsub::subscriber"
        );
        assert_eq!(
            abbreviate_target("ros_z::pubsub::subscriber", 16),
            "r::p::subscriber"
        );
        assert_eq!(
            abbreviate_target("ros_z::pubsub::subscriber", 8),
            "…scriber"
        );
    }
    #[test]
    fn legacy_minimum_severity_maps_to_enabled_levels() {
        assert_eq!(
            saved_filter(json!({"minimum": 2})).levels,
            [true, true, false, false, false]
        );
        assert_eq!(
            saved_filter(json!({"levels": [1, 4]})).levels,
            [true, false, false, true, false]
        );
        assert_eq!(saved_filter(json!({})).levels, DEFAULT_LEVELS);
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
            Entry::parse(&record).text(),
            "motion command changed  motion_command=Walk { forward: 0.25, left: 0.0 }  motion_type=Walk  previous_motion_type=Stand"
        );
        assert_eq!(
            Entry::parse(&json!({"message": "plain", "fields": "{}"})).text(),
            "plain"
        );
        assert_eq!(
            Entry::parse(&json!({"message": "plain", "fields": "legacy details"})).text(),
            "plain  legacy details"
        );
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn full_session_history_survives_switches_and_live_tail_limits() {
        use robot_logs::{Archive, LogRecord};
        let scope = format!("full_logs_{}", uuid::Uuid::new_v4().simple());
        let robot_context = ros_z::context::ContextBuilder::default()
            .with_json("namespace", &scope)
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap();
        let endpoint = robot_context.session().info().locators().await[0].to_string();
        let backend = Arc::new(
            RobotBackend::new_scoped(
                tokio::runtime::Handle::current(),
                Some(endpoint),
                "/hulks/1".into(),
                Some(scope),
            )
            .await
            .unwrap(),
        );
        let node = robot_context.create_node("logs").build().await.unwrap();
        let first = Archive::start(
            &node,
            tempfile::tempfile().unwrap(),
            "/hulks/1/logs/history",
        )
        .await
        .unwrap();
        let second = Archive::start(
            &node,
            tempfile::tempfile().unwrap(),
            "/opponents/1/logs/history",
        )
        .await
        .unwrap();
        let record = |sequence, message: &str| LogRecord {
            sequence,
            simulation_time: 42,
            wall_time: sequence as i64,
            level: 3,
            target: "motion".into(),
            message: message.into(),
            fields: "{}".into(),
        };
        for sequence in 0..6000 {
            first.append(record(sequence, "first robot")).await.unwrap();
            second
                .append(record(sequence, "second robot"))
                .await
                .unwrap();
        }
        let egui = egui::Context::default();
        let mut panel = LogsPanel::new(PanelCreationContext {
            backend: backend.clone(),
            value: None,
            egui_context: egui.clone(),
            render_state: None,
        });
        async fn wait_for(
            panel: &mut LogsPanel,
            backend: &Arc<RobotBackend>,
            egui: &egui::Context,
            count: usize,
            message: &str,
        ) {
            let started = std::time::Instant::now();
            loop {
                let _ = egui.run_ui(Default::default(), |ui| {
                    panel.ui(
                        ui,
                        PanelUiContext {
                            backend,
                            egui_context: egui,
                        },
                    );
                });
                assert!(panel.entries.iter().all(|entry| entry.message == message));
                if panel.entries.len() == count && !panel.history_loading {
                    assert_eq!(
                        panel
                            .entries
                            .iter()
                            .map(|entry| entry.sequence)
                            .collect::<Vec<_>>(),
                        (0..count as u64).collect::<Vec<_>>()
                    );
                    break;
                }
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "{message}: {} of {count}, {:?}",
                    panel.entries.len(),
                    panel.history_error
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        wait_for(&mut panel, &backend, &egui, 6000, "first robot").await;
        panel.filter.cleared_until = Some(i64::MAX);
        backend.set_namespace("/opponents/1".into()).unwrap();
        wait_for(&mut panel, &backend, &egui, 6000, "second robot").await;
        assert!(panel.filter.cleared_until.is_none());
        // More than the entire former viewer limit is produced while this robot is not selected.
        for sequence in 6000..11000 {
            first.append(record(sequence, "first robot")).await.unwrap();
        }
        backend.set_namespace("/hulks/1".into()).unwrap();
        wait_for(&mut panel, &backend, &egui, 11000, "first robot").await;
        first.append(record(11000, "first robot")).await.unwrap();
        wait_for(&mut panel, &backend, &egui, 11001, "first robot").await;

        // Namespace changes while an earlier request is in flight must not leak old pages.
        backend.set_namespace("/opponents/1".into()).unwrap();
        panel.connect(PanelUiContext {
            backend: &backend,
            egui_context: &egui,
        });
        backend.set_namespace("/hulks/1".into()).unwrap();
        wait_for(&mut panel, &backend, &egui, 11001, "first robot").await;
        first.finish().await.unwrap();
        let restarted = Archive::start(
            &node,
            tempfile::tempfile().unwrap(),
            "/hulks/1/logs/history",
        )
        .await
        .unwrap();
        restarted
            .append(record(0, "replacement robot"))
            .await
            .unwrap();
        // The session identifier must clear old entries even when the namespace is unchanged.
        let started = std::time::Instant::now();
        while panel.entries.len() != 1 || panel.entries[0].message != "replacement robot" {
            let _ = egui.run_ui(Default::default(), |ui| {
                panel.ui(
                    ui,
                    PanelUiContext {
                        backend: &backend,
                        egui_context: &egui,
                    },
                )
            });
            assert!(started.elapsed() < Duration::from_secs(10));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(panel.entries[0].sequence, 0);
        let mut connection = backend.query_connection();
        connection.cursor = ros_z_debug::query::CursorHandle::new(ros_z_debug::query::Cursor::At(
            ros_z::time::Time::zero(),
        ));
        backend.set_connection(connection);
        let _ = egui.run_ui(Default::default(), |ui| {
            panel.ui(
                ui,
                PanelUiContext {
                    backend: &backend,
                    egui_context: &egui,
                },
            )
        });
        assert!(
            panel.history.is_none(),
            "MCAP replay must not load a live robot archive"
        );
        assert!(panel.entries.is_empty());
        backend.use_live();
        wait_for(&mut panel, &backend, &egui, 1, "replacement robot").await;
        drop(panel);
        restarted.finish().await.unwrap();
        second.finish().await.unwrap();
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
                    panel.entries.iter().all(|entry| entry.message == expected),
                    "old namespace leaked into viewer"
                );
                if panel.entries.len() == 512 {
                    assert_eq!(panel.entries.first().unwrap().sequence, 0);
                    assert_eq!(panel.entries.last().unwrap().sequence, 511);
                    break;
                }
                assert!(
                    started.elapsed() < Duration::from_secs(5),
                    "only {} retained logs for {namespace}: {:?}",
                    panel.entries.len(),
                    panel.observation.as_ref().unwrap().status()
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}
