use std::collections::BTreeMap;

use eframe::egui::{
    Color32, Grid, Label, ProgressBar, Rect, RichText, StrokeKind, Ui,
    collapsing_header::CollapsingState, vec2,
};
use egui_plot::{AxisHints, HoverPosition, Line, Plot, Points, uniform_grid_spacer};
use types::system_info::{DiskIo, MemoryUsage, NetworkIo, Tegrastats};

use super::{HISTORY, Records};

const TEAL: Color32 = Color32::from_rgb(80, 190, 160);
const PURPLE: Color32 = Color32::from_rgb(190, 140, 240);
const BLUE: Color32 = Color32::from_rgb(90, 180, 240);
const AMBER: Color32 = Color32::from_rgb(240, 180, 80);

/// Name, colour and line segments split at missing samples.
type Trace = (&'static str, Color32, Vec<Vec<[f64; 2]>>);
/// Name, percentage, detail (shown alone when the percentage is missing) and colour.
type Meter = (String, Option<f64>, String, Color32);

fn trace<T>(
    records: &Records<T>,
    name: &'static str,
    color: Color32,
    value: impl Fn(&T) -> Option<f64>,
) -> Trace {
    let mut segments: Vec<Vec<[f64; 2]>> = Vec::new();
    let end = records.last().map(|record| record.effective_time);
    let mut previous = None;
    for record in records {
        let time = record.effective_time;
        let Some(value) = value(&record.value).filter(|value| value.is_finite()) else {
            previous = None;
            continue;
        };
        // The producer samples at 5 Hz. Break the line at a missing value or a missed second.
        if previous.is_none_or(|previous| time.duration_since(previous).as_secs_f64() > 1.0) {
            segments.push(Vec::new());
        }
        previous = Some(time);
        let age = end
            .expect("the current record belongs to a nonempty history")
            .duration_since(time);
        segments
            .last_mut()
            .expect("a segment was started above")
            .push([-age.as_secs_f64(), value]);
    }
    (name, color, segments)
}

/// Smallest 1-2-5 step of at least `min_step`; infinite (no ticks) for a degenerate axis.
pub(super) fn nice_step(min_step: f64) -> f64 {
    let magnitude = 10_f64.powf(min_step.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|factor| factor * magnitude)
        .find(|step| *step > 0.0 && *step >= min_step)
        .unwrap_or(f64::INFINITY)
}

fn chart(ui: &mut Ui, id: &str, traces: Vec<Trace>, unit: &str, top: f64, height: f32) {
    // Grid and crosshair only guide reading, as in the Plot panel.
    let guide = ui.visuals().text_color().gamma_multiply(0.3);
    let plot = Plot::new(ui.id().with(id))
        .height(height)
        .include_y(0.0)
        .include_y(top)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .allow_axis_zoom_drag(false)
        .allow_double_click_reset(false)
        .grid_color(guide)
        .cursor_color(guide)
        // Steps of at least 24 px give short charts two or more fully visible tick labels.
        .custom_y_axes(vec![
            AxisHints::new_y()
                .min_thickness(48.0)
                .label(unit)
                .label_spacing(16.0..=20.0)
                // Enough decimals for the step: 0.5 steps need one, which egui_plot rounds away.
                .formatter(|mark, _| {
                    let decimals = (-mark.step_size.log10() - 1e-6).ceil().max(0.0) as usize;
                    format!("{:.decimals$}", mark.value)
                }),
        ])
        .y_grid_spacer(uniform_grid_spacer(|input| {
            [nice_step(3.0 * input.base_step_size); 3]
        }))
        .x_grid_spacer(uniform_grid_spacer(|input| {
            [nice_step(10.0 * input.base_step_size); 3]
        }))
        .x_axis_formatter(|mark, _| format!("{:.0}s", mark.value))
        .label_formatter(|hover| match hover {
            HoverPosition::NearDataPoint {
                plot_name,
                position,
                ..
            } => Some(format!(
                "{plot_name}\n{:.1} {unit}\nat {:.1}s",
                position.y, position.x
            )),
            HoverPosition::Elsewhere { .. } => None,
        });
    // Charts only display data. Unless the pointer reads values, keep the plot and its axes out of
    // the Tab order, without the dimming of disabled widgets.
    let rect = Rect::from_min_size(ui.cursor().min, vec2(ui.available_width(), height));
    ui.scope(|ui| {
        ui.visuals_mut().disabled_alpha = 1.0;
        if !ui.rect_contains_pointer(rect) {
            ui.disable();
        }
        plot.show(ui, |plot| {
            plot.set_auto_bounds([false, true]);
            plot.set_plot_bounds_x(-HISTORY.as_secs_f64()..=0.0);
            for (name, color, segments) in traces {
                for segment in segments {
                    if segment.len() == 1 {
                        plot.points(Points::new(name, segment).color(color).radius(2.0));
                    } else {
                        plot.line(Line::new(name, segment).color(color));
                    }
                }
            }
        });
    });
}

/// Monospace tokens that wrap between, never inside, values.
fn tokens(ui: &mut Ui, tokens: impl IntoIterator<Item = RichText>) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 16.0;
        for token in tokens {
            ui.add(Label::new(token.monospace()).extend());
        }
    });
}

/// Average over online cores only.
pub(super) fn cpu_mean(stats: &Tegrastats) -> Option<f64> {
    let online: Vec<_> = stats
        .cpus
        .iter()
        .filter_map(|cpu| cpu.utilization_percent)
        .collect();
    (!online.is_empty())
        .then(|| online.iter().map(|&value| f64::from(value)).sum::<f64>() / online.len() as f64)
}

pub(super) fn memory_percent(memory: Option<&MemoryUsage>) -> Option<f64> {
    let memory = memory?;
    (memory.total_mib > 0).then(|| 100.0 * memory.used_mib as f64 / memory.total_mib as f64)
}

fn hottest(stats: &Tegrastats) -> Option<f64> {
    stats
        .temperatures
        .iter()
        .map(|sensor| f64::from(sensor.celsius))
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

pub(super) fn system(ui: &mut Ui, stats: &Tegrastats, records: &Records<Tegrastats>) {
    let memory = |name: &str, memory: Option<&MemoryUsage>, color| -> Meter {
        let detail = memory.map_or_else(String::new, |memory| {
            let [used, total] = [memory.used_mib, memory.total_mib].map(|mib| mib as f64 / 1024.0);
            format!("{used:.2} / {total:.2} GiB")
        });
        (name.into(), memory_percent(memory), detail, color)
    };
    let mut meters = vec![("CPU".into(), cpu_mean(stats), String::new(), TEAL)];
    meters.extend(stats.cpus.iter().enumerate().map(|(index, cpu)| {
        let detail = match (cpu.utilization_percent, cpu.frequency_mhz) {
            (None, _) => "offline".into(),
            (Some(_), mhz) => mhz.map_or_else(String::new, |mhz| format!("{mhz:>4} MHz")),
        };
        let percent = cpu.utilization_percent.map(f64::from);
        (format!("CPU {index}"), percent, detail, TEAL)
    }));
    let gpu = stats.gpu_utilization_percent.map(f64::from);
    meters.extend([
        ("GPU".into(), gpu, String::new(), PURPLE),
        memory("RAM", stats.ram.as_ref(), BLUE),
        memory("Swap", stats.swap.as_ref(), AMBER),
    ]);
    // Both charts share the height the text rows leave, so a tall pane fills without scrolling.
    let text_rows = meters.len() + stats.power.len() + 6;
    let row = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
    let chart_height = ((ui.available_height() - text_rows as f32 * row) / 2.0).max(100.0);

    // Values sit beside, not on, the fill; the last column stretches the gauges to the edge.
    let right = ui.max_rect().right();
    Grid::new("meters").num_columns(3).show(ui, |ui| {
        for (name, percent, detail, color) in meters {
            ui.monospace(name);
            ui.monospace(match percent {
                Some(percent) => format!("{percent:>3.0}% {detail}"),
                None if detail.is_empty() => "unavailable".into(),
                None => detail,
            });
            let fraction = (percent.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0) as f32;
            ui.add(
                ProgressBar::new(fraction)
                    .desired_width((right - ui.cursor().left()).max(0.0))
                    .desired_height(6.0)
                    .corner_radius(0)
                    .fill(color),
            );
            ui.end_row();
        }
    });
    let traces = vec![
        trace(records, "CPU (online cores)", TEAL, cpu_mean),
        trace(records, "GPU", PURPLE, |stats| {
            stats.gpu_utilization_percent.map(f64::from)
        }),
        trace(records, "RAM", BLUE, |stats| {
            memory_percent(stats.ram.as_ref())
        }),
        trace(records, "Swap", AMBER, |stats| {
            memory_percent(stats.swap.as_ref())
        }),
    ];
    chart(ui, "utilization", traces, "%", 100.0, chart_height);

    ui.separator();
    let [text_color, hottest_color] = [ui.visuals().text_color(), ui.visuals().strong_text_color()];
    let peak = hottest(stats);
    let width = stats.temperatures.iter().map(|sensor| sensor.name.len());
    let width = width.max().unwrap_or(0);
    tokens(
        ui,
        stats.temperatures.iter().map(|sensor| {
            let text = format!("{:<width$} {:>5.1} °C", sensor.name, sensor.celsius);
            // The plotted hottest sensor shares the line's colour.
            let hottest = Some(f64::from(sensor.celsius)) == peak;
            RichText::new(text).color(if hottest { hottest_color } else { text_color })
        }),
    );
    if stats.temperatures.is_empty() {
        ui.label("Temperatures unavailable");
    } else {
        let traces = vec![trace(records, "Hottest sensor", hottest_color, hottest)];
        chart(ui, "temperature", traces, "°C", 100.0, chart_height);
    }

    ui.separator();
    if stats.power.is_empty() {
        ui.label("Power readings unavailable");
    }
    Grid::new("power").num_columns(3).show(ui, |ui| {
        for rail in &stats.power {
            let [now, average] = [rail.current_milliwatts, rail.average_milliwatts]
                .map(|milliwatts| f64::from(milliwatts) / 1000.0);
            ui.monospace(&rail.name);
            ui.monospace(format!("{now:>6.2} W"))
                .on_hover_text("Current power");
            ui.monospace(format!("{average:>6.2} W avg"));
            ui.end_row();
        }
    });
}

pub(super) fn byte_unit(value: f64) -> (f64, &'static str) {
    [(1 << 30, "GiB/s"), (1 << 20, "MiB/s"), (1 << 10, "KiB/s")]
        .into_iter()
        .map(|(scale, unit)| (f64::from(scale), unit))
        .find(|(scale, _)| value >= *scale)
        .unwrap_or((1.0, "B/s"))
}

pub(super) fn byte_rate(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        return "unavailable".into();
    }
    let (scale, unit) = byte_unit(value);
    format!("{:.1} {unit}", value / scale)
}

/// A network interface or disk.
pub(super) trait Device {
    /// Labels and colours of the two byte rates.
    const SERIES: [(&str, Color32); 2];
    fn rates(&self) -> [f64; 2];
    fn active(&self) -> bool;
    /// Values shown below the chart.
    fn details(&self, warn: Color32) -> Vec<RichText>;
}

impl Device for NetworkIo {
    const SERIES: [(&str, Color32); 2] = [("RX", BLUE), ("TX", AMBER)];

    fn rates(&self) -> [f64; 2] {
        [
            self.rates.receive_bytes_per_second,
            self.rates.transmit_bytes_per_second,
        ]
    }

    fn active(&self) -> bool {
        let r = &self.rates;
        self.rates().iter().sum::<f64>() > 0.0
            || r.receive_errors_per_second
                + r.transmit_errors_per_second
                + r.receive_drops_per_second
                + r.transmit_drops_per_second
                > 0.0
    }

    /// Errors and drops, only when there are any.
    fn details(&self, warn: Color32) -> Vec<RichText> {
        let rates = &self.rates;
        let [
            receive_errors,
            transmit_errors,
            receive_drops,
            transmit_drops,
        ] = [
            rates.receive_errors_per_second,
            rates.transmit_errors_per_second,
            rates.receive_drops_per_second,
            rates.transmit_drops_per_second,
        ];
        if receive_errors + transmit_errors + receive_drops + transmit_drops <= 0.0 {
            return Vec::new();
        }
        [
            format!("Errors/s RX {receive_errors:.1} TX {transmit_errors:.1}"),
            format!("Drops/s RX {receive_drops:.1} TX {transmit_drops:.1}"),
        ]
        .map(|text| RichText::new(text).color(warn))
        .into()
    }
}

impl Device for DiskIo {
    const SERIES: [(&str, Color32); 2] = [("Read", BLUE), ("Write", AMBER)];

    fn rates(&self) -> [f64; 2] {
        [
            self.rates.read_bytes_per_second,
            self.rates.write_bytes_per_second,
        ]
    }

    fn active(&self) -> bool {
        let rates = &self.rates;
        rates.read_operations_per_second + rates.write_operations_per_second > 0.0
            || self.outstanding_requests > 0
    }

    fn details(&self, _warn: Color32) -> Vec<RichText> {
        let rates = &self.rates;
        let latency =
            |latency: Option<f64>| latency.map_or("n/a".into(), |ms| format!("{ms:.2} ms"));
        [
            format!(
                "IOPS R/W {:.1} / {:.1}",
                rates.read_operations_per_second, rates.write_operations_per_second
            ),
            format!("In flight {}", self.outstanding_requests),
            format!(
                "Latency R/W {} / {}",
                latency(rates.average_read_latency_ms),
                latency(rates.average_write_latency_ms)
            ),
        ]
        .map(RichText::new)
        .into()
    }
}

/// Devices active in the retained history (or all), each with its rates, chart and details.
pub(super) fn devices<T: Device>(
    ui: &mut Ui,
    devices: &BTreeMap<String, T>,
    records: &Records<BTreeMap<String, T>>,
    show_idle: bool,
) {
    let active = |name: &String| {
        records
            .iter()
            .any(|record| record.value.get(name).is_some_and(T::active))
    };
    let shown: Vec<_> = devices
        .iter()
        .filter(|(name, _)| show_idle || active(name))
        .collect();
    if shown.is_empty() {
        ui.label(if devices.is_empty() {
            "No devices in this sample"
        } else {
            "No device was active in the last 60 s"
        });
    }
    let width = devices.keys().map(String::len).max().unwrap_or(0);
    let row = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
    let height = ui.available_height() / shown.len().max(1) as f32 - 2.0 * row;
    for (name, device) in shown {
        let (toggle, ..) =
            CollapsingState::load_with_default_open(ui.ctx(), ui.make_persistent_id(name), true)
                .show_header(ui, |ui| {
                    let rates = T::SERIES.iter().zip(device.rates());
                    let rates = rates.map(|((label, color), rate)| {
                        RichText::new(format!("{label} {:>12}", byte_rate(rate))).color(*color)
                    });
                    tokens(
                        ui,
                        std::iter::once(format!("{name:<width$}").into()).chain(rates),
                    );
                })
                .body(|ui| {
                    if !active(name) {
                        return;
                    }
                    let max = records
                        .iter()
                        .filter_map(|record| Some(record.value.get(name)?.rates()))
                        .flatten()
                        .fold(0.0, f64::max);
                    let (scale, unit) = byte_unit(max);
                    let traces = (0..2).map(|index| {
                        let (label, color) = T::SERIES[index];
                        trace(records, label, color, |devices| {
                            Some(devices.get(name)?.rates()[index] / scale)
                        })
                    });
                    let height = height.clamp(80.0, 180.0);
                    chart(ui, name, traces.collect(), unit, 1.0, height);
                    tokens(ui, device.details(ui.visuals().warn_fg_color));
                });
        // The arrow is the focusable toggle; show keyboard focus like other widgets do.
        if toggle.has_focus() {
            let stroke = ui.visuals().widgets.active.bg_stroke;
            ui.painter()
                .rect_stroke(toggle.rect, 2.0, stroke, StrokeKind::Outside);
        }
    }
}
