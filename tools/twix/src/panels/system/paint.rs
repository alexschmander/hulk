use std::collections::BTreeMap;

use eframe::egui::{Color32, Grid, ProgressBar, Ui};
use egui_plot::{Legend, Line, Plot, Points};
use types::system_info::{DiskIo, MemoryUsage, NetworkIo, Tegrastats};

use super::{HISTORY, Records};

const CPU: Color32 = Color32::from_rgb(80, 190, 160);
const GPU: Color32 = Color32::from_rgb(190, 140, 240);
const RECEIVE: Color32 = Color32::from_rgb(90, 180, 240);
const TRANSMIT: Color32 = Color32::from_rgb(240, 180, 80);

struct Trace {
    name: String,
    color: Color32,
    segments: Vec<Vec<[f64; 2]>>,
}

fn trace<T>(
    records: &Records<T>,
    name: impl Into<String>,
    color: Color32,
    value: impl Fn(&T) -> Option<f64>,
) -> Trace {
    let mut trace = Trace {
        name: name.into(),
        color,
        segments: Vec::new(),
    };
    let Some(end) = records.last().map(|record| record.effective_time) else {
        return trace;
    };
    let mut segment = Vec::new();
    let mut previous = None;
    for record in records {
        let time = record.effective_time;
        let value = value(&record.value).filter(|value| value.is_finite());
        // The producer samples at 5 Hz. Leave a gap across a missed second.
        if (value.is_none()
            || previous.is_some_and(|previous| time.duration_since(previous).as_secs_f64() > 1.0))
            && !segment.is_empty()
        {
            trace.segments.push(std::mem::take(&mut segment));
        }
        if let Some(value) = value {
            segment.push([-end.duration_since(time).as_secs_f64(), value]);
        }
        previous = Some(time);
    }
    if !segment.is_empty() {
        trace.segments.push(segment);
    }
    trace
}

fn chart(ui: &mut Ui, id: &str, traces: Vec<Trace>, unit: &str, percent: bool, height: f32) {
    Plot::new(ui.id().with(id))
        .height(height)
        .include_y(0.0)
        .legend(Legend::default())
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .allow_axis_zoom_drag(false)
        .allow_double_click_reset(false)
        .y_axis_label(unit)
        .x_axis_formatter(|mark, _| format!("{:.0}s", mark.value))
        .show(ui, |plot| {
            plot.set_auto_bounds([false, !percent]);
            plot.set_plot_bounds_x(-HISTORY.as_secs_f64()..=0.0);
            if percent {
                plot.set_plot_bounds_y(0.0..=100.0);
            }
            for trace in traces {
                for segment in trace.segments {
                    if segment.len() == 1 {
                        plot.points(
                            Points::new(&trace.name, segment)
                                .color(trace.color)
                                .radius(2.0),
                        );
                    } else {
                        plot.line(Line::new(&trace.name, segment).color(trace.color));
                    }
                }
            }
        });
}

pub(super) fn cpu_mean(stats: &Tegrastats) -> Option<f64> {
    let (sum, count) = stats
        .cpus
        .iter()
        .filter_map(|cpu| cpu.utilization_percent)
        .fold((0_u64, 0_u64), |(sum, count), value| {
            (sum + u64::from(value), count + 1)
        });
    (count > 0).then(|| sum as f64 / count as f64)
}

fn meter(ui: &mut Ui, label: &str, percent: Option<f64>, detail: String, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(label);
        let text = match percent {
            Some(percent) => format!("{percent:.0}%  {detail}"),
            None => detail,
        };
        ui.add(
            ProgressBar::new((percent.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0) as f32)
                .desired_width(ui.available_width().max(40.0))
                .fill(color)
                .text(text),
        );
    });
}

pub(super) fn compute(ui: &mut Ui, stats: &Tegrastats, records: &Records<Tegrastats>) {
    let average = cpu_mean(stats);
    meter(
        ui,
        "CPU",
        average,
        if average.is_some() {
            "online core average"
        } else {
            "unavailable"
        }
        .into(),
        CPU,
    );
    meter(
        ui,
        "GPU",
        stats.gpu_utilization_percent.map(f64::from),
        if stats.gpu_utilization_percent.is_none() {
            "unavailable"
        } else {
            ""
        }
        .into(),
        GPU,
    );
    chart(
        ui,
        "utilization",
        vec![
            trace(records, "CPU average", CPU, cpu_mean),
            trace(records, "GPU", GPU, |stats| {
                stats.gpu_utilization_percent.map(f64::from)
            }),
        ],
        "%",
        true,
        (ui.available_height() - 110.0).clamp(90.0, 180.0),
    );
    let columns = if ui.available_width() >= 550.0 { 2 } else { 1 };
    let core_width =
        ((ui.available_width() - 16.0 * (columns - 1) as f32) / columns as f32).max(120.0);
    Grid::new("cpu-cores")
        .num_columns(columns)
        .spacing([16.0, 5.0])
        .show(ui, |ui| {
            for (index, cpu) in stats.cpus.iter().enumerate() {
                ui.vertical(|ui| {
                    ui.set_width(core_width);
                    let frequency = cpu
                        .frequency_mhz
                        .map(|mhz| format!("{mhz} MHz"))
                        .unwrap_or_default();
                    meter(
                        ui,
                        &format!("CPU {index}"),
                        cpu.utilization_percent.map(f64::from),
                        if cpu.utilization_percent.is_none() {
                            "offline / unavailable".into()
                        } else {
                            frequency
                        },
                        CPU,
                    );
                });
                if (index + 1) % columns == 0 {
                    ui.end_row();
                }
            }
        });
}

pub(super) fn memory_percent(memory: Option<&MemoryUsage>) -> Option<f64> {
    let memory = memory?;
    (memory.total_mib > 0).then(|| 100.0 * memory.used_mib as f64 / memory.total_mib as f64)
}

fn memory_meter(ui: &mut Ui, name: &str, memory: Option<&MemoryUsage>, color: Color32) {
    let detail = memory
        .map(|memory| {
            format!(
                "{:.2} / {:.2} GiB",
                memory.used_mib as f64 / 1024.0,
                memory.total_mib as f64 / 1024.0
            )
        })
        .unwrap_or_else(|| "unavailable".into());
    meter(ui, name, memory_percent(memory), detail, color);
}

pub(super) fn memory(ui: &mut Ui, stats: &Tegrastats, records: &Records<Tegrastats>) {
    memory_meter(ui, "RAM", stats.ram.as_ref(), RECEIVE);
    memory_meter(ui, "Swap", stats.swap.as_ref(), TRANSMIT);
    chart(
        ui,
        "memory",
        vec![
            trace(records, "RAM", RECEIVE, |stats| {
                memory_percent(stats.ram.as_ref())
            }),
            trace(records, "Swap", TRANSMIT, |stats| {
                memory_percent(stats.swap.as_ref())
            }),
        ],
        "%",
        true,
        ui.available_height().clamp(90.0, 180.0),
    );
}

pub(super) fn thermals(ui: &mut Ui, stats: &Tegrastats, records: &Records<Tegrastats>) {
    if stats.temperatures.is_empty() {
        ui.label("Temperature readings unavailable");
    } else {
        chart(
            ui,
            "temperature",
            vec![trace(records, "Hottest sensor", TRANSMIT, |stats| {
                stats
                    .temperatures
                    .iter()
                    .map(|sensor| f64::from(sensor.celsius))
                    .filter(|value| value.is_finite())
                    .reduce(f64::max)
            })],
            "°C",
            false,
            100.0,
        );
        ui.horizontal_wrapped(|ui| {
            for sensor in &stats.temperatures {
                ui.monospace(format!("{} {:.1} °C", sensor.name, sensor.celsius));
            }
        });
    }
    ui.separator();
    if stats.power.is_empty() {
        ui.label("Power readings unavailable");
    } else {
        Grid::new("power-rails").striped(true).show(ui, |ui| {
            ui.strong("Power rail");
            ui.strong("Current");
            ui.strong("Average");
            ui.end_row();
            for rail in &stats.power {
                ui.label(&rail.name);
                ui.monospace(format!(
                    "{:.2} W",
                    f64::from(rail.current_milliwatts) / 1000.0
                ));
                ui.monospace(format!(
                    "{:.2} W",
                    f64::from(rail.average_milliwatts) / 1000.0
                ));
                ui.end_row();
            }
        });
    }
}

pub(super) fn byte_rate(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        return "unavailable".into();
    }
    let (value, unit) = if value >= 1024.0 * 1024.0 * 1024.0 {
        (value / (1024.0 * 1024.0 * 1024.0), "GiB/s")
    } else if value >= 1024.0 * 1024.0 {
        (value / (1024.0 * 1024.0), "MiB/s")
    } else if value >= 1024.0 {
        (value / 1024.0, "KiB/s")
    } else {
        (value, "B/s")
    };
    format!("{value:.1} {unit}")
}

pub(super) fn network(
    ui: &mut Ui,
    devices: &BTreeMap<String, NetworkIo>,
    records: &Records<BTreeMap<String, NetworkIo>>,
    show_idle: bool,
) {
    if devices.is_empty() {
        ui.label("No network interfaces in this sample");
    }
    if !show_idle {
        ui.weak("Idle devices hidden until active. Enable Show idle devices to see all.");
    }
    for (name, device) in devices {
        if !show_idle
            && !records.iter().any(|record| {
                record.value.get(name).is_some_and(|device| {
                    let rates = &device.rates;
                    rates.receive_bytes_per_second > 0.0
                        || rates.transmit_bytes_per_second > 0.0
                        || rates.receive_errors_per_second > 0.0
                        || rates.transmit_errors_per_second > 0.0
                        || rates.receive_drops_per_second > 0.0
                        || rates.transmit_drops_per_second > 0.0
                })
            })
        {
            continue;
        }
        let rates = &device.rates;
        let title = format!(
            "{name}   RX {}   TX {}",
            byte_rate(rates.receive_bytes_per_second),
            byte_rate(rates.transmit_bytes_per_second)
        );
        eframe::egui::CollapsingHeader::new(title)
            .id_salt(name)
            .default_open(
                rates.receive_bytes_per_second > 0.0 || rates.transmit_bytes_per_second > 0.0,
            )
            .show(ui, |ui| {
                chart(
                    ui,
                    name,
                    vec![
                        trace(records, "Receive", RECEIVE, |devices| {
                            Some(devices.get(name)?.rates.receive_bytes_per_second / 1024.0)
                        }),
                        trace(records, "Transmit", TRANSMIT, |devices| {
                            Some(devices.get(name)?.rates.transmit_bytes_per_second / 1024.0)
                        }),
                    ],
                    "KiB/s",
                    false,
                    90.0,
                );
                let errors = rates.receive_errors_per_second + rates.transmit_errors_per_second;
                let drops = rates.receive_drops_per_second + rates.transmit_drops_per_second;
                let text = format!(
                    "Errors RX/TX {:.1}/{:.1} /s   Drops RX/TX {:.1}/{:.1} /s",
                    rates.receive_errors_per_second,
                    rates.transmit_errors_per_second,
                    rates.receive_drops_per_second,
                    rates.transmit_drops_per_second
                );
                if errors > 0.0 || drops > 0.0 {
                    ui.colored_label(ui.visuals().warn_fg_color, text);
                } else {
                    ui.weak(text);
                }
            });
    }
}

pub(super) fn disks(
    ui: &mut Ui,
    devices: &BTreeMap<String, DiskIo>,
    records: &Records<BTreeMap<String, DiskIo>>,
    show_idle: bool,
) {
    if devices.is_empty() {
        ui.label("No disk devices in this sample");
    }
    ui.weak("Devices and partitions are shown separately; rates are not summed.");
    if !show_idle {
        ui.weak("Idle devices hidden until active. Enable Show idle devices to see all.");
    }
    for (name, device) in devices {
        if !show_idle
            && !records.iter().any(|record| {
                record.value.get(name).is_some_and(|device| {
                    device.rates.read_operations_per_second > 0.0
                        || device.rates.write_operations_per_second > 0.0
                        || device.outstanding_requests > 0
                })
            })
        {
            continue;
        }
        let rates = &device.rates;
        let title = format!(
            "{name}   R {}   W {}",
            byte_rate(rates.read_bytes_per_second),
            byte_rate(rates.write_bytes_per_second)
        );
        eframe::egui::CollapsingHeader::new(title)
            .id_salt(name)
            .default_open(rates.read_bytes_per_second > 0.0 || rates.write_bytes_per_second > 0.0)
            .show(ui, |ui| {
                chart(
                    ui,
                    name,
                    vec![
                        trace(records, "Read", RECEIVE, |devices| {
                            Some(devices.get(name)?.rates.read_bytes_per_second / 1024.0)
                        }),
                        trace(records, "Write", TRANSMIT, |devices| {
                            Some(devices.get(name)?.rates.write_bytes_per_second / 1024.0)
                        }),
                    ],
                    "KiB/s",
                    false,
                    90.0,
                );
                ui.label(format!(
                    "IOPS R/W {:.1}/{:.1}   In flight {}",
                    rates.read_operations_per_second,
                    rates.write_operations_per_second,
                    device.outstanding_requests
                ));
                let latency = |latency: Option<f64>| {
                    latency
                        .map(|ms| format!("{ms:.2} ms"))
                        .unwrap_or_else(|| "n/a".into())
                };
                ui.weak(format!(
                    "Latency R {}   W {}",
                    latency(rates.average_read_latency_ms),
                    latency(rates.average_write_latency_ms)
                ));
            });
    }
}
