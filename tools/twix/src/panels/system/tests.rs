use eframe::egui::{CentralPanel, Context, RawInput, Rect, Shape, Vec2};
use types::system_info::{CpuInfo, MemoryUsage};

use super::*;

#[test]
fn helpers_keep_missing_metrics_unavailable_and_scale_readably() {
    let mut stats = Tegrastats::default();
    assert_eq!(paint::cpu_mean(&stats), None);
    stats.cpus = vec![
        CpuInfo::default(),
        CpuInfo {
            utilization_percent: Some(80),
            frequency_mhz: Some(1900),
        },
        CpuInfo {
            utilization_percent: Some(20),
            frequency_mhz: Some(1900),
        },
    ];
    // Offline cores do not lower the average.
    assert_eq!(paint::cpu_mean(&stats), Some(50.0));
    assert_eq!(paint::memory_percent(None), None);
    assert_eq!(
        paint::memory_percent(Some(&MemoryUsage {
            used_mib: 0,
            total_mib: 0
        })),
        None
    );
    assert_eq!(
        paint::memory_percent(Some(&MemoryUsage {
            used_mib: 1024,
            total_mib: 4096
        })),
        Some(25.0)
    );
    assert_eq!(paint::byte_rate(2048.0), "2.0 KiB/s");
    assert_eq!(paint::byte_rate(0.0), "0.0 B/s");
    assert_eq!(paint::byte_rate(f64::NAN), "unavailable");
    assert_eq!(
        paint::byte_unit(195.0 * 1024.0 * 1024.0),
        (1024.0 * 1024.0, "MiB/s")
    );
    assert_eq!(paint::nice_step(3.0), 5.0);
    assert_eq!(paint::nice_step(11.7), 20.0);
    assert_eq!(paint::nice_step(50.3), 100.0);
    assert_eq!(paint::nice_step(0.0), f64::INFINITY);
    // Views saved before the tegrastats views were merged open the merged view.
    let legacy: Settings = serde_json::from_str(r#"{"view":"Thermals"}"#).unwrap();
    assert!(legacy.view == View::System);
}

/// Renders the panel and returns all painted text.
fn render(
    panel: &mut SystemPanel,
    backend: &Arc<crate::backend::RobotBackend>,
    context: &Context,
    width: f32,
) -> String {
    let output = context.run_ui(
        RawInput {
            screen_rect: Some(Rect::from_min_size(
                Default::default(),
                Vec2::new(width, 600.0),
            )),
            ..Default::default()
        },
        |root| {
            let ctx = root.ctx().clone();
            CentralPanel::default().show(root, |ui| {
                let context = || PanelUiContext {
                    backend,
                    egui_context: &ctx,
                };
                panel.header_ui(ui, context());
                panel.ui(ui, context());
            });
        },
    );
    output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            Shape::Text(text) => Some(text.galley.text()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn panel(
    backend: &Arc<crate::backend::RobotBackend>,
    context: &Context,
    view: View,
) -> SystemPanel {
    let saved = serde_json::to_value(Settings {
        view,
        ..Default::default()
    })
    .unwrap();
    let panel = SystemPanel::new(PanelCreationContext {
        backend: backend.clone(),
        value: Some(&saved),
        egui_context: context.clone(),
    });
    assert_eq!(panel.save(), saved);
    panel
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_preset_renders_typed_metrics_and_discards_them_on_namespace_change() {
    use ros_z::context::ContextBuilder;
    use types::system_info::{DiskIoRates, NetworkIoRates, Power, Temperature};

    let router = ContextBuilder::default()
        .with_mode("router")
        .disable_multicast_scouting()
        .with_connect_endpoints(std::iter::empty::<&str>())
        .with_listen_endpoints(["tcp/127.0.0.1:0"])
        .build()
        .await
        .unwrap();
    let endpoint = router.session().info().locators().await[0].to_string();
    let backend = Arc::new(
        crate::backend::RobotBackend::new(
            tokio::runtime::Handle::current(),
            Some(endpoint),
            "/system_test".into(),
        )
        .await
        .unwrap(),
    );
    let context = Context::default();
    egui_material_icons::initialize(&context);
    let mut panels: Vec<_> = VIEWS
        .into_iter()
        .map(|view| panel(&backend, &context, view))
        .collect();
    for panel in &mut panels {
        assert!(render(panel, &backend, &context, 320.0).contains("Waiting for system/"));
    }
    let stats = backend
        .node()
        .publisher::<Tegrastats>("/system_test/system/tegrastats")
        .build()
        .await
        .unwrap();
    let network = backend
        .node()
        .publisher::<BTreeMap<String, NetworkIo>>("/system_test/system/network_io")
        .build()
        .await
        .unwrap();
    let disks = backend
        .node()
        .publisher::<BTreeMap<String, DiskIo>>("/system_test/system/disk_io")
        .build()
        .await
        .unwrap();
    let mut layout = crate::layout::TwixLayout::from_serialized(
        include_str!("../../../presets/System.json"),
        &backend,
        &context,
    )
    .unwrap();
    // GPU, swap and power are missing and one core is offline.
    let sample = Tegrastats {
        cpus: vec![
            CpuInfo {
                utilization_percent: Some(80),
                frequency_mhz: Some(1900),
            },
            CpuInfo::default(),
        ],
        gpu_utilization_percent: None,
        ram: Some(MemoryUsage {
            used_mib: 4096,
            total_mib: 8192,
        }),
        swap: None,
        temperatures: vec![Temperature {
            name: "cpu".into(),
            celsius: 65.5,
        }],
        power: Vec::<Power>::new(),
    };
    let network_sample = BTreeMap::from([(
        "test0".into(),
        NetworkIo {
            rates: NetworkIoRates {
                receive_bytes_per_second: 2048.0,
                transmit_bytes_per_second: 1024.0,
                receive_errors_per_second: 1.0,
                transmit_errors_per_second: 0.0,
                receive_drops_per_second: 0.0,
                transmit_drops_per_second: 2.0,
            },
        },
    )]);
    let disk_sample = BTreeMap::from([(
        "testdisk".into(),
        DiskIo {
            outstanding_requests: 3,
            rates: DiskIoRates {
                read_bytes_per_second: 4096.0,
                write_bytes_per_second: 2048.0,
                read_operations_per_second: 4.0,
                write_operations_per_second: 2.0,
                average_read_latency_ms: Some(1.5),
                average_write_latency_ms: None,
            },
        },
    )]);
    let count = |panel: &SystemPanel| match panel.source.as_ref().unwrap() {
        Source::Stats(source) => source.observation.get_all().len(),
        Source::Network(source) => source.observation.get_all().len(),
        Source::Disks(source) => source.observation.get_all().len(),
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        while !panels.iter().all(|panel| count(panel) >= 3) {
            stats.publish(&sample).await.unwrap();
            network.publish(&network_sample).await.unwrap();
            disks.publish(&disk_sample).await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    for _ in 0..3 {
        let _ = context.run_ui(
            RawInput {
                screen_rect: Some(Rect::from_min_size(
                    Default::default(),
                    Vec2::new(1280.0, 720.0),
                )),
                ..Default::default()
            },
            |ui| {
                CentralPanel::default().show(ui, |ui| layout.ui(ui, &backend));
            },
        );
    }
    let expected = [
        [
            "unavailable",
            "offline",
            "65.5 °C",
            "Power readings unavailable",
        ],
        [
            "RX    2.0 KiB/s",
            "TX    1.0 KiB/s",
            "Errors/s RX 1.0 TX 0.0",
            "Drops/s RX 0.0 TX 2.0",
        ],
        [
            "Read    4.0 KiB/s",
            "IOPS R/W 4.0 / 2.0",
            "In flight 3",
            "Latency R/W 1.50 ms / n/a",
        ],
    ];
    for width in [320.0, 500.0, 900.0] {
        for (panel, expected) in panels.iter_mut().zip(expected) {
            let text = render(panel, &backend, &context, width);
            for expected in expected {
                assert!(
                    text.contains(expected),
                    "{expected:?} missing at {width} px:\n{text}"
                );
            }
            assert!(!text.contains("Stale"));
        }
    }
    let Ok(Source::Stats(source)) = &mut panels[0].source else {
        panic!("the system view observes tegrastats");
    };
    source.last_update -= Duration::from_secs(3);
    assert!(
        render(&mut panels[0], &backend, &context, 500.0).contains("Stale: no new sample for 3 s")
    );
    let before = count(&panels[0]);
    tokio::time::timeout(Duration::from_secs(5), async {
        while count(&panels[0]) == before {
            stats.publish(&sample).await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(!render(&mut panels[0], &backend, &context, 500.0).contains("Stale"));
    backend.set_namespace("/empty_robot".into()).unwrap();
    for panel in &mut panels {
        let text = render(panel, &backend, &context, 500.0);
        assert!(
            text.contains("Waiting for system/"),
            "previous robot's metrics must disappear immediately"
        );
        assert_eq!(panel.built_for.as_ref().unwrap().0, "/empty_robot");
    }
    router.shutdown().unwrap();
}
