use eframe::egui::{CentralPanel, Context, RawInput, Rect, Vec2};
use types::system_info::{CpuInfo, MemoryUsage};

use super::*;

#[test]
fn missing_metrics_remain_unavailable_and_offline_cores_do_not_lower_average() {
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
}

fn backend(
    router: Option<String>,
    namespace: &str,
) -> (tokio::runtime::Runtime, Arc<crate::backend::RobotBackend>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let backend = Arc::new(
        runtime
            .block_on(crate::backend::RobotBackend::new(
                runtime.handle().clone(),
                router,
                namespace.into(),
            ))
            .unwrap(),
    );
    (runtime, backend)
}

fn render(
    panel: &mut SystemPanel,
    backend: &Arc<crate::backend::RobotBackend>,
    context: &Context,
    width: f32,
) {
    let _ = context.run_ui(
        RawInput {
            screen_rect: Some(Rect::from_min_size(
                Default::default(),
                Vec2::new(width, 360.0),
            )),
            ..Default::default()
        },
        |root| {
            let ctx = root.ctx().clone();
            CentralPanel::default().show(root, |ui| {
                panel.header_ui(
                    ui,
                    PanelUiContext {
                        backend,
                        egui_context: &ctx,
                    },
                );
                panel.ui(
                    ui,
                    PanelUiContext {
                        backend,
                        egui_context: &ctx,
                    },
                );
            });
        },
    );
}

#[test]
fn views_restore_render_without_data_and_follow_namespace_changes() {
    let (_runtime, backend) = backend(None, "/");
    let context = Context::default();
    for view in View::ALL {
        let saved = serde_json::to_value(Settings {
            view,
            ..Default::default()
        })
        .unwrap();
        let mut panel = SystemPanel::new(PanelCreationContext {
            backend: backend.clone(),
            value: Some(&saved),
            egui_context: context.clone(),
        });
        assert_eq!(panel.save(), saved);
        render(&mut panel, &backend, &context, 320.0);
        backend.set_namespace("/other".into()).unwrap();
        render(&mut panel, &backend, &context, 1280.0);
        assert_eq!(panel.namespace, "/other");
        backend.set_namespace("/".into()).unwrap();
    }
}

// Explicit opt-in, read-only smoke test against the robot. No robot is needed by the normal suite.
#[test]
#[ignore = "requires robot 42 at tcp/10.1.24.42:7447"]
fn live_robot_system_topics_decode_and_render() {
    let (runtime, backend) = backend(Some("tcp/10.1.24.42:7447".into()), "/42");
    let context = Context::default();
    let mut panels: Vec<_> = View::ALL
        .into_iter()
        .map(|view| {
            let saved = serde_json::to_value(Settings {
                view,
                ..Default::default()
            })
            .unwrap();
            SystemPanel::new(PanelCreationContext {
                backend: backend.clone(),
                value: Some(&saved),
                egui_context: context.clone(),
            })
        })
        .collect();
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                let ready = panels
                    .iter()
                    .all(|panel| match panel.source.as_ref().unwrap() {
                        Source::Stats(source) => source.observation.latest().is_some(),
                        Source::Network(source) => source.observation.latest().is_some(),
                        Source::Disks(source) => source.observation.latest().is_some(),
                    });
                if ready {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .expect("all three system topics must provide typed data");
    });
    for panel in &mut panels {
        render(panel, &backend, &context, 500.0);
        render(panel, &backend, &context, 1200.0);
        match panel.source.as_ref().unwrap() {
            Source::Stats(source) => {
                let record = source.observation.latest().unwrap();
                assert!(!record.value.cpus.is_empty());
                assert!(record.value.ram.is_some());
                println!(
                    "tegrastats: {} cores, GPU {:?}%",
                    record.value.cpus.len(),
                    record.value.gpu_utilization_percent
                );
            }
            Source::Network(source) => println!(
                "network: {:?}",
                source
                    .observation
                    .latest()
                    .unwrap()
                    .value
                    .keys()
                    .collect::<Vec<_>>()
            ),
            Source::Disks(source) => println!(
                "disks: {:?}",
                source
                    .observation
                    .latest()
                    .unwrap()
                    .value
                    .keys()
                    .collect::<Vec<_>>()
            ),
        }
    }
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
    // Individual views also allow us to check their decoded values and invalidation.
    let mut panels: Vec<_> = View::ALL
        .into_iter()
        .map(|view| {
            let saved = serde_json::to_value(Settings {
                view,
                ..Default::default()
            })
            .unwrap();
            SystemPanel::new(PanelCreationContext {
                backend: backend.clone(),
                value: Some(&saved),
                egui_context: context.clone(),
            })
        })
        .collect();
    let sample = Tegrastats {
        cpus: vec![
            CpuInfo {
                utilization_percent: Some(80),
                frequency_mhz: Some(1900),
            },
            CpuInfo::default(),
        ],
        gpu_utilization_percent: Some(45),
        ram: Some(MemoryUsage {
            used_mib: 4096,
            total_mib: 8192,
        }),
        swap: None,
        temperatures: vec![Temperature {
            name: "cpu".into(),
            celsius: 65.5,
        }],
        power: vec![Power {
            name: "VDD_IN".into(),
            current_milliwatts: 12500,
            average_milliwatts: 12000,
        }],
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
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            stats.publish(&sample).await.unwrap();
            network.publish(&network_sample).await.unwrap();
            disks.publish(&disk_sample).await.unwrap();
            let ready = panels
                .iter()
                .all(|panel| match panel.source.as_ref().unwrap() {
                    Source::Stats(source) => source.observation.get_all().len() >= 3,
                    Source::Network(source) => source.observation.get_all().len() >= 3,
                    Source::Disks(source) => source.observation.get_all().len() >= 3,
                });
            if ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    for width in [500.0, 1280.0] {
        for _ in 0..3 {
            let _ = context.run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        Default::default(),
                        Vec2::new(width, 800.0),
                    )),
                    ..Default::default()
                },
                |ui| {
                    CentralPanel::default().show(ui, |ui| layout.ui(ui, &backend));
                },
            );
        }
        for panel in &mut panels {
            render(panel, &backend, &context, width);
        }
    }
    for panel in &panels {
        match panel.source.as_ref().unwrap() {
            Source::Stats(source) => assert_eq!(source.observation.latest().unwrap().value, sample),
            Source::Network(source) => {
                assert_eq!(source.observation.latest().unwrap().value, network_sample)
            }
            Source::Disks(source) => {
                assert_eq!(source.observation.latest().unwrap().value, disk_sample)
            }
        }
    }
    backend.set_namespace("/empty_robot".into()).unwrap();
    for panel in &mut panels {
        render(panel, &backend, &context, 500.0);
        let empty = match panel.source.as_ref().unwrap() {
            Source::Stats(source) => source.observation.latest().is_none(),
            Source::Network(source) => source.observation.latest().is_none(),
            Source::Disks(source) => source.observation.latest().is_none(),
        };
        assert!(empty, "previous robot's metrics must disappear immediately");
    }
    router.shutdown().unwrap();
}
