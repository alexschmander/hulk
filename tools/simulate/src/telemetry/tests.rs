// Exercise the same schema validation, indexes and seek implementation used by mcap-replay.
#[allow(dead_code)]
#[path = "../../../mcap-replay/src/recording.rs"]
mod replay_recording;

use super::*;
use ros_z::time::Time;
use std::time::Duration;

#[derive(Clone, serde::Serialize, serde::Deserialize, ros_z::Message)]
#[message(name = "tests::LateRecordingSample")]
struct LateRecordingSample {
    value: u64,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disabling_mcap_keeps_logs_without_subscribing_or_creating_recordings() {
    use tracing::instrument::WithSubscriber;

    let directory = tempfile::tempdir().unwrap();
    let overrides = directory.path().join("overrides");
    std::fs::create_dir(&overrides).unwrap();
    std::fs::write(
        overrides.join("simulator_recording.json5"),
        "{mcap_enabled: false}",
    )
    .unwrap();
    let path = directory.path().join("robot");
    let clock = Clock::logical(Time::from_nanos(1_000_000_000));
    let startup = Startup::new(path.clone(), clock.clone()).unwrap();
    let dispatch =
        tracing::Dispatch::new(tracing_subscriber::registry().with(startup.capture.clone()));
    async {
        let layers = vec![
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parameters"),
            overrides,
        ];
        let context = Arc::new(
            ContextBuilder::default()
                .with_namespace("/recording_disabled/1")
                .with_clock(clock)
                .with_parameter_layers(layers.clone())
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .build()
                .await
                .unwrap(),
        );
        let telemetry = Telemetry::start(startup, &context, "/recording_disabled/1", &layers)
            .await
            .unwrap();
        assert!(!telemetry.mcap_enabled());
        let node = context.create_node("test").build().await.unwrap();
        let publisher = node
            .publisher::<u64>("ground_truth/unobserved")
            .build()
            .await
            .unwrap();
        assert!(!publisher.has_subscribers());
        publisher.publish(&42).await.unwrap();
        tracing::info!("Robot logs remain available without MCAP");
        assert!(telemetry.status().is_none());
        assert!(
            telemetry
                .recording_checkpoint()
                .await
                .unwrap_err()
                .to_string()
                .contains("disabled")
        );
        assert!(telemetry.finish().await.is_none());
        context.shutdown().unwrap();
    }
    .with_subscriber(dispatch)
    .await;
    assert!(
        std::fs::read_to_string(path.join("logs.jsonl"))
            .unwrap()
            .contains("Robot logs remain available without MCAP")
    );
    assert!(!path.join("recording.mcap").exists());
    assert!(!path.join("recording-segments.json").exists());
    let finished: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("finished.json")).unwrap()).unwrap();
    assert_eq!(finished["mcap_enabled"], false);
    assert_eq!(finished["dropped_samples"], 0);
    assert!(finished["error"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn isolated_robots_capture_nested_tasks_and_record_startup_late_topics_and_pause() {
    // Install the log bridge as Twix does. A local dispatcher still owns each robot's records.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let directory = tempfile::tempdir().unwrap();
    let clock = Clock::logical(Time::from_nanos(1_000_000_000));
    let mut robots = Vec::new();
    for namespace in ["/test_a/1", "/test_b/1"] {
        let path = directory.path().join(namespace.trim_matches('/'));
        let startup = Startup::new(path.clone(), clock.clone()).unwrap();
        let runtime = RobotRuntime::new(startup.capture.clone()).unwrap();
        let logical_clock = clock.clone();
        let (context, telemetry, publisher, producers) = runtime
            .handle()
            .spawn(async move {
                let layers = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parameters")];
                let context = Arc::new(
                    ContextBuilder::default()
                        .with_namespace(namespace)
                        .with_clock(logical_clock)
                        .with_parameter_layers(layers.clone())
                        .disable_multicast_scouting()
                        .with_listen_endpoints(["tcp/127.0.0.1:0"])
                        .with_connect_endpoints(std::iter::empty::<&str>())
                        .build()
                        .await
                        .unwrap(),
                );
                let node = Arc::new(context.create_node("test").build().await.unwrap());
                let telemetry = Telemetry::start(startup, &context, namespace, &layers)
                    .await
                    .unwrap();
                tracing::info!("{namespace} startup");
                tokio::spawn(async move {
                    tracing::warn!("{namespace} nested task");
                })
                .await
                .unwrap();
                tokio::task::spawn_blocking(move || {
                    tracing::error!("{namespace} blocking worker");
                    log::warn!("{namespace} log bridge");
                })
                .await
                .unwrap();
                let publisher = node
                    .publisher::<u64>("ground_truth/test")
                    .build()
                    .await
                    .unwrap();
                publisher.publish(&42).await.unwrap();
                tokio::time::sleep(Duration::from_millis(50)).await;
                let late_node = context.create_node("short_lived").build().await.unwrap();
                let late = late_node
                    .publisher::<LateRecordingSample>("late")
                    .build()
                    .await
                    .unwrap();
                late.publish(&LateRecordingSample { value: 7 })
                    .await
                    .unwrap();
                let optional = late_node
                    .publisher::<Option<u32>>("optional")
                    .build()
                    .await
                    .unwrap();
                optional.publish(&Some(17)).await.unwrap();
                drop((late, optional, late_node));
                // The same paused robot clock must support reading buffered startup logs.
                let history = node.service_client::<robot_logs::ReadLogs>("logs/history")
                    .build().await.unwrap();
                tokio::time::timeout(Duration::from_secs(3), async {
                    let mut request = robot_logs::ReadLogsRequest::default();
                    let mut records = Vec::new();
                    loop {
                        let page = history.call_async(&request).await.unwrap();
                        assert!(page.error.is_empty(), "{}", page.error);
                        request.session = page.session;
                        request.offset = page.next_offset;
                        records.extend(page.records);
                        if ["startup", "nested task", "blocking worker", "log bridge"].iter().all(|text| {
                            records.iter().any(|record| record.message.contains(&format!("{namespace} {text}")))
                        }) { break; }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }).await.expect("startup logs should be available through ROS-Z without advancing simulation time");
                (context, telemetry, publisher, node)
            })
            .await
            .unwrap();
        robots.push((
            runtime, context, telemetry, publisher, path, namespace, producers,
        ));
    }
    tokio::time::sleep(Duration::from_millis(80)).await;
    for (_, _, _, publisher, _, _, _) in &robots {
        publisher
            .publish_with_source_time(&43, Time::from_nanos(99_000_000_000))
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(30)).await;
    clock.advance(Duration::from_secs(4)).unwrap();
    for (_, _, _, publisher, _, _, _) in &robots {
        publisher.publish(&44).await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(30)).await;
    for (runtime, context, telemetry, publisher, path, namespace, producers) in robots {
        let error = runtime.handle().spawn(telemetry.finish()).await.unwrap();
        assert!(error.is_none(), "{error:?}");
        let logs = std::fs::read_to_string(path.join("logs.jsonl")).unwrap();
        for text in ["startup", "nested task", "blocking worker", "log bridge"] {
            assert!(
                logs.contains(&format!("{namespace} {text}")),
                "missing {text}: {logs}"
            );
        }
        let other = if namespace == "/test_a/1" {
            "/test_b/1"
        } else {
            "/test_a/1"
        };
        assert!(!logs.contains(other), "robot logs mixed");
        let mut replay =
            replay_recording::Recording::open(&path.join("recording.mcap"), "/replay").unwrap();
        assert!(
            replay
                .channels
                .values()
                .any(|channel| channel.topic == "/replay/logs")
        );
        let snapshot = replay.snapshot(1_000_000_000).unwrap();
        let paused_input = snapshot
            .iter()
            .find(|message| {
                replay.channels[&message.header.channel_id]
                    .topic
                    .ends_with("/ground_truth/test")
            })
            .unwrap();
        assert_eq!(paused_input.header.publish_time, 99_000_000_000);
        assert_eq!(paused_input.replay_time, 1_000_000_000);
        assert!(!replay.snapshot(replay.end).unwrap().is_empty());
        let bytes = std::fs::read(path.join("recording.mcap")).unwrap();
        assert!(
            mcap::Summary::read(&bytes).unwrap().is_some(),
            "MCAP footer missing"
        );
        let messages: Vec<_> = mcap::MessageStream::new(&bytes)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert!(messages.iter().any(|m| m.channel.topic == "logs"));
        assert!(messages.iter().any(|m| m.channel.topic == "late"));
        let optional = messages
            .iter()
            .find(|m| m.channel.topic == "optional")
            .unwrap();
        assert_eq!(
            ros_z::message::SerdeCdrCodec::<Option<u32>>::deserialize(&optional.data).unwrap(),
            Some(17)
        );
        let times: Vec<_> = messages
            .iter()
            .filter(|m| m.channel.topic == "ground_truth/test")
            .map(|m| m.log_time)
            .collect();
        assert_eq!(times, [1_000_000_000, 1_000_000_000, 5_000_000_000]);
        assert!(
            messages
                .iter()
                .any(|m| m.channel.topic == "ground_truth/test"
                    && m.log_time == 1_000_000_000
                    && m.publish_time == 99_000_000_000)
        );
        use ros_z::message::WireDecoder;
        let receipts: Vec<_> = messages
            .iter()
            .filter(|message| message.channel.topic == "_recording/receipt")
            .map(|message| {
                ros_z::message::SerdeCdrCodec::<super::recording::Receipt>::deserialize(
                    &message.data,
                )
                .unwrap()
            })
            .collect();
        assert_eq!(receipts.len() * 2, messages.len());
        assert!(receipts.last().unwrap().wall_time > receipts[0].wall_time);
        drop((publisher, producers));
        context.shutdown().unwrap();
        drop(runtime);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn log_dispatch_follows_each_robots_nested_and_blocking_tasks() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let mut robots = Vec::new();
    for robot in ["hulks/1", "opponents/1"] {
        let (capture, receiver) = logs::Capture::new(Clock::logical(Time::from_nanos(123)));
        let runtime = RobotRuntime::new(capture).unwrap();
        runtime
            .handle()
            .spawn(async move {
                tracing::info!("{robot} root");
                tokio::spawn(async move {
                    tracing::warn!("{robot} child");
                })
                .await
                .unwrap();
                tokio::task::spawn_blocking(move || {
                    tracing::error!("{robot} blocking");
                    log::warn!("{robot} legacy");
                })
                .await
                .unwrap();
            })
            .await
            .unwrap();
        robots.push((runtime, receiver, robot));
    }
    for (_runtime, mut receiver, robot) in robots {
        let mut records = Vec::new();
        while let Ok(record) = receiver.try_recv() {
            records.push(record);
        }
        for suffix in ["root", "child", "blocking", "legacy"] {
            assert!(
                records
                    .iter()
                    .any(|record| record.message == format!("{robot} {suffix}")),
                "missing {robot} {suffix}: {records:?}"
            );
        }
        assert!(
            records
                .iter()
                .all(|record| record.message.starts_with(robot))
        );
        assert!(records.iter().all(|record| record.simulation_time == 123));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn checkpoint_is_readable_while_recording_continues_and_replay_spans_segments() {
    use ros_z::message::WireDecoder;
    let directory = tempfile::tempdir().unwrap();
    let clock = Clock::logical(Time::from_nanos(1_000_000_000));
    let context = Arc::new(
        ContextBuilder::default()
            .with_namespace("/checkpoint/1")
            .with_parameter_layers(vec![
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parameters"),
            ])
            .with_clock(clock.clone())
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap(),
    );
    let node = Arc::new(context.create_node("test").build().await.unwrap());
    let layers = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parameters")];
    let telemetry = Telemetry::start(
        Startup::new(directory.path().to_owned(), clock.clone()).unwrap(),
        &context,
        "/checkpoint/1",
        &layers,
    )
    .await
    .unwrap();
    let publisher = node
        .publisher::<u64>("checkpoint_value")
        .build()
        .await
        .unwrap();
    publisher.publish(&1).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let checkpoint = telemetry.recording_checkpoint().await.unwrap();
    assert_eq!(checkpoint.simulation_time, 1_000_000_000);
    assert!(!checkpoint.finalized);
    assert_eq!(checkpoint.dropped_samples, 0);
    let first = std::fs::read(directory.path().join(&checkpoint.files[0])).unwrap();
    assert!(mcap::Summary::read(&first).unwrap().is_some());
    clock.advance(Duration::from_secs(1)).unwrap();
    publisher.publish(&2).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let second = telemetry.recording_checkpoint().await.unwrap();
    assert!(second.ingestion_sequence > checkpoint.ingestion_sequence);
    assert_eq!(
        std::fs::read(directory.path().join(&checkpoint.files[0])).unwrap(),
        first
    );
    assert!(telemetry.finish().await.is_none());
    let mut replay = replay_recording::Recording::open(directory.path(), "/replay").unwrap();
    let topic = replay
        .channels
        .iter()
        .find(|(_, c)| c.topic == "/replay/checkpoint_value")
        .unwrap()
        .0
        .to_owned();
    for (time, expected) in [(1_000_000_000, 1u64), (2_000_000_000, 2)] {
        let snapshot = replay.snapshot(time).unwrap();
        let sample = snapshot
            .iter()
            .find(|m| m.header.channel_id == topic)
            .unwrap();
        assert_eq!(
            ros_z::message::SerdeCdrCodec::<u64>::deserialize(&sample.data.to_bytes()).unwrap(),
            expected
        );
    }
}
