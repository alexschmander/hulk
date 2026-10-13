use super::*;
use std::{fs::File, time::Duration};

fn record(sequence: u64) -> LogRecord {
    LogRecord {
        sequence,
        simulation_time: 42,
        wall_time: sequence as i64,
        level: 3,
        target: "motion".into(),
        message: format!("entry {sequence}"),
        fields: "{}".into(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn complete_history_pages_follow_appends_and_reset_on_new_sessions() {
    let context = ros_z::context::ContextBuilder::default()
        .with_clock(ros_z::time::Clock::logical(ros_z::time::Time::from_nanos(
            42,
        )))
        .disable_multicast_scouting()
        .with_listen_endpoints(std::iter::empty::<&str>())
        .with_connect_endpoints(std::iter::empty::<&str>())
        .build()
        .await
        .unwrap();
    let node = context.create_node("archive_test").build().await.unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("logs.jsonl");
    let file = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let archive = Archive::start(&node, file, "logs/history").await.unwrap();
    for sequence in 0..6200 {
        archive.append(record(sequence)).await.unwrap();
    }
    let client = node
        .service_client::<ReadLogs>("logs/history")
        .build()
        .await
        .unwrap();
    let mut request = ReadLogsRequest::default();
    let mut records = Vec::new();
    loop {
        let page = client
            .call_with_timeout_async(&request, Duration::from_secs(3))
            .await
            .unwrap();
        assert!(page.error.is_empty(), "{}", page.error);
        assert!(page.records.len() <= 512);
        assert!(page.next_offset > request.offset);
        request = ReadLogsRequest {
            session: page.session,
            offset: page.next_offset,
        };
        records.extend(page.records);
        if request.offset == page.end {
            break;
        }
    }
    assert_eq!(records, (0..6200).map(record).collect::<Vec<_>>());
    let empty = client.call_async(&request).await.unwrap();
    assert!(empty.records.is_empty());
    assert_eq!(empty.next_offset, request.offset);

    // An invalid cursor cannot corrupt the file position used by subsequent appends.
    let invalid = client
        .call_async(&ReadLogsRequest {
            session: request.session.clone(),
            offset: 1,
        })
        .await
        .unwrap();
    assert!(!invalid.error.is_empty());
    archive.append(record(6200)).await.unwrap();
    let tail = client.call_async(&request).await.unwrap();
    assert_eq!(tail.records, [record(6200)]);
    archive.finish().await.unwrap();
    let saved: Vec<LogRecord> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(saved, (0..6201).map(record).collect::<Vec<_>>());

    let next = Archive::start(&node, tempfile::tempfile().unwrap(), "logs/history")
        .await
        .unwrap();
    next.append(record(99)).await.unwrap();
    let restarted = client.call_async(&request).await.unwrap();
    assert!(restarted.error.is_empty());
    assert_ne!(restarted.session, request.session);
    assert_eq!(restarted.records, [record(99)]);
    next.finish().await.unwrap();
}
