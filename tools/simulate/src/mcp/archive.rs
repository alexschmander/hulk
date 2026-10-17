//! Indexed selection over immutable recorder segments, shared by reads and exports.
use super::{
    rosz::{RecordingDecoder, Target},
    server::{id, string},
};
use crate::telemetry::recording::{Checkpoint, Receipt};
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::message::{SerdeCdrCodec, WireDecoder};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};
use tokio_util::sync::CancellationToken;

pub(super) async fn checkpoint(target: &Target) -> Result<Checkpoint> {
    if target
        .worker
        .stopped
        .load(std::sync::atomic::Ordering::Acquire)
    {
        Ok(serde_json::from_slice(&std::fs::read(
            target.directory.join("recording-segments.json"),
        )?)?)
    } else {
        Ok(serde_json::from_value(
            target
                .worker
                .call("checkpoint", json!({"target_id":target.id}))
                .await?,
        )?)
    }
}
pub(super) fn metadata(target: &Target, checkpoint: &Checkpoint) -> Value {
    let tail = if target
        .worker
        .stopped
        .load(std::sync::atomic::Ordering::Acquire)
        && !checkpoint.finalized
    {
        "partial"
    } else {
        "complete_to_barrier"
    };
    json!({"checkpoint_id":id(),"recorders":[{"target_id":target.id,
        "simulation_receipt_until_ns":(checkpoint.simulation_time-target.worker.origin).to_string(),
        "wall_receipt_until_ns":checkpoint.wall_time.to_string(),"ingestion_sequence":checkpoint.ingestion_sequence.to_string(),
        "known_dropped_samples":checkpoint.dropped_samples,"tail_status":tail}]})
}

pub(super) struct Selection {
    pub sources: Vec<Value>,
    namespace: String,
    origin: i64,
    start: i64,
    end: i64,
    clock: String,
}
impl Selection {
    pub fn new(namespace: &str, origin: i64, sources: &[Value], range: &Value) -> Result<Self> {
        let start = string(range, "start_ns")?.parse()?;
        let end = string(range, "end_ns")?.parse()?;
        ensure!(start < end, "History range must be increasing");
        Ok(Self {
            sources: sources.to_vec(),
            namespace: format!("{namespace}/"),
            origin,
            start,
            end,
            clock: string(range, "clock")?.into(),
        })
    }
    fn matches(&self, source: &Value, topic: &str) -> bool {
        source["topic"]
            .as_str()
            .is_some_and(|name| name.strip_prefix(&self.namespace).unwrap_or(name) == topic)
    }
    fn includes(&self, message: &mcap::Message<'_>, receipt: &Receipt) -> bool {
        let time = match self.clock.as_str() {
            "source" => message.publish_time as i128,
            "wall_receipt" => receipt.wall_time as i128,
            _ => message.log_time as i128 - self.origin as i128,
        };
        time >= self.start as i128 && time < self.end as i128
    }
    fn chunk_in_range(&self, chunk: &mcap::records::ChunkIndex) -> bool {
        self.clock != "simulation_receipt"
            || (chunk.message_end_time as i128 >= self.start as i128 + self.origin as i128
                && (chunk.message_start_time as i128) < self.end as i128 + self.origin as i128)
    }
}

#[derive(Default)]
pub(super) struct ScanReport {
    recorded: BTreeMap<String, u64>,
    selected: BTreeMap<String, u64>,
    pub segments: usize,
    pub segments_read: usize,
    pub chunks: usize,
    pub chunks_read: usize,
    pub linear_segments: usize,
}
impl ScanReport {
    pub fn json(&self, sources: &[Value]) -> Value {
        json!({"segments":self.segments,"segments_read":self.segments_read,"chunks":self.chunks,
            "chunks_read":self.chunks_read,"linear_segments":self.linear_segments,
            "sources":sources.iter().map(|source| {
                let topic=source["topic"].as_str().unwrap();
                let count=self.recorded.get(topic).copied().unwrap_or(0);
                json!({"topic":topic,"recorded_samples":count,"status":if count==0 {"topic_never_recorded"} else if self.selected.get(topic).copied().unwrap_or(0)==0 {"no_samples_in_range"}else{"recorded"}})
            }).collect::<Vec<_>>()})
    }
}

// Keep the adjacent publication/receipt pair together even across a chunk boundary.
fn consume<'a>(
    message: mcap::Message<'a>,
    pending: &mut Option<mcap::Message<'a>>,
    selection: &Selection,
    report: &mut ScanReport,
    linear: bool,
    emit: &mut impl FnMut(&Value, &mcap::Message<'_>, &Receipt) -> Result<()>,
) -> Result<()> {
    if message.channel.topic == "_recording/receipt" {
        if let Some(data) = pending.take() {
            let receipt: Receipt = SerdeCdrCodec::deserialize(&message.data)?;
            ensure!(
                receipt.channel_id == data.channel.id
                    && message.log_time == data.log_time
                    && data.sequence == u32::try_from(receipt.sequence).unwrap_or(u32::MAX),
                "Recording receipt does not match publication"
            );
            if selection.includes(&data, &receipt) {
                for source in selection
                    .sources
                    .iter()
                    .filter(|source| selection.matches(source, &data.channel.topic))
                {
                    *report
                        .selected
                        .entry(source["topic"].as_str().unwrap().into())
                        .or_default() += 1;
                    emit(source, &data, &receipt)?;
                }
            }
        }
    } else {
        ensure!(
            pending.is_none(),
            "Recorded publication has no matching receipt"
        );
        if selection
            .sources
            .iter()
            .any(|source| selection.matches(source, &message.channel.topic))
        {
            if linear {
                let mut topics = BTreeSet::new();
                for source in &selection.sources {
                    if topics.insert(source["topic"].as_str().unwrap())
                        && selection.matches(source, &message.channel.topic)
                    {
                        *report
                            .recorded
                            .entry(source["topic"].as_str().unwrap().into())
                            .or_default() += 1;
                    }
                }
            }
            let time = match selection.clock.as_str() {
                "source" => Some(message.publish_time as i128),
                "simulation_receipt" => Some(message.log_time as i128 - selection.origin as i128),
                _ => None,
            };
            if time
                .is_none_or(|time| time >= selection.start as i128 && time < selection.end as i128)
            {
                *pending = Some(message);
            }
        }
    }
    Ok(())
}

pub(super) fn scan(
    directory: &Path,
    checkpoint: &Checkpoint,
    selection: &Selection,
    cancellation: &CancellationToken,
    mut emit: impl FnMut(&Value, &mcap::Message<'_>, &Receipt) -> Result<()>,
) -> Result<ScanReport> {
    let mut report = ScanReport::default();
    for path in &checkpoint.files {
        ensure!(!cancellation.is_cancelled(), "Archive query cancelled");
        report.segments += 1;
        let file = std::fs::File::open(directory.join(path))?;
        // Only immutable finalized files from the frozen manifest are mapped.
        let mapped = unsafe { memmap2::Mmap::map(&file)? };
        let summary = mcap::Summary::read(&mapped)?;
        if let Some(summary) = summary.filter(|s| {
            s.stats.is_some()
                && (!s.chunk_indexes.is_empty() || s.stats.as_ref().unwrap().message_count == 0)
        }) {
            let stats = summary.stats.as_ref().unwrap();
            let mut topics = BTreeSet::new();
            for source in &selection.sources {
                if !topics.insert(source["topic"].as_str().unwrap()) {
                    continue;
                }
                let count = summary
                    .channels
                    .iter()
                    .filter(|(_, channel)| selection.matches(source, &channel.topic))
                    .map(|(id, _)| stats.channel_message_counts.get(id).copied().unwrap_or(0))
                    .sum::<u64>();
                *report
                    .recorded
                    .entry(source["topic"].as_str().unwrap().into())
                    .or_default() += count;
            }
            report.chunks += summary.chunk_indexes.len();
            let mut pending = None;
            let mut read_segment = false;
            for chunk in &summary.chunk_indexes {
                ensure!(!cancellation.is_cancelled(), "Archive query cancelled");
                let contains_topic = chunk.message_index_offsets.is_empty()
                    || chunk.message_index_offsets.keys().any(|id| {
                        summary.channels.get(id).is_some_and(|channel| {
                            selection
                                .sources
                                .iter()
                                .any(|source| selection.matches(source, &channel.topic))
                        })
                    });
                if pending.is_none() && (!contains_topic || !selection.chunk_in_range(chunk)) {
                    continue;
                }
                report.chunks_read += 1;
                read_segment = true;
                for message in summary.stream_chunk(&mapped, chunk)? {
                    ensure!(!cancellation.is_cancelled(), "Archive query cancelled");
                    consume(
                        message?,
                        &mut pending,
                        selection,
                        &mut report,
                        false,
                        &mut emit,
                    )?;
                }
            }
            ensure!(
                pending.is_none(),
                "Recorded publication has no matching receipt"
            );
            report.segments_read += usize::from(read_segment);
        } else {
            report.linear_segments += 1;
            report.segments_read += 1;
            let mut pending = None;
            for message in mcap::MessageStream::new(&mapped)? {
                ensure!(!cancellation.is_cancelled(), "Archive query cancelled");
                consume(
                    message?,
                    &mut pending,
                    selection,
                    &mut report,
                    true,
                    &mut emit,
                )?;
            }
            ensure!(
                pending.is_none(),
                "Recorded publication has no matching receipt"
            );
        }
    }
    Ok(report)
}

pub(super) async fn read(
    target: &Arc<Target>,
    sources: &[Value],
    range: &Value,
    cancellation: Option<CancellationToken>,
) -> Result<(Vec<Value>, Value, Value)> {
    let checkpoint = checkpoint(target).await?;
    let target = target.clone();
    let selection = Selection::new(&target.namespace, target.worker.origin, sources, range)?;
    let range = range.clone();
    tokio::task::spawn_blocking(move || {
        let mut rows=Vec::new();
        let mut bytes=0;
        let mut decoder=RecordingDecoder::default();
        let report=scan(&target.directory,&checkpoint,&selection,&cancellation.unwrap_or_default(),|source,data,receipt| {
            if let Some(sample)=decoder.sample(&target,source,data,receipt,&range)? {
                bytes+=serde_json::to_vec(&sample)?.len();
                ensure!(rows.len()<100_000 && bytes<=64*1024*1024,"Recorded query exceeds snapshot budget; narrow its time range or use rosz_export");
                rows.push(sample);
            }
            Ok(())
        })?;
        Ok((rows,metadata(&target,&checkpoint),report.json(&selection.sources)))
    }).await.map_err(|e|eyre!("Archive reader: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_z::{Message, message::WireEncoder};
    use std::collections::BTreeMap;

    fn segment(path: &Path, start: u64, indexed: bool) {
        let mut writer = mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Lz4))
            .chunk_size(Some(100))
            .emit_summary_records(indexed)
            .create(std::fs::File::create(path).unwrap())
            .unwrap();
        let schema = writer.add_schema("test", "bytes", b"{}").unwrap();
        let wanted = writer
            .add_channel(schema, "wanted", "bytes", &BTreeMap::new())
            .unwrap();
        let other = writer
            .add_channel(schema, "other", "bytes", &BTreeMap::new())
            .unwrap();
        let receipt_schema = writer
            .add_schema(
                "receipt",
                "ros-z-schema-json",
                &serde_json::to_vec(&Receipt::schema()).unwrap(),
            )
            .unwrap();
        let receipt_channel = writer
            .add_channel(
                receipt_schema,
                "_recording/receipt",
                "ros-z-cdr",
                &BTreeMap::new(),
            )
            .unwrap();
        for i in 0..30u32 {
            let time = start + i as u64;
            let channel = if i % 2 == 0 { wanted } else { other };
            let receipt:Receipt=serde_json::from_value(json!({"channel_id":channel,
                "publisher":ros_z::attachment::EndpointGlobalId::from([1;ros_z::attachment::ENDPOINT_GLOBAL_ID_SIZE]),
                "sequence":i,"wall_time":5000+time,"transport_time":null})).unwrap();
            writer
                .write_to_known_channel(
                    &mcap::records::MessageHeader {
                        channel_id: channel,
                        sequence: i,
                        log_time: time,
                        publish_time: 10000 - time,
                    },
                    &[42; 110],
                )
                .unwrap();
            writer
                .write_to_known_channel(
                    &mcap::records::MessageHeader {
                        channel_id: receipt_channel,
                        sequence: i,
                        log_time: time,
                        publish_time: time,
                    },
                    &SerdeCdrCodec::serialize(&receipt).unwrap(),
                )
                .unwrap();
        }
        writer.finish().unwrap();
    }
    #[test]
    fn indexed_selection_skips_chunks_and_keeps_receipts_across_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        segment(&dir.path().join("a.mcap"), 100, true);
        segment(&dir.path().join("b.mcap"), 200, true);
        let checkpoint = Checkpoint {
            files: vec!["a.mcap".into(), "b.mcap".into()],
            ..Default::default()
        };
        let sources = vec![json!({"topic":"wanted"}), json!({"topic":"never"})];
        let selection = Selection::new(
            "/test",
            100,
            &sources,
            &json!({"clock":"simulation_receipt","start_ns":"4","end_ns":"10"}),
        )
        .unwrap();
        let mut found = Vec::new();
        let report = scan(
            dir.path(),
            &checkpoint,
            &selection,
            &CancellationToken::new(),
            |_, message, receipt| {
                found.push((message.log_time, receipt.sequence));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(found, vec![(104, 4), (106, 6), (108, 8)]);
        assert_eq!(report.segments_read, 1);
        assert!(report.chunks_read < report.chunks / 2);
        let json = report.json(&sources);
        assert_eq!(json["sources"][0]["recorded_samples"], 30);
        assert_eq!(json["sources"][1]["status"], "topic_never_recorded");
    }
    #[test]
    fn source_and_wall_ranges_do_not_use_simulation_time_indexes() {
        let dir = tempfile::tempdir().unwrap();
        for indexed in [false, true] {
            segment(&dir.path().join("a.mcap"), 100, indexed);
            let checkpoint = Checkpoint {
                files: vec!["a.mcap".into()],
                ..Default::default()
            };
            for (clock, start, end) in [("source", 9891, 9897), ("wall_receipt", 5104, 5110)] {
                let selection = Selection::new(
                    "/test",
                    100,
                    &[json!({"topic":"wanted"})],
                    &json!({"clock":clock,"start_ns":start.to_string(),"end_ns":end.to_string()}),
                )
                .unwrap();
                let mut found = Vec::new();
                let report = scan(
                    dir.path(),
                    &checkpoint,
                    &selection,
                    &CancellationToken::new(),
                    |_, message, _| {
                        found.push(message.log_time);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(found, vec![104, 106, 108]);
                assert_eq!(report.linear_segments, usize::from(!indexed));
            }
            let cancelled = CancellationToken::new();
            cancelled.cancel();
            let selection = Selection::new(
                "/test",
                0,
                &[json!({"topic":"wanted"})],
                &json!({"clock":"source","start_ns":"0","end_ns":"1"}),
            )
            .unwrap();
            assert!(
                scan(
                    dir.path(),
                    &checkpoint,
                    &selection,
                    &cancelled,
                    |_, _, _| Ok(())
                )
                .is_err()
            );
        }
    }
}
