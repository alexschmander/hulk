//! Queries only finalized recorder segments, never the mutable MCAP tail.
use super::{
    rosz::{Target, recorded_sample},
    server::id,
};
use crate::telemetry::recording::Checkpoint;
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::message::{SerdeCdrCodec, WireDecoder};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub(super) async fn read(
    target: &Arc<Target>,
    sources: &[Value],
    range: &Value,
    cancellation: Option<CancellationToken>,
) -> Result<(Vec<Value>, Value)> {
    let checkpoint: Checkpoint = if target
        .worker
        .stopped
        .load(std::sync::atomic::Ordering::Acquire)
    {
        serde_json::from_slice(&std::fs::read(
            target.directory.join("recording-segments.json"),
        )?)?
    } else {
        serde_json::from_value(
            target
                .worker
                .call("checkpoint", json!({"target_id":target.id}))
                .await?,
        )?
    };
    let target = target.clone();
    let sources = sources.to_vec();
    let range = range.clone();
    tokio::task::spawn_blocking(move || {
        let prefix = format!("{}/", target.namespace);
        let matches = |source: &Value, topic: &str| {
            source["topic"].as_str().is_some_and(|name| name.strip_prefix(&prefix).unwrap_or(name) == topic)
        };
        let mut rows = Vec::new();
        let mut bytes = 0;
        for path in &checkpoint.files {
            let file = std::fs::File::open(target.directory.join(path))?;
            // The manifest contains immutable, finalized segments only.
            let mapped = unsafe { memmap2::Mmap::map(&file)? };
            let mut pending: Option<mcap::Message<'_>> = None;
            for message in mcap::MessageStream::new(&mapped)? {
                ensure!(!cancellation.as_ref().is_some_and(CancellationToken::is_cancelled), "Archive query cancelled");
                let message = message?;
                if message.channel.topic == "_recording/receipt" {
                    let receipt: crate::telemetry::recording::Receipt = SerdeCdrCodec::deserialize(&message.data)?;
                    if let Some(data) = pending.take() {
                        ensure!(
                            receipt.channel_id == data.channel.id
                                && message.log_time == data.log_time
                                && data.sequence == u32::try_from(receipt.sequence).unwrap_or(u32::MAX),
                            "Recording receipt does not match publication"
                        );
                        for source in sources.iter().filter(|source| matches(source, &data.channel.topic)) {
                            if let Some(sample) = recorded_sample(&target, source, &data, &receipt, &range)? {
                                bytes += serde_json::to_vec(&sample)?.len();
                                rows.push(sample);
                            }
                        }
                    }
                } else if sources.iter().any(|source| matches(source, &message.channel.topic)) {
                    ensure!(pending.is_none(), "Recorded publication has no matching receipt");
                    pending = Some(message);
                } else {
                    ensure!(pending.is_none(), "Recorded publication has no matching receipt");
                }
                ensure!(rows.len() <= 100_000 && bytes <= 64 * 1024 * 1024,
                    "Recorded query exceeds snapshot budget; narrow its time range");
            }
            ensure!(pending.is_none(), "Recorded publication has no matching receipt");
        }
        let tail_status = if target.worker.stopped.load(std::sync::atomic::Ordering::Acquire) && !checkpoint.finalized {
            "partial"
        } else {
            "complete_to_barrier"
        };
        let metadata = json!({
            "checkpoint_id": id(),
            "recorders": [{
                "target_id": target.id,
                "simulation_receipt_until_ns": (checkpoint.simulation_time - target.worker.origin).to_string(),
                "wall_receipt_until_ns": checkpoint.wall_time.to_string(),
                "ingestion_sequence": checkpoint.ingestion_sequence.to_string(),
                "known_dropped_samples": checkpoint.dropped_samples,
                "tail_status": tail_status,
            }],
        });
        Ok((rows, metadata))
    }).await.map_err(|error|eyre!("Archive reader: {error}"))?
}
