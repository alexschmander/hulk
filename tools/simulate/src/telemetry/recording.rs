//! Capture before publisher startup and retain local schemas while queued samples drain.
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::dynamic::schema_service::LocalSchemaCapture;
use ros_z::message::{SerdeCdrCodec, WireEncoder};
use ros_z::{attachment::Attachment, prelude::*, time::Clock};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::mpsc;
use zenoh::sample::Sample;

const RECEIPT_TOPIC: &str = "_recording/receipt";
#[derive(Clone, Debug, Serialize, Deserialize, ros_z::Message)]
#[message(name = "simulator::RecordingReceipt")]
pub(super) struct Receipt {
    channel_id: u16,
    publisher: ros_z::attachment::EndpointGlobalId,
    sequence: i64,
    pub(super) wall_time: i64,
    transport_time: Option<i64>,
}

struct SampleEnvelope {
    sample: Sample,
    simulation_time: u64,
    wall_time: i64,
}
struct Channel {
    topic: String,
    name: String,
    schema: Vec<u8>,
    metadata: BTreeMap<String, String>,
}
struct WriteSample {
    channel: Arc<Channel>,
    envelope: SampleEnvelope,
}
pub struct Recording {
    stop: tokio_util::sync::CancellationToken,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    pub dropped: Arc<AtomicU64>,
}
impl Recording {
    pub async fn start(
        context: &Arc<Context>,
        node: Arc<Node>,
        namespace: &str,
        directory: &std::path::Path,
        parameters: &super::Parameters,
        health: super::Health,
    ) -> Result<Self> {
        let local_schemas = LocalSchemaCapture::default();
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join("recording.mcap"))?;
        let (sender, mut receiver) = mpsc::channel(parameters.queue_capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let drops = dropped.clone();
        let clock = context.clock().clone();
        let prefix = format!("rt/{}/", namespace.trim_matches('/'));
        let patterns = parameters
            .topics
            .iter()
            .map(|p| zenoh::key_expr::OwnedKeyExpr::try_from(p.clone()).map_err(|e| eyre!("{e}")))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !patterns.is_empty(),
            "Recording topic patterns must not be empty"
        );
        let topic_prefix = prefix.clone();
        let subscriber = context
            .session()
            .declare_subscriber(format!("{prefix}**"))
            .callback(move |sample| {
                // ROS-Z data keys end in type name and hash; SDK traffic is outside this namespace.
                let Some(topic) = sample
                    .key_expr()
                    .as_str()
                    .strip_prefix(&topic_prefix)
                    .and_then(|key| key.rsplit_once('/'))
                    .and_then(|(key, _)| key.rsplit_once('/'))
                    .map(|(topic, _)| topic)
                else {
                    return;
                };
                let Ok(topic) = zenoh::key_expr::KeyExpr::try_from(topic) else {
                    return;
                };
                if !patterns.iter().any(|pattern| pattern.includes(&topic)) {
                    return;
                }
                let envelope = SampleEnvelope {
                    sample,
                    simulation_time: clock.now().as_nanos() as u64,
                    wall_time: Clock::wallclock().now().as_nanos(),
                };
                if sender.try_send(envelope).is_err() {
                    drops.fetch_add(1, Ordering::Relaxed);
                }
            })
            .await
            .map_err(|e| eyre!("Cannot capture robot topics: {e}"))?;
        let stop = tokio_util::sync::CancellationToken::new();
        let cancelled = stop.clone();
        let capacity = parameters.queue_capacity;
        let lost = dropped.clone();
        use tracing::instrument::WithSubscriber;
        let work = async move {
            let (writer_tx, writer_rx) = mpsc::channel(capacity);
            let writer = tokio::task::spawn_blocking(move || write(file, writer_rx));
            let mut channels = BTreeMap::<String, Arc<Channel>>::new();
            let mut schemas = std::collections::HashMap::new();
            let mut subscriber = Some(subscriber);
            let mut failed = std::collections::HashSet::new();
            loop {
                let next = tokio::select! {
                    biased;
                    _ = cancelled.cancelled(), if subscriber.is_some() => {
                        if let Some(subscriber) = subscriber.take() { subscriber.undeclare().await.map_err(|e| eyre!("{e}"))?; }
                        continue;
                    }
                    next = receiver.recv() => next,
                };
                let Some(envelope) = next else {
                    break;
                };
                let key = envelope.sample.key_expr().to_string();
                if failed.contains(&key) {
                    lost.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let channel = if let Some(channel) = channels.get(&key) {
                    channel.clone()
                } else {
                    let channel =
                        match resolve_channel(&node, &key, &prefix, &local_schemas, &mut schemas)
                            .await
                        {
                            Ok(channel) => Arc::new(channel),
                            Err(error) => {
                                health.error(format!("Cannot record {key}: {error:#}"));
                                failed.insert(key);
                                lost.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                        };
                    channels.insert(key, channel.clone());
                    channel
                };
                if writer_tx
                    .send(WriteSample { channel, envelope })
                    .await
                    .is_err()
                {
                    health.error("MCAP writer stopped");
                    break;
                }
            }
            drop(subscriber);
            drop(writer_tx);
            writer.await??;
            Ok(())
        };
        let task = tokio::spawn(work.with_subscriber(tracing::Dispatch::none()));
        Ok(Self {
            stop,
            task: Some(task),
            dropped,
        })
    }
    pub async fn finish(mut self) -> Result<()> {
        self.stop.cancel();
        self.task.take().unwrap().await?
    }
    pub fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(|task| task.is_finished())
    }
}
async fn resolve_channel(
    node: &Node,
    key: &str,
    prefix: &str,
    local_schemas: &LocalSchemaCapture,
    schemas: &mut std::collections::HashMap<
        (String, ros_z::entity::SchemaHash),
        ros_z::dynamic::Schema,
    >,
) -> Result<Channel> {
    let (without_hash, hash) = key
        .rsplit_once('/')
        .ok_or_else(|| eyre!("Missing schema hash"))?;
    let (qualified_topic, name) = without_hash
        .rsplit_once('/')
        .ok_or_else(|| eyre!("Missing type name"))?;
    let topic = qualified_topic
        .strip_prefix(prefix)
        .ok_or_else(|| eyre!("Topic outside robot namespace"))?;
    let hash = ros_z::entity::SchemaHash::from_hash_string(hash).map_err(|e| eyre!("{e}"))?;
    let schema = if let Some(schema) = schemas.get(&(name.to_owned(), hash)) {
        schema.clone()
    } else {
        let schema = match local_schemas.get(name, &hash)? {
            Some(registered) => registered.schema,
            None => {
                let discovered = node
                    .discover_topic_schema(topic, Duration::from_secs(2))
                    .await?;
                ensure!(
                    discovered.schema_hash == hash && discovered.root_name == name,
                    "Schema changed during discovery"
                );
                discovered.schema
            }
        };
        schemas.insert((name.to_owned(), hash), schema.clone());
        schema
    };
    Ok(Channel {
        topic: topic.to_owned(),
        name: name.to_owned(),
        schema: serde_json::to_vec(schema.as_ref())?,
        metadata: BTreeMap::from([
            ("ros_z.type_name".into(), name.to_owned()),
            ("ros_z.replay_clock".into(), "log_time".into()),
            ("ros_z.schema_hash".into(), hash.to_hash_string()),
            (
                "simulator.clock".into(),
                "simulation time at receipt; publish_time preserves source clock".into(),
            ),
        ]),
    })
}

fn write(file: File, mut receiver: mpsc::Receiver<WriteSample>) -> Result<()> {
    let mut file = BufWriter::new(file);
    let mut writer = mcap::WriteOptions::new()
        .compression(Some(mcap::Compression::Lz4))
        .create(&mut file)?;
    let receipt_type = Receipt::type_info();
    let schema_id = writer.add_schema(
        &receipt_type.name,
        "ros-z-schema-json",
        &serde_json::to_vec(&Receipt::schema())?,
    )?;
    let receipt_channel = writer.add_channel(
        schema_id,
        RECEIPT_TOPIC,
        "ros-z-cdr",
        &BTreeMap::from([
            ("ros_z.type_name".into(), receipt_type.name),
            (
                "ros_z.schema_hash".into(),
                receipt_type.hash.to_hash_string(),
            ),
        ]),
    )?;
    let mut channels = BTreeMap::new();
    while let Some(WriteSample { channel, envelope }) = receiver.blocking_recv() {
        let channel_id = if let Some(id) = channels.get(&(
            channel.topic.clone(),
            channel.metadata["ros_z.schema_hash"].clone(),
        )) {
            *id
        } else {
            let schema = writer.add_schema(&channel.name, "ros-z-schema-json", &channel.schema)?;
            let id = writer.add_channel(schema, &channel.topic, "ros-z-cdr", &channel.metadata)?;
            channels.insert(
                (
                    channel.topic.clone(),
                    channel.metadata["ros_z.schema_hash"].clone(),
                ),
                id,
            );
            id
        };
        let attachment = envelope
            .sample
            .attachment()
            .ok_or_else(|| eyre!("Missing ROS-Z attachment on {}", channel.topic))?;
        let attachment = Attachment::try_from(attachment).map_err(|e| eyre!("{e}"))?;
        let sequence = u32::try_from(attachment.sequence_number).unwrap_or(u32::MAX);
        writer.write_to_known_channel(
            &mcap::records::MessageHeader {
                channel_id,
                sequence,
                log_time: envelope.simulation_time,
                publish_time: attachment.source_time().as_nanos() as u64,
            },
            envelope.sample.payload().to_bytes().as_ref(),
        )?;
        let receipt = Receipt {
            channel_id,
            publisher: attachment.source_global_id,
            sequence: attachment.sequence_number,
            wall_time: envelope.wall_time,
            transport_time: envelope.sample.timestamp().map(|t| {
                ros_z::time::Time::from_wallclock(t.get_time().to_system_time()).as_nanos()
            }),
        };
        writer.write_to_known_channel(
            &mcap::records::MessageHeader {
                channel_id: receipt_channel,
                sequence,
                log_time: envelope.simulation_time,
                publish_time: envelope.simulation_time,
            },
            &SerdeCdrCodec::<Receipt>::serialize(&receipt)?,
        )?;
    }
    writer.finish()?;
    drop(writer);
    file.flush()?;
    Ok(())
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resolves_queued_primitive_and_optional_topics_after_producer_teardown() {
        let context = ContextBuilder::default()
            .with_namespace("/schema_capture")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap();
        let recorder = context.create_node("recorder").build().await.unwrap();
        let local_schemas = LocalSchemaCapture::default();
        let producer = context.create_node("producer").build().await.unwrap();
        let primitive = producer
            .publisher::<u16>("primitive")
            .build()
            .await
            .unwrap();
        let optional = producer
            .publisher::<Option<u16>>("optional")
            .build()
            .await
            .unwrap();
        drop((primitive, optional, producer));
        let mut schemas = std::collections::HashMap::new();
        for (topic, info) in [
            ("primitive", u16::type_info()),
            ("optional", Option::<u16>::type_info()),
        ] {
            let key = format!(
                "rt/schema_capture/{topic}/{}/{}",
                info.name,
                info.hash.to_hash_string()
            );
            let channel = resolve_channel(
                &recorder,
                &key,
                "rt/schema_capture/",
                &local_schemas,
                &mut schemas,
            )
            .await
            .unwrap();
            assert_eq!(channel.topic, topic);
            assert_eq!(channel.name, info.name);
            assert_eq!(
                channel.metadata["ros_z.schema_hash"],
                info.hash.to_hash_string()
            );
        }
        context.shutdown().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn full_disk_failure_stays_inside_the_recording_worker() {
        let file = File::options().write(true).open("/dev/full").unwrap();
        let (sender, receiver) = mpsc::channel(1);
        drop(sender);
        // mcap may panic when an I/O failure occurs while opening a chunk.
        // Both I/O errors and writer panics are contained by the blocking task.
        let result = tokio::task::spawn_blocking(move || write(file, receiver)).await;
        assert!(!matches!(result, Ok(Ok(()))));
    }
}
