//! Capture before publisher startup and retain local schemas while queued samples drain.
use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::dynamic::schema_service::LocalSchemaCapture;
use ros_z::message::{CdrCodec, WireEncoder};
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
pub(crate) struct Receipt {
    pub(crate) channel_id: u16,
    pub(crate) publisher: ros_z::attachment::EndpointGlobalId,
    pub(crate) sequence: i64,
    pub(crate) wall_time: i64,
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Checkpoint {
    pub files: Vec<std::path::PathBuf>,
    pub simulation_time: i64,
    pub wall_time: i64,
    pub ingestion_sequence: u64,
    pub dropped_samples: u64,
    pub finalized: bool,
}
type CheckpointReply = tokio::sync::oneshot::Sender<Result<Checkpoint>>;
struct Admission {
    sequence: u64,
    dropped: u64,
}

enum Ingest {
    Sample(SampleEnvelope),
    Checkpoint(Checkpoint, CheckpointReply),
}
enum WriteEvent {
    Sample(WriteSample),
    Checkpoint(Checkpoint, CheckpointReply),
    Finish(Checkpoint),
}
pub struct Recording {
    sender: mpsc::Sender<Ingest>,
    admission: Arc<std::sync::Mutex<Admission>>,
    clock: Clock,
    stop: tokio_util::sync::CancellationToken,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    pub dropped: Arc<AtomicU64>,
    _runtime: Arc<super::RobotRuntime>,
}
impl Recording {
    pub async fn start(
        context: &Arc<Context>,
        node: Arc<Node>,
        namespace: &str,
        directory: &std::path::Path,
        parameters: &super::Parameters,
        health: super::Health,
        runtime: Arc<super::RobotRuntime>,
    ) -> Result<Self> {
        let local_schemas = LocalSchemaCapture::default();
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join("recording.mcap"))?;
        let (sender, mut receiver) = mpsc::channel(parameters.queue_capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let drops = dropped.clone();
        let admission = Arc::new(std::sync::Mutex::new(Admission {
            sequence: 0,
            dropped: 0,
        }));
        let admitted = admission.clone();
        let ingest = sender.clone();
        let directory = directory.to_owned();
        let clock = context.clock().clone();
        let receipt_clock = clock.clone();
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
                let mut admitted = admitted.lock().unwrap();
                let envelope = SampleEnvelope {
                    sample,
                    simulation_time: receipt_clock.now().as_nanos() as u64,
                    wall_time: Clock::wallclock().now().as_nanos(),
                };
                if ingest.try_send(Ingest::Sample(envelope)).is_err() {
                    drops.fetch_add(1, Ordering::Relaxed);
                    admitted.dropped += 1;
                } else {
                    admitted.sequence += 1;
                }
            })
            .await
            .map_err(|e| eyre!("Cannot capture robot topics: {e}"))?;
        let stop = tokio_util::sync::CancellationToken::new();
        let cancelled = stop.clone();
        let capacity = parameters.queue_capacity;
        let lost = dropped.clone();
        let closing_admission = admission.clone();
        let closing_clock = clock.clone();
        use tracing::instrument::WithSubscriber;
        let owner = runtime.clone();
        let work = async move {
            // Also retain the runtime when startup cleanup drops the recorder.
            let _runtime = owner;
            let (writer_tx, writer_rx) = mpsc::channel(capacity);
            let writer = tokio::task::spawn_blocking(move || write(file, directory, writer_rx));
            let mut channels = BTreeMap::<String, Arc<Channel>>::new();
            let mut schemas = std::collections::HashMap::new();
            let mut subscriber = Some(subscriber);
            let mut failed = std::collections::HashSet::new();
            let mut unresolved = 0;
            loop {
                let next = tokio::select! {
                    biased;
                    _ = cancelled.cancelled(), if subscriber.is_some() => {
                        if let Some(subscriber) = subscriber.take() { subscriber.undeclare().await.map_err(|e| eyre!("{e}"))?; }
                        receiver.close();
                        continue;
                    }
                    next = receiver.recv() => next,
                };
                let Some(envelope) = next else {
                    break;
                };
                let envelope = match envelope {
                    Ingest::Sample(envelope) => envelope,
                    Ingest::Checkpoint(mut checkpoint, reply) => {
                        checkpoint.dropped_samples += unresolved;
                        writer_tx
                            .send(WriteEvent::Checkpoint(checkpoint, reply))
                            .await
                            .map_err(|_| eyre!("MCAP writer stopped"))?;
                        continue;
                    }
                };
                let key = envelope.sample.key_expr().as_str();
                if failed.contains(key) {
                    lost.fetch_add(1, Ordering::Relaxed);
                    unresolved += 1;
                    continue;
                }
                let channel = if let Some(channel) = channels.get(key) {
                    channel.clone()
                } else {
                    let channel =
                        match resolve_channel(&node, key, &prefix, &local_schemas, &mut schemas)
                            .await
                        {
                            Ok(channel) => Arc::new(channel),
                            Err(error) => {
                                health.error(format!("Cannot record {key}: {error:#}"));
                                failed.insert(key.to_owned());
                                lost.fetch_add(1, Ordering::Relaxed);
                                unresolved += 1;
                                continue;
                            }
                        };
                    channels.insert(key.to_owned(), channel.clone());
                    channel
                };
                if writer_tx
                    .send(WriteEvent::Sample(WriteSample { channel, envelope }))
                    .await
                    .is_err()
                {
                    health.error("MCAP writer stopped");
                    break;
                }
            }
            drop(subscriber);
            let checkpoint = {
                let admitted = closing_admission.lock().unwrap();
                Checkpoint {
                    files: Vec::new(),
                    simulation_time: closing_clock.now().as_nanos(),
                    wall_time: Clock::wallclock().now().as_nanos(),
                    ingestion_sequence: admitted.sequence,
                    dropped_samples: admitted.dropped + unresolved,
                    finalized: true,
                }
            };
            writer_tx
                .send(WriteEvent::Finish(checkpoint))
                .await
                .map_err(|_| eyre!("MCAP writer stopped"))?;
            drop(writer_tx);
            writer.await??;
            Ok(())
        };
        // Keep capture beside its producers to avoid cross-runtime wakeups on
        // every sample. The owner also keeps this runtime alive while a dropped
        // recorder drains during failed or cancelled robot startup.
        let task = runtime
            .handle()
            .spawn(work.with_subscriber(tracing::Dispatch::none()));
        Ok(Self {
            sender,
            admission,
            clock,
            stop,
            task: Some(task),
            dropped,
            _runtime: runtime,
        })
    }
    /// Serialize a checkpoint behind all samples already admitted to the ingest queue.
    pub(crate) async fn checkpoint(&self) -> Result<Checkpoint> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let permit = self
            .sender
            .reserve()
            .await
            .map_err(|_| eyre!("Recording closed"))?;
        {
            // Queue insertion and receipt timestamps share one admission boundary with the callback.
            let admitted = self.admission.lock().unwrap();
            permit.send(Ingest::Checkpoint(
                Checkpoint {
                    files: Vec::new(),
                    simulation_time: self.clock.now().as_nanos(),
                    wall_time: Clock::wallclock().now().as_nanos(),
                    ingestion_sequence: admitted.sequence,
                    dropped_samples: admitted.dropped,
                    finalized: false,
                },
                tx,
            ));
        }
        rx.await?
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

fn write(
    file: File,
    directory: std::path::PathBuf,
    mut receiver: mpsc::Receiver<WriteEvent>,
) -> Result<()> {
    let mut completed = Vec::new();
    let mut path = directory.join("recording.mcap");
    let mut next_file = file;
    let mut final_checkpoint = Checkpoint::default();
    let mut receipt_buffer = Vec::new();
    'segments: loop {
        let mut file = BufWriter::new(next_file);
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
        let mut channels = BTreeMap::<String, BTreeMap<String, u16>>::new();
        while let Some(event) = receiver.blocking_recv() {
            let WriteEvent::Sample(WriteSample { channel, envelope }) = event else {
                let (mut checkpoint, reply) = match event {
                    WriteEvent::Checkpoint(checkpoint, reply) => (checkpoint, reply),
                    WriteEvent::Finish(checkpoint) => {
                        final_checkpoint = checkpoint;
                        break;
                    }
                    _ => unreachable!(),
                };
                writer.finish()?;
                drop(writer);
                file.flush()?;
                completed.push(path.file_name().unwrap().into());
                checkpoint.files = completed.clone();
                // Publish only finalized files. A crash can leave the current tail incomplete.
                let temporary = directory.join("recording-segments.json.tmp");
                std::fs::write(&temporary, serde_json::to_vec(&checkpoint)?)?;
                std::fs::rename(temporary, directory.join("recording-segments.json"))?;
                path = directory.join(format!("recording-{:06}.mcap", completed.len()));
                next_file = File::options().write(true).create_new(true).open(&path)?;
                let _ = reply.send(Ok(checkpoint));
                continue 'segments;
            };
            let hash = &channel.metadata["ros_z.schema_hash"];
            let channel_id = if let Some(id) = channels
                .get(&channel.topic)
                .and_then(|types| types.get(hash))
            {
                *id
            } else {
                let schema =
                    writer.add_schema(&channel.name, "ros-z-schema-json", &channel.schema)?;
                let id =
                    writer.add_channel(schema, &channel.topic, "ros-z-cdr", &channel.metadata)?;
                channels
                    .entry(channel.topic.clone())
                    .or_default()
                    .insert(hash.clone(), id);
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
            CdrCodec::<Receipt>::serialize_to_buf(&receipt, &mut receipt_buffer)?;
            writer.write_to_known_channel(
                &mcap::records::MessageHeader {
                    channel_id: receipt_channel,
                    sequence,
                    log_time: envelope.simulation_time,
                    publish_time: envelope.simulation_time,
                },
                &receipt_buffer,
            )?;
        }
        writer.finish()?;
        drop(writer);
        file.flush()?;
        completed.push(path.file_name().unwrap().into());
        final_checkpoint.files = completed;
        let temporary = directory.join("recording-segments.json.tmp");
        std::fs::write(&temporary, serde_json::to_vec(&final_checkpoint)?)?;
        std::fs::rename(temporary, directory.join("recording-segments.json"))?;
        return Ok(());
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn dropping_capture_drains_queued_samples_and_releases_its_runtime() {
        let context = Arc::new(
            ContextBuilder::default()
                .with_namespace("/capture_lifetime")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(std::iter::empty::<&str>())
                .build()
                .await
                .unwrap(),
        );
        let node = Arc::new(context.create_node("capture").build().await.unwrap());
        let directory = tempfile::tempdir().unwrap();
        let startup =
            super::super::Startup::new(directory.path().join("logs"), context.clock().clone())
                .unwrap();
        let owner = Arc::new(super::super::RobotRuntime::new(startup.capture).unwrap());
        let recording = Recording::start(
            &context,
            node.clone(),
            "/capture_lifetime",
            directory.path(),
            &super::super::Parameters {
                mcap_enabled: true,
                topics: vec!["**".into()],
                queue_capacity: 8192,
                capture_level: 3,
            },
            super::super::Health::default(),
            owner.clone(),
        )
        .await
        .unwrap();
        let runtime = Arc::downgrade(&recording._runtime);
        let publisher = node.publisher::<u32>("sample").build().await.unwrap();
        for value in 0..20 {
            publisher.publish(&value).await.unwrap();
        }
        drop(recording);
        // Simulate failed startup dropping the robot's last external owner.
        drop(owner);
        tokio::time::timeout(Duration::from_secs(5), async {
            while runtime.strong_count() != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("capture runtime must stop after the queued samples drain");
        let file = File::open(directory.path().join("recording.mcap")).unwrap();
        let mapping = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let summary = mcap::Summary::read(&mapping).unwrap().unwrap();
        let channel = summary
            .channels
            .iter()
            .find(|(_, c)| c.topic == "sample")
            .unwrap()
            .0;
        assert_eq!(
            summary.stats.as_ref().unwrap().channel_message_counts[channel],
            20
        );
        context.shutdown().unwrap();
    }

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
        let result =
            tokio::task::spawn_blocking(move || write(file, std::env::temp_dir(), receiver)).await;
        assert!(!matches!(result, Ok(Ok(()))));
    }
}
