use color_eyre::Result;
use ros_z::{prelude::*, time::Clock};
use std::{
    collections::BTreeMap,
    fs::File,
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};
use tokio::sync::mpsc;
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{Layer, layer::Context};

pub use robot_logs::LogRecord;

#[derive(Clone)]
pub struct Capture {
    sender: mpsc::Sender<LogRecord>,
    clock: Clock,
    sequence: Arc<AtomicU64>,
    pub dropped: Arc<AtomicU64>,
    pub level: Arc<AtomicU8>,
}
impl Capture {
    pub fn new(clock: Clock) -> (Self, mpsc::Receiver<LogRecord>) {
        let (sender, receiver) = mpsc::channel(4096);
        (
            Self {
                sender,
                clock,
                sequence: Arc::new(AtomicU64::new(0)),
                dropped: Arc::new(AtomicU64::new(0)),
                level: Arc::new(AtomicU8::new(3)),
            },
            receiver,
        )
    }
}
#[derive(Default)]
struct Fields(BTreeMap<String, String>);
impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }
}
fn severity(level: &tracing::Level) -> u8 {
    match *level {
        tracing::Level::ERROR => 1,
        tracing::Level::WARN => 2,
        tracing::Level::INFO => 3,
        tracing::Level::DEBUG => 4,
        tracing::Level::TRACE => 5,
    }
}
impl<S: Subscriber> Layer<S> for Capture {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn enabled(&self, metadata: &tracing::Metadata<'_>, _: Context<'_, S>) -> bool {
        severity(metadata.level()) <= self.level.load(Ordering::Relaxed)
    }

    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let level = severity(event.metadata().level());
        if level > self.level.load(Ordering::Relaxed) {
            return;
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        let record = LogRecord {
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed),
            simulation_time: self.clock.now().as_nanos(),
            wall_time: Clock::wallclock().now().as_nanos(),
            level,
            target: fields
                .0
                .remove("log.target")
                .unwrap_or_else(|| event.metadata().target().to_owned()),
            message: fields.0.remove("message").unwrap_or_default(),
            fields: serde_json::to_string(&fields.0).unwrap_or_default(),
        };
        if self.sender.try_send(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub struct Logs {
    stop: tokio_util::sync::CancellationToken,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
}
impl Logs {
    pub async fn start(
        node: &Node,
        file: File,
        mut receiver: mpsc::Receiver<LogRecord>,
        failures: super::Health,
    ) -> Result<Self> {
        let publisher = node
            .publisher::<LogRecord>("logs")
            .qos(QosProfile {
                durability: ros_z::qos::QosDurability::TransientLocal,
                history: ros_z::qos::QosHistory::KeepLast(NonZeroUsize::new(512).unwrap()),
                ..Default::default()
            })
            .build()
            .await?;
        let archive = robot_logs::Archive::start(node, file, "logs/history").await?;
        let stop = tokio_util::sync::CancellationToken::new();
        let cancelled = stop.clone();
        // Transport diagnostics must not feed their own log publisher.
        use tracing::instrument::WithSubscriber;
        let task = tokio::spawn(
            async move {
                let mut publish = true;
                loop {
                    let next = tokio::select! {
                        biased;
                        _ = cancelled.cancelled() => { receiver.close(); receiver.recv().await }
                        next = receiver.recv() => next,
                    };
                    let Some(record) = next else {
                        break;
                    };
                    // Preserve the file even when ROS-Z publication fails.
                    if archive.append(record.clone()).await.is_err() {
                        failures.error("Log file writer stopped");
                        break;
                    }
                    if !publish {
                        continue;
                    }
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(1),
                        publisher.publish_with_source_time(
                            &record,
                            ros_z::time::Time::from_nanos(record.simulation_time),
                        ),
                    )
                    .await
                    {
                        Ok(Ok(())) => {}
                        _ => {
                            publish = false;
                            failures.error("Log topic publication failed; file capture continues");
                        }
                    }
                }
                archive.finish().await?;
                Ok(())
            }
            .with_subscriber(tracing::Dispatch::none()),
        );
        Ok(Self {
            stop,
            task: Some(task),
        })
    }
    pub async fn finish(mut self) -> Result<()> {
        self.stop.cancel();
        self.task.take().unwrap().await?
    }
}

impl Drop for Logs {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}
