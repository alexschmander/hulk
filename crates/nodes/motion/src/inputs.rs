use std::sync::Arc;
use std::time::Duration;

use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::{Message, prelude::*, pubsub::Received, time::Time};
use tokio::{sync::watch, task::JoinHandle};

pub(super) struct Sample<T> {
    pub received: Received<T>,
    pub receipt_time: Time,
}

/// Retains source and local receipt times; a stopped receiver cannot renew its lease.
pub(super) struct Latest<T> {
    values: watch::Receiver<Option<Arc<Sample<T>>>>,
    task: JoinHandle<()>,
}
impl<T: Message + Send + Sync + 'static> Latest<T>
where
    T::Codec: Send + Sync,
{
    pub async fn subscribe(node: &Node, topic: &str, qos: QosProfile) -> Result<Self> {
        let subscriber = node.subscriber::<T>(topic).qos(qos).build().await?;
        let (sender, values) = watch::channel(None);
        let clock = node.clock().clone();
        let task = tokio::spawn(async move {
            let mut last = None;
            while let Ok(received) = subscriber.recv_with_metadata().await {
                if last.is_some_and(|time| received.source_time <= time) {
                    continue;
                }
                last = Some(received.source_time);
                sender.send_replace(Some(Arc::new(Sample {
                    received,
                    receipt_time: clock.now(),
                })));
            }
        });
        Ok(Self { values, task })
    }
    pub fn latest(&self) -> Option<Arc<Sample<T>>> {
        self.values.borrow().clone()
    }
    pub fn fresh(&self, now: Time, age: Duration) -> Result<Arc<Sample<T>>> {
        ensure!(!self.task.is_finished(), "input receiver stopped");
        let sample = self.latest().ok_or_else(|| eyre!("input unavailable"))?;
        ensure!(
            sample.received.source_time <= now && sample.receipt_time <= now,
            "input timestamp is in the future"
        );
        ensure!(
            now.duration_since(sample.received.source_time) <= age
                && now.duration_since(sample.receipt_time) <= age,
            "input expired"
        );
        Ok(sample)
    }
}
impl<T> Drop for Latest<T> {
    fn drop(&mut self) {
        self.task.abort();
    }
}
