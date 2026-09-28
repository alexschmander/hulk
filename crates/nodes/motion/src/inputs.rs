use std::sync::Arc;
use std::time::Duration;

use color_eyre::{
    Result,
    eyre::{ensure, eyre},
};
use ros_z::{
    Message,
    prelude::*,
    pubsub::Received,
    time::{Clock, Time},
};
use tokio::{sync::watch, task::JoinHandle};

pub(super) struct Sample<T> {
    pub received: Received<T>,
    pub receipt_time: Time,
}

impl<T> Sample<T> {
    pub fn validate_freshness(&self, now: Time, age: Duration) -> Result<()> {
        let now_ns = now.as_nanos();
        let source_age_ns = i128::from(now_ns) - i128::from(self.received.source_time.as_nanos());
        let receipt_age_ns = i128::from(now_ns) - i128::from(self.receipt_time.as_nanos());
        ensure!(
            self.received.source_time <= now && self.receipt_time <= now,
            "input timestamp is in the future: now_ns={now_ns}, \
             source_age_ns={source_age_ns}, receipt_age_ns={receipt_age_ns}"
        );
        ensure!(
            now.duration_since(self.received.source_time) <= age
                && now.duration_since(self.receipt_time) <= age,
            "input expired: now_ns={now_ns}, source_age_ns={source_age_ns}, \
             receipt_age_ns={receipt_age_ns}, maximum_age_ns={}",
            age.as_nanos()
        );
        Ok(())
    }
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
                // Commands can change while logical time is paused. Preserve
                // arrival order for equal timestamps without renewing source age.
                let receipt_time = clock.now();
                if received.source_time <= receipt_time {
                    if last.is_some_and(|time| received.source_time < time) {
                        continue;
                    }
                    last = Some(received.source_time);
                }
                // Surface invalid future input, but let the next valid sample
                // recover instead of advancing the ordering watermark.
                sender.send_replace(Some(Arc::new(Sample {
                    received,
                    receipt_time,
                })));
            }
        });
        Ok(Self { values, task })
    }
    pub fn latest(&self) -> Option<Arc<Sample<T>>> {
        self.values.borrow().clone()
    }
    pub fn snapshot(&self) -> Result<Arc<Sample<T>>> {
        ensure!(!self.task.is_finished(), "input receiver stopped");
        self.latest().ok_or_else(|| eyre!("input unavailable"))
    }
    pub fn fresh(&self, clock: &Clock, age: Duration) -> Result<Arc<Sample<T>>> {
        let sample = self.snapshot()?;
        // A subscription may update concurrently, so capture the sample before the clock.
        sample.validate_freshness(clock.now(), age)?;
        Ok(sample)
    }
}
impl<T> Drop for Latest<T> {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::motion_command::MotionCommand;

    #[tokio::test(flavor = "multi_thread")]
    async fn command_updates_preserve_source_age_and_accept_stops_while_paused() {
        let clock = Clock::logical(Time::from_nanos(1_000_000_000));
        let context = ContextBuilder::default()
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .with_clock(clock.clone())
            .build()
            .await
            .unwrap();
        let node = context
            .create_node("motion_input_test")
            .build()
            .await
            .unwrap();
        let publisher = node
            .publisher::<MotionCommand>("command")
            .build()
            .await
            .unwrap();
        let latest = Latest::<MotionCommand>::subscribe(&node, "command", QosProfile::default())
            .await
            .unwrap();
        let mut changes = latest.values.clone();
        let source = clock.now();
        for command in [MotionCommand::Prepare, MotionCommand::Damping] {
            publisher.publish(&command).await.unwrap();
            tokio::time::timeout(Duration::from_secs(2), changes.changed())
                .await
                .unwrap()
                .unwrap();
        }
        assert!(matches!(
            latest
                .fresh(&clock, Duration::from_millis(10))
                .unwrap()
                .received
                .message,
            MotionCommand::Damping
        ));
        clock.advance(Duration::from_millis(20)).unwrap();
        publisher
            .publish_with_source_time(&MotionCommand::Prepare, source)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), changes.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(
            latest.fresh(&clock, Duration::from_millis(10)).is_err(),
            "retransmission cannot renew source age"
        );
        publisher
            .publish_with_source_time(
                &MotionCommand::Damping,
                clock.now() + Duration::from_millis(20),
            )
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), changes.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(
            latest.fresh(&clock, Duration::from_millis(100)).is_err(),
            "future source times are invalid"
        );
        publisher.publish(&MotionCommand::Damping).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), changes.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(
            latest.fresh(&clock, Duration::from_millis(10)).is_ok(),
            "valid input recovers after a future timestamp"
        );
        context.shutdown().unwrap();
    }
}
