//! Optional provenance of the behavior command selected for a kick inference.
use std::sync::Arc;

use color_eyre::Result;
use motion_inference::inference::KickCommand;
use ros_z::{prelude::*, qos::QosReliability, time::Time};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub const KICK_INPUT_TOPIC: &str = "motion/kick_input_trace";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Message)]
pub struct KickInput {
    pub command_time: Time,
    pub prepared_at: Time,
    pub command: KickCommand,
}

pub(super) struct KickInputTracer {
    publisher: Arc<Publisher<KickInput>>,
    sender: mpsc::Sender<KickInput>,
}

impl KickInputTracer {
    pub async fn new(node: &Node) -> Result<Self> {
        let publisher = Arc::new(
            node.publisher(KICK_INPUT_TOPIC)
                .qos(QosProfile {
                    reliability: QosReliability::BestEffort,
                    ..Default::default()
                })
                .build()
                .await?,
        );
        let (sender, mut receiver) = mpsc::channel(32);
        let output = publisher.clone();
        tokio::spawn(async move {
            while let Some(record) = receiver.recv().await {
                if let Err(error) = output.publish(&record).await {
                    tracing::warn!(%error, "kick input trace publication failed");
                }
            }
        });
        Ok(Self { publisher, sender })
    }

    pub fn record(&self, record: KickInput) {
        if self.publisher.has_subscribers() {
            // Diagnostics must not stall motion or service dispatch.
            let _ = self.sender.try_send(record);
        }
    }
}
