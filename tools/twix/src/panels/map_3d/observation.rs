use std::{num::NonZeroUsize, sync::Arc, time::Duration};

use color_eyre::Result;
use ros_z::Message;
use ros_z_debug::{
    ObservationPolicy, RetentionPolicy, SampleRecord, TargetIdentity, TopicObservation,
    TopicReference,
};

use crate::repaint::{ObservationContext, ObservationRepaint, RepaintOnUpdates};

pub(super) struct Observation<T> {
    observation: TopicObservation<T>,
    topic: TopicReference,
    _repaint: ObservationRepaint,
}

impl<T> Observation<T>
where
    T: Message + Send + Sync + 'static,
    T::Codec: Send + Sync,
{
    pub(super) fn new(
        context: &impl ObservationContext,
        topic: &'static str,
        capacity: usize,
        policy: ObservationPolicy,
    ) -> Result<Self> {
        let _runtime = context.backend().runtime_handle().enter();
        let observation = context
            .backend()
            .observer()
            .observe_typed::<T>(topic)?
            .policy(policy)
            .retention(RetentionPolicy::time_window_with_max_samples(
                Duration::from_secs(2),
                NonZeroUsize::new(capacity).expect("nonzero history capacity"),
            )?)
            .spawn();
        let repaint = observation.repaint_on_updates(context);
        Ok(Self {
            observation,
            topic: TopicReference::new(topic)?,
            _repaint: repaint,
        })
    }
}

impl<T> Observation<T> {
    pub(super) fn latest(&self, namespace: &str) -> Option<Arc<SampleRecord<T>>> {
        let topic = self
            .topic
            .resolve(&TargetIdentity::new(namespace).ok()?)
            .ok()?;
        self.observation
            .latest()
            .filter(|record| record.metadata.resolved_topic == topic)
    }
}
