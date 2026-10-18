//! Observe native subscriptions, including raw MCAP capture, without polling the ROS-Z graph.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use color_eyre::{Result, eyre::eyre};
use ros_z::{Message, prelude::*};

#[derive(Clone)]
pub(crate) struct Demand {
    active: Arc<AtomicBool>,
    // This publisher only owns a matching listener. Data uses the original ROS-Z publisher.
    _monitor: Arc<zenoh::pubsub::Publisher<'static>>,
}

impl Demand {
    pub async fn new<T: Message>(node: &Node, publisher: &Publisher<T>) -> Result<Self> {
        let key = ros_z_protocol::format::topic_key_expr(publisher.entity())?;
        let monitor = node
            .session()
            .declare_publisher((*key).clone())
            .await
            .map_err(|error| eyre!("Ground-truth demand publisher: {error}"))?;
        let active = Arc::new(AtomicBool::new(false));
        let matching = active.clone();
        monitor
            .matching_listener()
            .callback(move |status| matching.store(status.matching(), Ordering::Relaxed))
            .background()
            .await
            .map_err(|error| eyre!("Ground-truth matching listener: {error}"))?;
        Ok(Self {
            active,
            _monitor: Arc::new(monitor),
        })
    }

    pub fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}
