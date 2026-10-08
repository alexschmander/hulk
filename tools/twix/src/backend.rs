use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use color_eyre::{Result, eyre::Context as _};
use ros_z::{context::ContextBuilder, graph::Graph, node::Node, prelude::*};
use ros_z_debug::{TopicObserver, TopicObserverOptions};
use tokio::runtime::Handle;
use uuid::Uuid;

pub struct RobotBackend {
    runtime_handle: Handle,
    #[cfg_attr(not(feature = "simulator"), allow(dead_code))]
    router: Option<String>,
    transport_scope: Option<String>,
    context: Arc<Context>,
    node: Arc<Node>,
    observer: TopicObserver,
    namespace: Mutex<String>,
    replay_revision: AtomicU64,
    replay: Mutex<crate::replay::ReplaySession>,
}

impl RobotBackend {
    #[cfg(test)]
    pub async fn new(
        runtime_handle: Handle,
        router: Option<String>,
        namespace: String,
    ) -> Result<Self> {
        Self::new_scoped(runtime_handle, router, namespace, None).await
    }

    pub async fn new_scoped(
        runtime_handle: Handle,
        router: Option<String>,
        namespace: String,
        transport_scope: Option<String>,
    ) -> Result<Self> {
        let mut builder = ContextBuilder::default();
        if let Some(router) = &router {
            builder = builder
                .with_router_endpoint(router)
                .wrap_err("failed to configure ROS-Z router endpoint")?;
        }

        if let Some(scope) = &transport_scope {
            builder = builder.with_json("namespace", scope);
        }
        #[cfg(feature = "simulator")]
        let local_simulator_router =
            router.is_none() && transport_scope.as_deref() == Some(simulate::ZENOH_NAMESPACE);
        #[cfg(feature = "simulator")]
        if local_simulator_router {
            builder = builder
                .with_mode("router")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints(["tcp/127.0.0.1:0"]);
        }
        let context = Arc::new(
            builder
                .build()
                .await
                .wrap_err("failed to build ROS-Z context")?,
        );
        if let Some(scope) = &transport_scope {
            color_eyre::eyre::ensure!(
                context
                    .session()
                    .config()
                    .get("namespace")
                    .is_ok_and(
                        |value| serde_json::from_str::<String>(&value).ok().as_ref() == Some(scope)
                    ),
                "ZENOH_CONFIG_OVERRIDE conflicts with --zenoh-namespace"
            );
        }
        #[cfg(feature = "simulator")]
        let router = if local_simulator_router {
            Some(
                context
                    .session()
                    .info()
                    .locators()
                    .await
                    .first()
                    .ok_or_else(|| color_eyre::eyre::eyre!("simulator router has no listener"))?
                    .to_string(),
            )
        } else {
            router
        };
        let node_name = twix_node_name();
        let node = Arc::new(
            context
                .create_node(node_name)
                .with_namespace("/_twix")
                .build()
                .await
                .wrap_err("failed to create Twix ROS-Z node")?,
        );
        let options = TopicObserverOptions::with_namespace(namespace.clone())
            .wrap_err("failed to configure initial Twix namespace")?;
        let observer = TopicObserver::new(Arc::clone(&node), options);

        Ok(Self {
            runtime_handle,
            router,
            transport_scope,
            context,
            node,
            observer,
            namespace: Mutex::new(namespace),
            replay_revision: AtomicU64::new(0),
            replay: Mutex::new(Default::default()),
        })
    }

    pub fn transport_scope(&self) -> Option<&str> {
        self.transport_scope.as_deref()
    }

    #[cfg(feature = "simulator")]
    pub fn router(&self) -> Option<&str> {
        self.router.as_deref()
    }

    pub fn runtime_handle(&self) -> &Handle {
        &self.runtime_handle
    }

    pub fn node(&self) -> Arc<Node> {
        self.node.clone()
    }

    pub fn graph(&self) -> &Graph {
        self.context.graph().as_ref()
    }

    pub fn observer(&self) -> &TopicObserver {
        &self.observer
    }

    pub fn namespace(&self) -> String {
        self.namespace
            .lock()
            .expect("namespace mutex should not be poisoned")
            .clone()
    }

    pub fn set_namespace(&self, namespace: String) -> Result<()> {
        self.observer
            .set_namespace(namespace.clone())
            .wrap_err("failed to set Twix target namespace")?;
        *self
            .namespace
            .lock()
            .expect("namespace mutex should not be poisoned") = namespace;
        Ok(())
    }

    pub fn set_replay_sources(&self, sources: Option<ros_z_debug::replay::ReplaySources>) {
        self.observer.set_replay_sources(sources);
        self.replay_revision.fetch_add(1, Ordering::Relaxed);
    }

    pub fn replay_revision(&self) -> u64 {
        self.replay_revision.load(Ordering::Relaxed)
    }

    pub fn replay(&self) -> std::sync::MutexGuard<'_, crate::replay::ReplaySession> {
        self.replay
            .lock()
            .expect("replay mutex should not be poisoned")
    }
}

impl Drop for RobotBackend {
    fn drop(&mut self) {
        if let Err(error) = self.context.shutdown() {
            log::error!("failed to shut down ROS-Z context: {error:#}");
        }
    }
}

fn twix_node_name() -> String {
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown-host".to_string());
    let host = sanitize_node_component(&host);
    let id = Uuid::new_v4().simple().to_string();
    let short_id = &id[..8];
    format!("twix_{short_id}_{host}")
}

fn sanitize_node_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "unknown-host".to_string()
    } else {
        sanitized
    }
}
