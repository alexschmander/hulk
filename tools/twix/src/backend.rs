use std::{
    collections::BTreeSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use crate::replay::ReplaySession;
use color_eyre::{Result, eyre::Context as _};
use ros_z::{context::ContextBuilder, graph::Graph, node::Node, prelude::*};
use ros_z_debug::{
    TopicObserver, TopicObserverOptions,
    query::{Connection, Cursor, CursorHandle, Live, LiveRetention, RemoteMcap, TopicList},
};
use tokio::runtime::Handle;
use uuid::Uuid;

pub struct RobotBackend {
    runtime_handle: Handle,
    #[cfg_attr(not(feature = "simulator"), allow(dead_code))]
    router: Option<String>,
    transport_scope: Option<String>,
    namespace_discovery: Option<crate::namespace_discovery::NamespaceDiscovery>,
    context: Arc<Context>,
    node: Arc<Node>,
    observer: TopicObserver,
    namespace: Mutex<String>,
    connection_generation: AtomicU64,
    replay: Mutex<ReplaySession>,
    connection: Mutex<Connection>,
    live: Arc<Live>,
    replay_revision: AtomicU64,
    replay_active: AtomicBool,
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
        let namespace_discovery = if transport_scope.as_deref() == Some("hulk_simulator") {
            Some(crate::namespace_discovery::NamespaceDiscovery::new(context.session()).await?)
        } else {
            None
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
        let live = Arc::new(Live::new(Arc::clone(&node), LiveRetention::default()));
        let connection = Connection {
            backend: live.clone(),
            cursor: CursorHandle::new(Cursor::Live),
        };
        observer.set_backend(connection.clone());

        Ok(Self {
            runtime_handle,
            router,
            transport_scope,
            namespace_discovery,
            context,
            node,
            observer,
            namespace: Mutex::new(namespace),
            connection_generation: AtomicU64::new(0),
            replay: Mutex::new(Default::default()),
            connection: Mutex::new(connection),
            live,
            replay_revision: AtomicU64::new(0),
            replay_active: AtomicBool::new(false),
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

    pub fn query_connection(&self) -> Connection {
        self.connection.lock().expect("connection mutex").clone()
    }

    pub fn topics(&self) -> TopicList {
        self.connection
            .lock()
            .expect("connection mutex")
            .backend
            .topics()
    }

    pub fn use_recording(&self, remote: Arc<RemoteMcap>, cursor: CursorHandle) {
        self.set_connection(Connection {
            backend: remote,
            cursor,
        });
    }

    pub fn use_live(&self) {
        self.set_connection(Connection {
            backend: self.live.clone(),
            cursor: CursorHandle::new(Cursor::Live),
        });
    }

    fn set_connection(&self, connection: Connection) {
        self.observer.set_backend(connection.clone());
        let mut current = self.connection.lock().expect("connection mutex");
        *current = connection;
        self.connection_generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn namespace_completions(&self) -> Vec<String> {
        if let Some(discovery) = &self.namespace_discovery {
            return discovery.namespaces();
        }
        self.graph()
            .lock()
            .nodes()
            .filter(|node| !node.namespace.starts_with("/_"))
            .map(|node| node.namespace.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn namespace(&self) -> String {
        self.namespace
            .lock()
            .expect("namespace mutex should not be poisoned")
            .clone()
    }

    pub fn namespaces(&self) -> Vec<String> {
        let graph = self.graph().lock();
        graph
            .nodes()
            .map(|node| node.namespace.trim_matches('/'))
            .filter(|namespace| *namespace != "_twix")
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|namespace| format!("/{namespace}"))
            .collect()
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

    /// Connection replacements and cursor seeks independently invalidate presentation state.
    pub fn presentation_identity(&self) -> (u64, u64) {
        let connection = self.connection.lock().expect("connection mutex");
        (
            self.connection_generation.load(Ordering::Relaxed),
            connection.cursor.position().revision,
        )
    }

    pub fn set_replay_sources(&self, sources: Option<ros_z_debug::replay::ReplaySources>) {
        self.replay_active
            .store(sources.is_some(), Ordering::Relaxed);
        self.observer.set_replay_sources(sources);
        self.replay_revision.fetch_add(1, Ordering::Relaxed);
    }

    pub fn is_replaying(&self) -> bool {
        self.replay_active.load(Ordering::Relaxed)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn connection_replacements_have_distinct_presentation_identities_at_revision_zero() {
        let backend = RobotBackend::new(Handle::current(), None, "/".into())
            .await
            .unwrap();
        let initial = backend.presentation_identity();
        assert_eq!(initial.1, 0);

        backend.set_connection(Connection {
            backend: Arc::new(Live::new(backend.node(), LiveRetention::default())),
            cursor: CursorHandle::new(Cursor::Live),
        });
        let replacement = backend.presentation_identity();
        assert_eq!(replacement.1, 0);
        assert_ne!(replacement.0, initial.0);

        backend.use_live();
        let live = backend.presentation_identity();
        assert_eq!(live.1, 0);
        assert_ne!(live.0, replacement.0);
        assert_ne!(live.0, initial.0);
    }
}

#[cfg(test)]
mod namespace_tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread")]
    async fn completions_use_node_namespaces_and_remove_departed_nodes() {
        let router = ContextBuilder::default()
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap();
        let endpoint = router.session().info().locators().await[0].to_string();
        let backend = RobotBackend::new(Handle::current(), Some(endpoint), "/42".into())
            .await
            .unwrap();
        let a = router
            .create_node("motion")
            .with_namespace("/42")
            .build()
            .await
            .unwrap();
        let b = router
            .create_node("behavior")
            .with_namespace("/42")
            .build()
            .await
            .unwrap();
        let c = router
            .create_node("motion")
            .with_namespace("/team/43")
            .build()
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while backend.namespace_completions() != ["/42", "/team/43"] {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        drop(c);
        tokio::time::timeout(Duration::from_secs(3), async {
            while backend.namespace_completions() != ["/42"] {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        drop((a, b));
        drop(backend);
        router.shutdown().unwrap();
    }
}
