//! Discover ROS namespaces inside child Zenoh scopes without joining their traffic.
use color_eyre::Result;
use ros_z_protocol::{
    Entity,
    format::{ADMIN_SPACE, parse_liveliness},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};
use zenoh::{Session, pubsub::Subscriber, sample::SampleKind};

pub struct NamespaceDiscovery {
    nodes: Arc<Mutex<BTreeMap<String, String>>>,
    _subscriber: Subscriber<()>,
}

impl NamespaceDiscovery {
    pub async fn new(session: &Session) -> Result<Self> {
        let nodes = Arc::new(Mutex::new(BTreeMap::new()));
        let discovered = nodes.clone();
        let subscriber = session
            .liveliness()
            .declare_subscriber(format!("**/{ADMIN_SPACE}/**"))
            .history(true)
            .callback(move |sample| {
                let key = sample.key_expr().as_str();
                match sample.kind() {
                    SampleKind::Put => {
                        if let Some(namespace) = child_node_namespace(key) {
                            discovered.lock().unwrap().insert(key.to_owned(), namespace);
                        }
                    }
                    SampleKind::Delete => {
                        discovered.lock().unwrap().remove(key);
                    }
                }
            })
            .await
            .map_err(|error| color_eyre::eyre::eyre!("Namespace discovery: {error}"))?;
        Ok(Self {
            nodes,
            _subscriber: subscriber,
        })
    }

    pub fn namespaces(&self) -> Vec<String> {
        self.nodes
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

fn child_node_namespace(key: &str) -> Option<String> {
    let (_, token) = key.split_once(&format!("/{ADMIN_SPACE}/"))?;
    let key = format!("{ADMIN_SPACE}/{token}").try_into().ok()?;
    let Entity::Node(node) = parse_liveliness(&key).ok()? else {
        return None;
    };
    (!node.namespace.starts_with("/_")).then_some(node.namespace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ros_z::context::ContextBuilder;
    use std::time::Duration;

    async fn wait_for(discovery: &NamespaceDiscovery, expected: &[&str]) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while discovery.namespaces() != expected {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("namespace discovery did not converge");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn discovers_history_and_dynamic_players_across_scopes_and_removes_departed_nodes() {
        let scope = format!("twix_discovery_{}", uuid::Uuid::new_v4().simple());
        let root = ContextBuilder::default()
            .with_json("namespace", &scope)
            .with_mode("router")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(["tcp/127.0.0.1:0"])
            .build()
            .await
            .unwrap();
        let endpoint = root.session().info().locators().await[0].to_string();
        let first = ContextBuilder::default()
            .with_router_endpoint(&endpoint)
            .unwrap()
            .with_json("namespace", format!("{scope}/hulks/1"))
            .with_namespace("/hulks/1")
            .build()
            .await
            .unwrap();
        let node1 = first.create_node("behavior").build().await.unwrap();
        let node2 = first.create_node("motion").build().await.unwrap();
        let _internal = first
            .create_node("twix")
            .with_namespace("/_twix")
            .build()
            .await
            .unwrap();
        let discovery = NamespaceDiscovery::new(root.session()).await.unwrap();
        wait_for(&discovery, &["/hulks/1"]).await;
        let second = ContextBuilder::default()
            .with_router_endpoint(&endpoint)
            .unwrap()
            .with_json("namespace", format!("{scope}/opponents/1"))
            .with_namespace("/opponents/1")
            .build()
            .await
            .unwrap();
        let other = second.create_node("behavior").build().await.unwrap();
        wait_for(&discovery, &["/hulks/1", "/opponents/1"]).await;
        drop(node1);
        // Removing one node must not remove the namespace's remaining nodes.
        drop(other);
        wait_for(&discovery, &["/hulks/1"]).await;
        drop(node2);
        wait_for(&discovery, &[]).await;
        first.shutdown().unwrap();
        second.shutdown().unwrap();
        root.shutdown().unwrap();
    }
}
