use std::{sync::Arc, time::Duration};

use eframe::egui;
use robot_logs::{ReadLogs, ReadLogsRequest, ReadLogsResponse};
use ros_z::node::Node;
use tokio::{sync::mpsc, task::JoinHandle};

pub(super) enum Update {
    Page(ReadLogsResponse),
    Error(String),
}

/// Fetch from the start and keep the same file cursor while following new records.
/// The queue bounds work waiting for the UI; dropping it cancels the old namespace.
pub(super) struct History {
    pub updates: mpsc::Receiver<Update>,
    task: JoinHandle<()>,
}
impl History {
    pub fn start(node: Arc<Node>, namespace: &str, repaint: egui::Context) -> Self {
        let service = format!("{}/logs/history", namespace.trim_end_matches('/'));
        let (sender, updates) = mpsc::channel(4);
        let task = tokio::spawn(async move {
            let client = match node.service_client::<ReadLogs>(&service).build().await {
                Ok(client) => client,
                Err(error) => {
                    let _ = sender.send(Update::Error(error.to_string())).await;
                    repaint.request_repaint();
                    return;
                }
            };
            let mut request = ReadLogsRequest::default();
            loop {
                let response = client
                    .call_with_timeout_async(&request, Duration::from_secs(2))
                    .await;
                let (update, caught_up) = match response {
                    Ok(page) if page.error.is_empty() => {
                        let caught_up = page.next_offset == page.end;
                        request.session = page.session.clone();
                        request.offset = page.next_offset;
                        (Update::Page(page), caught_up)
                    }
                    Ok(page) => (Update::Error(page.error), true),
                    Err(error) => (Update::Error(error.to_string()), true),
                };
                let failed = matches!(update, Update::Error(_));
                if sender.send(update).await.is_err() {
                    return;
                }
                repaint.request_repaint();
                if caught_up {
                    tokio::time::sleep(if failed {
                        Duration::from_secs(1)
                    } else {
                        Duration::from_millis(250)
                    })
                    .await;
                }
            }
        });
        Self { updates, task }
    }
}
impl Drop for History {
    fn drop(&mut self) {
        self.task.abort();
    }
}
