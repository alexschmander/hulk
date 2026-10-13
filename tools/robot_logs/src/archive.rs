use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
};

use color_eyre::{Result, eyre::ensure};
use ros_z::node::Node;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tracing::instrument::WithSubscriber;

use crate::{LogRecord, ReadLogs, ReadLogsRequest, ReadLogsResponse};

const PAGE_RECORDS: usize = 512;
const PAGE_BYTES: u64 = 256 * 1024;

enum Command {
    Append(LogRecord),
    Read(ReadLogsRequest, oneshot::Sender<ReadLogsResponse>),
}

/// One writer owns the file and serializes reads with appends, including buffered records.
/// Only bounded pages cross ROS-Z; the publisher never holds the complete log in memory.
pub struct Archive {
    commands: Option<mpsc::Sender<Command>>,
    writer: Option<JoinHandle<Result<()>>>,
    server: JoinHandle<()>,
}
impl Archive {
    /// `file` must be a newly created, readable and writable log file.
    /// `service` is resolved relative to the robot node's namespace.
    pub async fn start(node: &Node, file: File, service: &str) -> Result<Self> {
        let mut server = node.service_server::<ReadLogs>(service).build().await?;
        let (commands, mut receiver) = mpsc::channel(4096);
        let reader = commands.clone();
        let writer = tokio::task::spawn_blocking(move || {
            let mut store = Store {
                file: BufWriter::new(file),
                session: uuid::Uuid::new_v4().to_string(),
            };
            while let Some(command) = receiver.blocking_recv() {
                match command {
                    Command::Append(record) => {
                        serde_json::to_writer(&mut store.file, &record)?;
                        store.file.write_all(b"\n")?;
                    }
                    Command::Read(request, reply) => {
                        let _ = reply.send(store.page(&request));
                    }
                }
            }
            store.file.flush()?;
            Ok(())
        });
        let server = tokio::spawn(
            async move {
                loop {
                    let Ok(request) = server.take_request_async().await else {
                        continue;
                    };
                    let (message, reply) = request.into_parts();
                    let (result, response) = oneshot::channel();
                    if reader.send(Command::Read(message, result)).await.is_err() {
                        break;
                    }
                    let Ok(response) = response.await else {
                        break;
                    };
                    // A viewer may switch namespaces while a page is being read.
                    let _ = reply.reply_async(&response).await;
                }
            }
            .with_subscriber(tracing::Dispatch::none()),
        );
        Ok(Self {
            commands: Some(commands),
            writer: Some(writer),
            server,
        })
    }
    pub async fn append(&self, record: LogRecord) -> Result<()> {
        self.commands
            .as_ref()
            .unwrap()
            .send(Command::Append(record))
            .await
            .map_err(|_| color_eyre::eyre::eyre!("Log file writer stopped"))
    }
    pub async fn finish(mut self) -> Result<()> {
        self.server.abort();
        let _ = (&mut self.server).await;
        self.commands.take();
        self.writer.take().unwrap().await?
    }
}
impl Drop for Archive {
    fn drop(&mut self) {
        self.server.abort();
    }
}

struct Store {
    file: BufWriter<File>,
    session: String,
}
impl Store {
    fn page(&mut self, request: &ReadLogsRequest) -> ReadLogsResponse {
        let mut response = ReadLogsResponse {
            session: self.session.clone(),
            next_offset: 0,
            end: 0,
            records: Vec::new(),
            error: String::new(),
        };
        if let Err(error) = self.read(request, &mut response) {
            response.records.clear();
            response.error = error.to_string();
        }
        response
    }
    fn read(&mut self, request: &ReadLogsRequest, response: &mut ReadLogsResponse) -> Result<()> {
        self.file.flush()?;
        let end = self.file.seek(SeekFrom::End(0))?;
        response.end = end;
        let offset = if request.session == self.session {
            request.offset
        } else {
            0
        };
        let result = (|| -> Result<()> {
            ensure!(offset <= end, "Log cursor is beyond the end of the file");
            if offset > 0 {
                self.file.seek(SeekFrom::Start(offset - 1))?;
                let mut previous = [0];
                self.file.get_mut().read_exact(&mut previous)?;
                ensure!(previous == *b"\n", "Log cursor is not at a record boundary");
            }
            self.file.seek(SeekFrom::Start(offset))?;
            let mut reader = BufReader::new(self.file.get_mut());
            response.next_offset = offset;
            let mut line = String::new();
            while response.records.len() < PAGE_RECORDS && response.next_offset < end {
                line.clear();
                let bytes = reader.read_line(&mut line)?;
                ensure!(bytes > 0 && line.ends_with('\n'), "Incomplete log record");
                response.records.push(serde_json::from_str(&line)?);
                response.next_offset += bytes as u64;
                if response.next_offset - offset >= PAGE_BYTES {
                    break;
                }
            }
            Ok(())
        })();
        // Reads must never change where the next record is written, including failed requests.
        self.file.seek(SeekFrom::Start(end))?;
        result
    }
}
