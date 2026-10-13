//! The robot log stream and disk-backed, paged history shared by simulator tools and Twix.
mod archive;
pub use archive::Archive;

use ros_z::{Message, ServiceTypeInfo, entity::TypeInfo, message::Service};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Message)]
#[message(name = "hulk::LogRecord")]
pub struct LogRecord {
    pub sequence: u64,
    pub simulation_time: i64,
    pub wall_time: i64,
    pub level: u8,
    pub target: String,
    pub message: String,
    pub fields: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Message)]
#[message(name = "hulk::ReadLogsRequest")]
pub struct ReadLogsRequest {
    /// Empty for the first request. A changed session restarts reading from the beginning.
    pub session: String,
    /// An opaque file cursor returned by the previous page, or zero for the first page.
    pub offset: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Message)]
#[message(name = "hulk::ReadLogsResponse")]
pub struct ReadLogsResponse {
    pub session: String,
    pub next_offset: u64,
    pub end: u64,
    pub records: Vec<LogRecord>,
    pub error: String,
}

pub struct ReadLogs;
impl Service for ReadLogs {
    type Request = ReadLogsRequest;
    type Response = ReadLogsResponse;
}
impl ServiceTypeInfo for ReadLogs {
    fn service_type_info() -> TypeInfo {
        let descriptor = ros_z_schema::ServiceDef::new(
            "hulk::ReadLogs",
            ReadLogsRequest::type_name(),
            ReadLogsResponse::type_name(),
        )
        .expect("static log history service descriptor");
        TypeInfo::new(
            descriptor.type_name.as_str(),
            ros_z_schema::compute_hash(&descriptor).expect("static log history service hash"),
        )
    }
}

#[cfg(test)]
mod tests;
