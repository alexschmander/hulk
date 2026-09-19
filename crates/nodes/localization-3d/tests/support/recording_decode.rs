use std::any::type_name;

use color_eyre::{Result, eyre::WrapErr as _};
use ros_z::{SerdeCdrCodec, message::WireDecoder};
use serde::de::DeserializeOwned;

pub fn decode_recorded_message<T>(message: &mcap::Message<'_>) -> Result<T>
where
    T: DeserializeOwned,
{
    SerdeCdrCodec::<T>::deserialize(message.data.as_ref()).wrap_err_with(|| {
        format!(
            "failed to decode {} from topic '{}' at log_time={} publish_time={}",
            type_name::<T>(),
            message.channel.topic,
            message.log_time,
            message.publish_time
        )
    })
}

#[test]
fn decode_error_preserves_recording_metadata_and_source() {
    let message = mcap::Message {
        channel: std::sync::Arc::new(mcap::Channel {
            id: 1,
            topic: "camera_matrix".into(),
            schema: None,
            message_encoding: "cdr".into(),
            metadata: Default::default(),
        }),
        sequence: 0,
        log_time: 123,
        publish_time: 456,
        data: (&[][..]).into(),
    };
    let error = decode_recorded_message::<u64>(&message).unwrap_err();
    assert_eq!(
        error.to_string(),
        "failed to decode u64 from topic 'camera_matrix' at log_time=123 publish_time=456"
    );
    assert!(
        error
            .chain()
            .any(|source| source.is::<ros_z::message::CdrError>())
    );
}
