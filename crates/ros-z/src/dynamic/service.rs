//! Schema discovery and calls for native ROS-Z services without a compile-time message type.
use super::{DynamicCdrCodec, DynamicError, DynamicPayload, Schema};
use crate::{
    TypeInfo,
    attachment::{Attachment, EndpointGlobalId},
    endpoint_builder::EndpointBuilderContext,
    entity::{EndpointEntity, EndpointKind},
    message::{Message, Service},
    node::Node,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use zenoh::Wait;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceSchema {
    #[serde(with = "type_info_json")]
    pub type_info: TypeInfo,
    pub request_name: String,
    pub response_name: String,
    pub request: Schema,
    pub response: Schema,
}
fn discovery_key(entity: &EndpointEntity) -> String {
    format!(
        "@ros-z/service-schema/{}/{}/{}",
        entity.node.z_id, entity.node.id, entity.id
    )
}
pub(crate) async fn advertise<T: Service>(
    context: &EndpointBuilderContext,
    entity: &EndpointEntity,
) -> crate::Result<zenoh::query::Queryable<()>> {
    let schema = ServiceSchema {
        type_info: entity.type_info.clone(),
        request_name: T::Request::type_name(),
        response_name: T::Response::type_name(),
        request: T::Request::schema().into(),
        response: T::Response::schema().into(),
    };
    let bytes = serde_json::to_vec(&schema)
        .map_err(|e| crate::Error::zenoh("serialize service schema", Box::new(e)))?;
    context
        .session
        .declare_queryable(discovery_key(entity))
        .callback(move |query| {
            if let Err(error) = query.reply(query.key_expr(), bytes.clone()).wait() {
                tracing::warn!(%error,"Service schema reply failed");
            }
        })
        .await
        .map_err(|e| crate::Error::zenoh("advertise service schema", e))
}
impl Node {
    /// Resolve a native service's request and response schemas. Conflicting service types fail.
    pub async fn discover_service_schema(
        &self,
        name: &str,
        timeout: Duration,
    ) -> Result<ServiceSchema, DynamicError> {
        let context = self.endpoint_builder_context();
        let topic = crate::topic_name::qualify_service_name(
            name,
            &context.node.namespace,
            &context.node.name,
        )
        .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let endpoints: Vec<_> = self
                .graph()
                .lock()
                .services_named(&topic)
                .cloned()
                .collect();
            if endpoints.len() > 1 {
                return Err(DynamicError::SerializationError(
                    "Dynamic service calls require one unambiguous server endpoint".into(),
                ));
            }
            if let Some(first) = endpoints.first() {
                if endpoints.iter().any(|e| e.type_info != first.type_info) {
                    return Err(DynamicError::SerializationError(
                        "Conflicting service types".into(),
                    ));
                }
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                let replies = context
                    .session
                    .get(discovery_key(first))
                    .timeout(remaining)
                    .await
                    .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
                if let Ok(reply) = replies.recv_async().await
                    && let Ok(sample) = reply.into_result()
                {
                    if sample.payload().len() > 4 * 1024 * 1024 {
                        return Err(DynamicError::SerializationError(
                            "Service schema exceeds size limit".into(),
                        ));
                    }
                    let schema: ServiceSchema =
                        serde_json::from_slice(&sample.payload().to_bytes())
                            .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
                    if schema.type_info != first.type_info {
                        return Err(DynamicError::SerializationError(
                            "Service schema identity changed".into(),
                        ));
                    }
                    schema
                        .request
                        .validate()
                        .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
                    schema
                        .response
                        .validate()
                        .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
                    return Ok(schema);
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(DynamicError::SerializationError(format!(
                    "Service schema unavailable: {topic}"
                )));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    /// Build a reusable dynamic client. Callers must obtain `schema` from discovery.
    pub async fn dynamic_service_client(
        &self,
        name: &str,
        schema: ServiceSchema,
    ) -> Result<DynamicServiceClient, DynamicError> {
        let context = self.endpoint_builder_context();
        let topic = crate::topic_name::qualify_service_name(
            name,
            &context.node.namespace,
            &context.node.name,
        )
        .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        let entity = context.endpoint_entity(
            EndpointKind::Client,
            topic.clone(),
            schema.type_info.clone(),
            crate::qos::QosProfile::default().to_protocol_qos(),
        );
        let key = ros_z_protocol::format::topic_key_expr(&entity)
            .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        let token = context
            .session
            .liveliness()
            .declare_token(
                entity
                    .liveliness_key_expr()
                    .map_err(|e| DynamicError::SerializationError(e.to_string()))?
                    .0,
            )
            .await
            .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        Ok(DynamicServiceClient {
            context,
            key: key.to_string(),
            topic,
            identity: EndpointGlobalId::from(&entity),
            sequence: AtomicUsize::new(1),
            schema,
            _token: token,
        })
    }
}
pub struct DynamicServiceClient {
    context: EndpointBuilderContext,
    key: String,
    topic: String,
    identity: EndpointGlobalId,
    sequence: AtomicUsize,
    pub schema: ServiceSchema,
    _token: zenoh::liveliness::LivelinessToken,
}
impl DynamicServiceClient {
    /// A timeout after dispatch does not imply that the service had no effect.
    pub async fn call(
        &self,
        request: &DynamicPayload,
        timeout: Duration,
    ) -> Result<DynamicPayload, DynamicError> {
        if request.schema != self.schema.request {
            return Err(DynamicError::SerializationError(
                "Request schema differs from service".into(),
            ));
        }
        if self
            .context
            .graph
            .lock()
            .services_named(&self.topic)
            .count()
            != 1
        {
            return Err(DynamicError::SerializationError(
                "Service must have exactly one discovered server before dispatch".into(),
            ));
        }
        let bytes = DynamicCdrCodec::try_serialize_payload(request)?;
        let attachment = Attachment::with_clock(
            self.sequence.fetch_add(1, Ordering::AcqRel) as i64,
            self.identity,
            &self.context.clock,
        );
        let replies = self
            .context
            .session
            .get(self.key.clone())
            .payload(bytes)
            .attachment(attachment)
            .target(zenoh::query::QueryTarget::BestMatching)
            .consolidation(zenoh::query::ConsolidationMode::None)
            .timeout(timeout)
            .await
            .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        let reply = tokio::time::timeout(timeout, replies.recv_async())
            .await
            .map_err(|_| {
                DynamicError::SerializationError(
                    "Service timed out after dispatch; effect unknown".into(),
                )
            })?
            .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        let sample = reply
            .into_result()
            .map_err(|e| DynamicError::SerializationError(e.to_string()))?;
        DynamicCdrCodec::decode(&sample.payload().to_bytes(), &self.schema.response)
    }
}

mod type_info_json {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &TypeInfo,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        (value.name.as_str(), value.hash.to_hash_string()).serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<TypeInfo, D::Error> {
        let (name, hash): (String, String) = Deserialize::deserialize(deserializer)?;
        Ok(TypeInfo::new(
            name,
            crate::SchemaHash::from_hash_string(&hash).map_err(serde::de::Error::custom)?,
        ))
    }
}
