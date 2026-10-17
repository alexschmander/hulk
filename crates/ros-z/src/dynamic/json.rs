use serde_json::{Map, Number, Value};

use super::{DynamicPayload, DynamicStruct, DynamicValue, EnumPayloadValue, EnumValue};

/// Rendering options for dynamic payload JSON values.
///
/// Primitive values render as their matching JSON scalar. Structs render as JSON
/// objects, sequences as arrays, maps as arrays of `{ "key", "value" }` entries,
/// optional `None` as `null`, and enums as objects with `variant_index`,
/// `variant_name`, and `payload` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicJsonRenderPolicy {
    /// Tag present optional values to distinguish Some(None) and Some(()) from None.
    pub tag_optionals: bool,
    /// Render i64 and u64 as decimal strings to preserve precision in JSON clients.
    pub wide_integers_as_strings: bool,
    /// Rendering policy for `DynamicValue::Bytes`.
    pub bytes: ByteRenderPolicy,
    /// Rendering policy for `NaN` and infinite floating point values.
    pub non_finite_float: NonFiniteFloatRenderPolicy,
}

impl Default for DynamicJsonRenderPolicy {
    fn default() -> Self {
        Self {
            tag_optionals: false,
            wide_integers_as_strings: false,
            bytes: ByteRenderPolicy::Compact { preview_len: 32 },
            non_finite_float: NonFiniteFloatRenderPolicy::Object,
        }
    }
}

/// JSON representation for dynamic byte arrays.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteRenderPolicy {
    /// Render bytes as `{ "$type": "bytes", "len", "preview", "truncated" }`.
    Compact { preview_len: usize },
    /// Render bytes as a JSON array of byte values.
    FullArray,
}

/// JSON representation for non-finite floating point values.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NonFiniteFloatRenderPolicy {
    /// Render non-finite floats as JSON `null`.
    Null,
    /// Render non-finite floats as `{ "$type": "non_finite_float", "value" }`.
    Object,
}

/// Render a dynamic payload value as JSON according to `policy`.
pub fn dynamic_payload_to_json(payload: &DynamicPayload, policy: DynamicJsonRenderPolicy) -> Value {
    dynamic_value_to_json(&payload.value, policy)
}

/// Render a dynamic value as JSON according to `policy`.
pub fn dynamic_value_to_json(value: &DynamicValue, policy: DynamicJsonRenderPolicy) -> Value {
    match value {
        DynamicValue::Bool(value) => Value::Bool(*value),
        DynamicValue::Unit => Value::Null,
        DynamicValue::Int8(value) => Value::Number((*value).into()),
        DynamicValue::Int16(value) => Value::Number((*value).into()),
        DynamicValue::Int32(value) => Value::Number((*value).into()),
        DynamicValue::Int64(value) if policy.wide_integers_as_strings => {
            Value::String(value.to_string())
        }
        DynamicValue::Uint64(value) if policy.wide_integers_as_strings => {
            Value::String(value.to_string())
        }
        DynamicValue::Int64(value) => Value::Number((*value).into()),
        DynamicValue::Uint8(value) => Value::Number((*value).into()),
        DynamicValue::Uint16(value) => Value::Number((*value).into()),
        DynamicValue::Uint32(value) => Value::Number((*value).into()),
        DynamicValue::Uint64(value) => Value::Number((*value).into()),
        DynamicValue::Float32(value) => float_to_json(f64::from(*value), policy),
        DynamicValue::Float64(value) => float_to_json(*value, policy),
        DynamicValue::String(value) => Value::String(value.clone()),
        DynamicValue::Bytes(bytes) => bytes_to_json(bytes, policy.bytes),
        DynamicValue::Struct(value) => struct_to_json(value, policy),
        DynamicValue::Optional(None) => Value::Null,
        DynamicValue::Optional(Some(value)) if policy.tag_optionals => {
            serde_json::json!({"$type":"some","value":dynamic_value_to_json(value,policy)})
        }
        DynamicValue::Optional(Some(value)) => dynamic_value_to_json(value, policy),
        DynamicValue::Enum(value) => enum_to_json(value, policy),
        DynamicValue::Sequence(values) => Value::Array(
            values
                .iter()
                .map(|value| dynamic_value_to_json(value, policy))
                .collect(),
        ),
        DynamicValue::Map(entries) => Value::Array(
            entries
                .iter()
                .map(|(key, value)| {
                    let mut entry = Map::new();
                    entry.insert("key".to_string(), dynamic_value_to_json(key, policy));
                    entry.insert("value".to_string(), dynamic_value_to_json(value, policy));
                    Value::Object(entry)
                })
                .collect(),
        ),
    }
}

fn float_to_json(value: f64, policy: DynamicJsonRenderPolicy) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or_else(|| non_finite_float_to_json(value, policy.non_finite_float))
}

fn non_finite_float_to_json(value: f64, policy: NonFiniteFloatRenderPolicy) -> Value {
    match policy {
        NonFiniteFloatRenderPolicy::Null => Value::Null,
        NonFiniteFloatRenderPolicy::Object => {
            let mut fields = Map::new();
            fields.insert(
                "$type".to_string(),
                Value::String("non_finite_float".to_string()),
            );
            fields.insert(
                "value".to_string(),
                Value::String(non_finite_float_label(value).to_string()),
            );
            Value::Object(fields)
        }
    }
}

fn non_finite_float_label(value: f64) -> &'static str {
    if value.is_nan() {
        "NaN"
    } else if value.is_sign_negative() {
        "-Infinity"
    } else {
        "Infinity"
    }
}

fn bytes_to_json(bytes: &[u8], policy: ByteRenderPolicy) -> Value {
    match policy {
        ByteRenderPolicy::Compact { preview_len } => {
            let mut fields = Map::new();
            fields.insert("$type".to_string(), Value::String("bytes".to_string()));
            fields.insert("len".to_string(), Value::from(bytes.len()));
            fields.insert(
                "preview".to_string(),
                Value::Array(
                    bytes
                        .iter()
                        .take(preview_len)
                        .map(|byte| Value::from(*byte))
                        .collect(),
                ),
            );
            fields.insert(
                "truncated".to_string(),
                Value::Bool(bytes.len() > preview_len),
            );
            Value::Object(fields)
        }
        ByteRenderPolicy::FullArray => {
            Value::Array(bytes.iter().map(|byte| Value::from(*byte)).collect())
        }
    }
}

fn struct_to_json(value: &DynamicStruct, policy: DynamicJsonRenderPolicy) -> Value {
    Value::Object(
        value
            .iter()
            .map(|(name, value)| (name.to_string(), dynamic_value_to_json(value, policy)))
            .collect(),
    )
}

fn enum_to_json(value: &EnumValue, policy: DynamicJsonRenderPolicy) -> Value {
    let mut fields = Map::new();
    fields.insert(
        "variant_index".to_string(),
        Value::Number(value.variant_index.into()),
    );
    fields.insert(
        "variant_name".to_string(),
        Value::String(value.variant_name.clone()),
    );
    fields.insert(
        "payload".to_string(),
        enum_payload_to_json(&value.payload, policy),
    );
    Value::Object(fields)
}

pub(crate) fn enum_payload_to_json(
    payload: &EnumPayloadValue,
    policy: DynamicJsonRenderPolicy,
) -> Value {
    match payload {
        EnumPayloadValue::Unit => Value::Null,
        EnumPayloadValue::Newtype(value) => dynamic_value_to_json(value, policy),
        EnumPayloadValue::Tuple(values) => Value::Array(
            values
                .iter()
                .map(|value| dynamic_value_to_json(value, policy))
                .collect(),
        ),
        EnumPayloadValue::Struct(fields) => Value::Object(
            fields
                .iter()
                .map(|field| {
                    (
                        field.name.clone(),
                        dynamic_value_to_json(&field.value, policy),
                    )
                })
                .collect(),
        ),
    }
}

/// Decode the lossless inspector representation against the discovered schema.
/// Unknown fields, missing fields, narrowing integers and invalid enum tags fail before publication.
pub fn dynamic_payload_from_json(
    schema: super::Schema,
    value: &Value,
) -> Result<DynamicPayload, super::DynamicError> {
    fn error(message: impl Into<String>) -> super::DynamicError {
        super::DynamicError::SerializationError(message.into())
    }
    fn parse(
        schema: &super::Schema,
        shape: &ros_z_schema::TypeDef,
        value: &Value,
        depth: usize,
    ) -> Result<DynamicValue, super::DynamicError> {
        use DynamicValue as D;
        use ros_z_schema::{
            EnumPayloadDef as E, PrimitiveTypeDef as P, TypeDef as T, TypeDefinition as Def,
        };
        if depth > 128 {
            return Err(error("JSON nesting exceeds 128"));
        }
        let next = |shape: &T, value: &Value| parse(schema, shape, value, depth + 1);
        let signed = || {
            value
                .as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .or_else(|| value.as_i64())
                .ok_or_else(|| error("Expected signed integer"))
        };
        let unsigned = || {
            value
                .as_str()
                .and_then(|s| s.parse::<u64>().ok())
                .or_else(|| value.as_u64())
                .ok_or_else(|| error("Expected unsigned integer"))
        };
        let float = || -> Result<f64, super::DynamicError> {
            if let Some(v) = value.as_f64() {
                return Ok(v);
            }
            if value["$type"] == "non_finite_float" {
                match value["value"].as_str() {
                    Some("NaN") => return Ok(f64::NAN),
                    Some("Infinity") => return Ok(f64::INFINITY),
                    Some("-Infinity") => return Ok(f64::NEG_INFINITY),
                    _ => {}
                }
            }
            Err(error("Expected floating-point number"))
        };
        let fields = |fields: &[ros_z_schema::FieldDef],
                      value: &Value|
         -> Result<Vec<DynamicValue>, super::DynamicError> {
            let object = value.as_object().ok_or_else(|| error("Expected object"))?;
            if object.len() != fields.len() {
                return Err(error("Object fields differ from schema"));
            }
            fields
                .iter()
                .map(|field| {
                    next(
                        &field.shape,
                        object
                            .get(&field.name)
                            .ok_or_else(|| error(format!("Missing field {}", field.name)))?,
                    )
                })
                .collect()
        };
        Ok(match shape {
            T::Primitive(p) => match p {
                P::Unit => {
                    if !value.is_null() {
                        return Err(error("Expected null"));
                    }
                    D::Unit
                }
                P::Bool => D::Bool(value.as_bool().ok_or_else(|| error("Expected boolean"))?),
                P::I8 => D::Int8(signed()?.try_into().map_err(|_| error("i8 out of range"))?),
                P::I16 => D::Int16(
                    signed()?
                        .try_into()
                        .map_err(|_| error("i16 out of range"))?,
                ),
                P::I32 => D::Int32(
                    signed()?
                        .try_into()
                        .map_err(|_| error("i32 out of range"))?,
                ),
                P::I64 => D::Int64(signed()?),
                P::U64 => D::Uint64(unsigned()?),
                P::U8 => D::Uint8(
                    unsigned()?
                        .try_into()
                        .map_err(|_| error("u8 out of range"))?,
                ),
                P::U16 => D::Uint16(
                    unsigned()?
                        .try_into()
                        .map_err(|_| error("u16 out of range"))?,
                ),
                P::U32 => D::Uint32(
                    unsigned()?
                        .try_into()
                        .map_err(|_| error("u32 out of range"))?,
                ),
                P::F32 => {
                    let v = float()?;
                    if v.is_finite() && !(v as f32).is_finite() {
                        return Err(error("f32 out of range"));
                    }
                    D::Float32(v as f32)
                }
                P::F64 => D::Float64(float()?),
            },
            T::String => D::String(
                value
                    .as_str()
                    .ok_or_else(|| error("Expected string"))?
                    .into(),
            ),
            T::Optional(t) if value["$type"] == "some" => {
                let object = value.as_object().unwrap();
                if object.len() != 2 || !object.contains_key("value") {
                    return Err(error("Present optional requires exactly $type and value"));
                }
                D::Optional(Some(Box::new(next(t, &value["value"])?)))
            }
            T::Optional(t) => D::Optional(if value.is_null() {
                None
            } else {
                Some(Box::new(next(t, value)?))
            }),
            T::Sequence { element, .. } => D::Sequence(
                value
                    .as_array()
                    .ok_or_else(|| error("Expected array"))?
                    .iter()
                    .map(|v| next(element, v))
                    .collect::<Result<_, _>>()?,
            ),
            T::Map { key, value: shape } => D::Map(
                value
                    .as_array()
                    .ok_or_else(|| error("Expected map entry array"))?
                    .iter()
                    .map(|v| {
                        let object = v
                            .as_object()
                            .ok_or_else(|| error("Expected map entry object"))?;
                        if object.len() != 2
                            || !object.contains_key("key")
                            || !object.contains_key("value")
                        {
                            return Err(error("Map entry requires exactly key and value"));
                        }
                        Ok((next(key, &v["key"])?, next(shape, &v["value"])?))
                    })
                    .collect::<Result<_, super::DynamicError>>()?,
            ),
            T::Named(name) => match schema
                .definitions
                .get(name)
                .ok_or_else(|| error("Missing named definition"))?
            {
                Def::Struct(def) => D::Struct(Box::new(DynamicStruct::new(
                    schema.clone(),
                    name.clone(),
                    fields(&def.fields, value)?,
                )?)),
                Def::Enum(def) => {
                    let object = value
                        .as_object()
                        .ok_or_else(|| error("Expected enum object"))?;
                    if object.len() != 3 || !object.contains_key("payload") {
                        return Err(error(
                            "Enum requires variant_index, variant_name and payload",
                        ));
                    }
                    let index = value["variant_index"]
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| error("Missing enum variant index"))?;
                    let variant = def
                        .variants
                        .get(index)
                        .ok_or_else(|| error("Unknown enum variant"))?;
                    if value["variant_name"].as_str() != Some(&variant.name) {
                        return Err(error("Enum variant name and index differ"));
                    }
                    let payload = &value["payload"];
                    let payload = match &variant.payload {
                        E::Unit => {
                            if !payload.is_null() {
                                return Err(error("Unit variant payload must be null"));
                            }
                            EnumPayloadValue::Unit
                        }
                        E::Newtype(t) => EnumPayloadValue::Newtype(Box::new(next(t, payload)?)),
                        E::Tuple(types) => {
                            let values = payload
                                .as_array()
                                .ok_or_else(|| error("Expected tuple array"))?;
                            if values.len() != types.len() {
                                return Err(error("Tuple arity differs"));
                            }
                            EnumPayloadValue::Tuple(
                                types
                                    .iter()
                                    .zip(values)
                                    .map(|(t, v)| next(t, v))
                                    .collect::<Result<_, _>>()?,
                            )
                        }
                        E::Struct(def) => EnumPayloadValue::Struct(
                            def.iter()
                                .zip(fields(def, payload)?)
                                .map(|(f, value)| super::DynamicNamedValue {
                                    name: f.name.clone(),
                                    value,
                                })
                                .collect(),
                        ),
                    };
                    D::Enum(EnumValue::new(index as u32, &variant.name, payload))
                }
            },
        })
    }
    let dynamic = parse(&schema, &schema.root, value, 0)?;
    DynamicPayload::new(schema, dynamic)
}

#[cfg(test)]
mod input_tests {
    use super::*;
    use crate::Message;
    use std::collections::BTreeMap;
    #[derive(serde::Serialize, serde::Deserialize, crate::Message)]
    enum Choice {
        Empty,
        Number(u64),
        Named { active: bool, note: Option<String> },
    }
    #[derive(serde::Serialize, serde::Deserialize, crate::Message)]
    struct Fixture {
        signed: i64,
        unsigned: u64,
        choice: Choice,
        map: BTreeMap<String, Option<u64>>,
        bytes: Vec<u8>,
        float: f64,
    }
    #[test]
    fn tagged_optionals_preserve_nested_presence() {
        let schema: super::super::Schema = Option::<Option<()>>::schema().into();
        let policy = DynamicJsonRenderPolicy {
            tag_optionals: true,
            ..Default::default()
        };
        for value in [
            serde_json::json!(null),
            serde_json::json!({"$type":"some","value":null}),
            serde_json::json!({"$type":"some","value":{"$type":"some","value":null}}),
        ] {
            let payload = dynamic_payload_from_json(schema.clone(), &value).unwrap();
            let bytes = super::super::DynamicCdrCodec::try_serialize_payload(&payload).unwrap();
            let decoded = super::super::DynamicCdrCodec::decode(&bytes, &schema).unwrap();
            assert_eq!(dynamic_payload_to_json(&decoded, policy), value);
        }
    }
    #[test]
    fn canonical_json_round_trips_wide_integers_enums_maps_and_nonfinite_values() {
        let policy = DynamicJsonRenderPolicy {
            wide_integers_as_strings: true,
            bytes: ByteRenderPolicy::FullArray,
            ..Default::default()
        };
        for choice in [
            serde_json::json!({"variant_index":0,"variant_name":"Empty","payload":null}),
            serde_json::json!({"variant_index":1,"variant_name":"Number","payload":"18446744073709551615"}),
            serde_json::json!({"variant_index":2,"variant_name":"Named","payload":{"active":true,"note":null}}),
        ] {
            let json = serde_json::json!({"signed":"-9223372036854775808","unsigned":"18446744073709551615","choice":choice,"map":[{"key":"test","value":null}],"bytes":[0,255],"float":{"$type":"non_finite_float","value":"Infinity"}});
            let payload = dynamic_payload_from_json(Fixture::schema().into(), &json).unwrap();
            let cdr = super::super::DynamicCdrCodec::try_serialize_payload(&payload).unwrap();
            let decoded = super::super::DynamicCdrCodec::decode(&cdr, &payload.schema).unwrap();
            assert_eq!(dynamic_payload_to_json(&decoded, policy), json);
            assert_eq!(
                ".unsigned"
                    .parse::<super::super::ValuePath>()
                    .unwrap()
                    .select(&decoded)
                    .unwrap()
                    .to_json(policy),
                json["unsigned"]
            );
        }
    }
    #[test]
    fn invalid_values_are_rejected_before_encoding() {
        assert!(dynamic_payload_from_json(u8::schema().into(), &serde_json::json!(256)).is_err());
        assert!(
            dynamic_payload_from_json(
                Choice::schema().into(),
                &serde_json::json!({"variant_index":0,"variant_name":"Empty"})
            )
            .is_err()
        );
        assert!(
            dynamic_payload_from_json(
                BTreeMap::<String, Option<u8>>::schema().into(),
                &serde_json::json!([{"key":"x"}])
            )
            .is_err()
        );
        assert!(
            dynamic_payload_from_json(
                Choice::schema().into(),
                &serde_json::json!({"variant_index":0,"variant_name":"Number","payload":null})
            )
            .is_err()
        );
    }
}
