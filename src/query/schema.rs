// src/query/schema.rs
//
// Schema definition and validation for records.
//
// Validated: Cell 17 Gap 8
//   Valid record passes schema
//   Missing required field caught
//   Type error caught
//   Defaults applied correctly
//
// Each collection in UlmenDB can have an optional schema.
// Records are validated on insert/update. Invalid records are rejected.

use std::collections::HashMap;

/// Supported field types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    String,
    Int,
    Float,
    Bool,
    Bytes,
}

/// Schema field definition.
#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,
    pub required: bool,
    pub default: Option<Vec<u8>>,
}

/// A collection schema: ordered list of field definitions.
#[derive(Debug, Clone)]
pub struct Schema {
    pub name: String,
    pub fields: Vec<FieldDef>,
}

/// Validation error for a single field.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

/// A record to validate: field name -> raw bytes.
pub type Record = HashMap<String, Vec<u8>>;

impl Schema {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fields: Vec::new(),
        }
    }

    pub fn field(mut self, def: FieldDef) -> Self {
        self.fields.push(def);
        self
    }

    /// Validate a record against this schema.
    /// Returns a list of errors (empty = valid).
    pub fn validate(&self, record: &Record) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for field in &self.fields {
            match record.get(&field.name) {
                None => {
                    if field.required && field.default.is_none() {
                        errors.push(ValidationError {
                            field: field.name.clone(),
                            message: format!(
                                "field '{}' is required",
                                field.name
                            ),
                        });
                    }
                }
                Some(value) => {
                    if let Err(msg) = Self::check_type(
                        &field.name,
                        &field.field_type,
                        value,
                    ) {
                        errors.push(msg);
                    }
                }
            }
        }

        errors
    }

    /// Apply defaults to a record (fills in missing fields).
    pub fn apply_defaults(&self, record: &mut Record) {
        for field in &self.fields {
            if !record.contains_key(&field.name) {
                if let Some(ref default) = field.default {
                    record.insert(field.name.clone(), default.clone());
                }
            }
        }
    }

    fn check_type(
        name: &str,
        expected: &FieldType,
        value: &[u8],
    ) -> Result<(), ValidationError> {
        let valid = match expected {
            FieldType::String => std::str::from_utf8(value).is_ok(),
            FieldType::Int => {
                std::str::from_utf8(value)
                    .map(|s| s.parse::<i64>().is_ok())
                    .unwrap_or(false)
            }
            FieldType::Float => {
                std::str::from_utf8(value)
                    .map(|s| s.parse::<f64>().is_ok())
                    .unwrap_or(false)
            }
            FieldType::Bool => {
                matches!(value, b"true" | b"false" | b"0" | b"1")
            }
            FieldType::Bytes => true, // any bytes are valid
        };

        if valid {
            Ok(())
        } else {
            Err(ValidationError {
                field: name.to_string(),
                message: format!(
                    "field '{}': expected {:?}, got invalid data",
                    name, expected
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code_schema() -> Schema {
        Schema::new("code_record")
            .field(FieldDef {
                name: "key".into(),
                field_type: FieldType::String,
                required: true,
                default: None,
            })
            .field(FieldDef {
                name: "lang".into(),
                field_type: FieldType::String,
                required: true,
                default: None,
            })
            .field(FieldDef {
                name: "lines".into(),
                field_type: FieldType::Int,
                required: false,
                default: Some(b"0".to_vec()),
            })
    }

    #[test]
    fn valid_record() {
        let schema = code_schema();
        let mut record = Record::new();
        record.insert("key".into(), b"auth::fn".to_vec());
        record.insert("lang".into(), b"python".to_vec());
        let errors = schema.validate(&record);
        assert!(errors.is_empty(), "expected valid: {errors:?}");
    }

    #[test]
    fn missing_required() {
        let schema = code_schema();
        let mut record = Record::new();
        record.insert("key".into(), b"auth::fn".to_vec());
        // missing "lang"
        let errors = schema.validate(&record);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("required"));
    }

    #[test]
    fn type_mismatch() {
        let schema = code_schema();
        let mut record = Record::new();
        record.insert("key".into(), b"auth::fn".to_vec());
        record.insert("lang".into(), b"python".to_vec());
        record.insert("lines".into(), b"not_a_number".to_vec());
        let errors = schema.validate(&record);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("Int"));
    }

    #[test]
    fn defaults_applied() {
        let schema = code_schema();
        let mut record = Record::new();
        record.insert("key".into(), b"auth::fn".to_vec());
        record.insert("lang".into(), b"python".to_vec());
        schema.apply_defaults(&mut record);
        assert_eq!(record.get("lines"), Some(&b"0".to_vec()));
    }
}
