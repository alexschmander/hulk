//! Field geometry comes from the same location layers used on robots.
use color_eyre::{
    Result,
    eyre::{Context, ensure},
};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path};
use types::field_dimensions::FieldDimensions;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldConfiguration {
    pub dimensions: FieldDimensions,
    pub goal_height: f32,
}
impl FieldConfiguration {
    pub fn locations(root: &Path) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(root.join("location"))? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                names.push(
                    entry
                        .file_name()
                        .into_string()
                        .map_err(|_| color_eyre::eyre::eyre!("Non UTF-8 location name"))?,
                );
            }
        }
        names.sort();
        Ok(names)
    }
    pub fn load(root: &Path, location: &str) -> Result<Self> {
        ensure!(
            Path::new(location).components().count() == 1
                && matches!(
                    Path::new(location).components().next(),
                    Some(Component::Normal(_))
                ),
            "Invalid location name"
        );
        let location = root.join("location").join(location);
        ensure!(
            location.is_dir(),
            "Location does not exist: {}",
            location.display()
        );
        let mut fields = serde_json::Map::new();
        let mut height = serde_json::Value::Null;
        for layer in [
            root.join("base"),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("parameters"),
            location,
        ] {
            let global = layer.join("global.json5");
            if global.exists() {
                let value: serde_json::Value = json5::from_str(&std::fs::read_to_string(&global)?)
                    .wrap_err_with(|| format!("Invalid parameters in {}", global.display()))?;
                if let Some(values) = value.get("field_dimensions").and_then(|v| v.as_object()) {
                    fields.extend(values.clone());
                }
            }
            let simulator = layer.join("simulator.json5");
            if simulator.exists() {
                let value: serde_json::Value =
                    json5::from_str(&std::fs::read_to_string(&simulator)?)?;
                if let Some(value) = value.get("goal_height") {
                    height = value.clone();
                }
            }
        }
        let configuration = Self {
            dimensions: serde_json::from_value(fields.into())?,
            goal_height: serde_json::from_value(height)?,
        };
        configuration
            .validate()
            .map_err(|error| color_eyre::eyre::eyre!(error))?;
        Ok(configuration)
    }
    pub fn validate(&self) -> Result<(), String> {
        crate::parameters::validate_field_dimensions(&self.dimensions)?;
        if !self.goal_height.is_finite() || self.goal_height <= 0.0 {
            return Err("Goal height must be finite and positive".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discovers_and_loads_every_location_including_hsl_fields() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters");
        let locations = FieldConfiguration::locations(&root).unwrap();
        for name in [
            "hsl_small",
            "hsl_middle",
            "hsl_large",
            "incheon_small",
            "incheon_big",
        ] {
            assert!(locations.contains(&name.to_owned()));
        }
        for name in locations {
            FieldConfiguration::load(&root, &name).unwrap();
        }
        for (name, length, width, height) in [
            ("hsl_small", 9., 6., 1.2),
            ("hsl_middle", 14., 9., 1.8),
            ("hsl_large", 22., 14., 2.),
        ] {
            let field = FieldConfiguration::load(&root, name).unwrap();
            assert_eq!(
                (
                    field.dimensions.length,
                    field.dimensions.width,
                    field.goal_height
                ),
                (length, width, height)
            );
        }
    }
    #[test]
    fn a_new_partial_location_inherits_base_dimensions() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("base")).unwrap();
        std::fs::create_dir_all(root.path().join("location/custom")).unwrap();
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters/base/global.json5"),
            root.path().join("base/global.json5"),
        )
        .unwrap();
        std::fs::write(
            root.path().join("location/custom/global.json5"),
            "{field_dimensions: {length: 11.0}}",
        )
        .unwrap();
        std::fs::write(
            root.path().join("location/custom/simulator.json5"),
            "{goal_height: 1.9}",
        )
        .unwrap();
        assert_eq!(
            FieldConfiguration::locations(root.path()).unwrap(),
            ["custom"]
        );
        let field = FieldConfiguration::load(root.path(), "custom").unwrap();
        assert_eq!(field.dimensions.length, 11.);
        assert_eq!(field.dimensions.width, 5.95);
        assert_eq!(field.goal_height, 1.9);
        assert!(FieldConfiguration::load(root.path(), "../base").is_err());
    }
}
