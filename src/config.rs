use serde::Deserialize;
#[derive(Debug, Default, Deserialize)]
pub struct Config {
    pub model: String,
}
