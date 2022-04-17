use serde::Deserialize;
#[derive(Debug, Default, Deserialize)]
pub struct Config {
    pub model: String,
    pub token_max_length: u16,
    pub temperature: f32,
    pub top_p: f32,
}
