use async_trait::async_trait;
use std::error;

#[async_trait]
pub trait AI: Send + Sync {
    async fn response(
        &self,
        context: String,
        token_max_length: u16,
        temperature: f32,
        top_p: f32,
        stop_sequence: Option<String>,
    ) -> Result<String, Box<dyn error::Error>>;
    fn name(&self) -> String;
}
