use serde::{Deserialize, Serialize};
use std::error;

mod ai;
mod gpt2;
mod gptneo;

pub fn create_ai(ai: String) -> Box<dyn ai::AI> {
    match ai.as_str() {
        "gpt2" => Box::new(gpt2::GPT2::new()),
        "gptneo" => Box::new(gptneo::GPTNeo::new()),
        _ => Box::new(gptneo::GPTNeo::new()),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClassifyResponse {
    pub sequence: String,
    pub labels: Vec<String>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub text: String,
    pub prompt: String,
}

pub struct Bert {
    pub ai: Box<dyn ai::AI>,
}

impl Bert {
    pub async fn generate(
        &self,
        context: String,
        token_max_length: u16,
        _temperature: f32,
        _top_p: f32,
        _stop_sequence: Option<String>,
    ) -> Result<GenerateResponse, Box<dyn error::Error>> {

        let result = self.ai.response(context.to_string(), token_max_length).await.unwrap();

        let gr = GenerateResponse {
            model: "".to_string(),
            text: result,
            prompt: context,
        };
        Ok(gr)
    }
}

impl Default for Bert {
    fn default() -> Self {
        let ai = create_ai("".to_string());
        Bert { ai }
    }
}