use serde::{Deserialize, Serialize};
use std::error;

mod ai;
mod gpt2;
mod gptneo;
mod gptneo1;
mod gptneo2;
mod bart;

pub fn create_ai(
    ai: String,
    token_max_length: u16,
    temperature: f32,
    top_p: f32,
) -> Box<dyn ai::AI> {
    match ai.as_str() {
        "bart" => Box::new(bart::Bart::new()),
        "gpt2" => Box::new(gpt2::GPT2::new()),
        "gptneo" => Box::new(gptneo::GPTNeo::new()),
        "gptneo1" => Box::new(gptneo1::GPTNeo1::new()),
        "gptneo2" => Box::new(gptneo2::GPTNeo2::new()),
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
        temperature: f32,
        top_p: f32,
        stop_sequence: Option<String>,
    ) -> Result<GenerateResponse, Box<dyn error::Error>> {
        let result = self
            .ai
            .response(
                context.to_string(),
                token_max_length,
                temperature,
                top_p,
                stop_sequence,
            )
            .await
            .unwrap();

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
        let ai = create_ai("".to_string(), 100, 1.1, 0.9);
        Bert { ai }
    }
}
