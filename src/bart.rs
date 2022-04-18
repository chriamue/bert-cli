use async_trait::async_trait;
use rust_bert::bart::{
    BartConfigResources, BartGenerator, BartMergesResources, BartModelResources, BartVocabResources,
};
use rust_bert::pipelines::generation_utils::{GenerateConfig, GenerateOptions, LanguageGenerator};
use rust_bert::resources::RemoteResource;
use std::error;
use tch::Device;

use crate::ai::AI;

pub struct Bart {
    model: BartGenerator,
}

impl Bart {
    pub fn new(token_max_length: u16, temperature: f32, top_p: f32) -> Self {
        let config_resource = Box::new(RemoteResource::from_pretrained(
            BartConfigResources::BART_CNN,
        ));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(
            BartVocabResources::BART_CNN,
        ));
        let merges_resource = Box::new(RemoteResource::from_pretrained(
            BartMergesResources::BART_CNN,
        ));
        let model_resource = Box::new(RemoteResource::from_pretrained(
            BartModelResources::BART_CNN,
        ));
        let device = Device::cuda_if_available();
        let generate_config = GenerateConfig {
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            min_length: 10,
            max_length: token_max_length.into(),
            do_sample: true,
            early_stopping: false,
            repetition_penalty: 1.0,
            top_p: top_p.into(),
            top_k: 55,
            temperature: temperature.into(),
            device,
            ..Default::default()
        };
        let model = std::thread::spawn(move || BartGenerator::new(generate_config).unwrap())
            .join()
            .expect("Thread panicked");

        Bart { model }
    }
}

unsafe impl Send for Bart {}

unsafe impl Sync for Bart {}

#[async_trait]
impl AI for Bart {
    async fn response(
        &self,
        context: String,
        token_max_length: u16,
        temperature: f32,
        top_p: f32,
        _stop_sequence: Option<String>,
    ) -> Result<String, Box<dyn error::Error>> {
        let generate_options = GenerateOptions {
            max_length: Some(token_max_length.into()),
            do_sample: Some(true),
            early_stopping: Some(false),
            repetition_penalty: Some(1.1),
            temperature: Some(temperature.into()),
            top_p: Some(top_p.into()),
            top_k: Some(10),
            ..Default::default()
        };

        let output = self
            .model
            .generate(Some(&[context.to_string()]), Some(generate_options));
        let response = output[0].text.to_string();
        let response: String = response.replace(context.as_str(), "");
        Ok(response)
    }

    fn name(&self) -> String {
        "bart".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response() {
        let ai = Bart::new(42, 1.1, 0.9);
        let context = "Lots of Tesla cars to deliver before year end! Your support in taking delivery is much appreciated.".to_string();
        let output = ai
            .response(context.to_string(), 42, 1.1, 0.9, None)
            .await
            .unwrap();
        println!("{}", output);
        assert_ne!(output, context);
        assert_ne!(output.len(), 0);
        assert!(output.len() > 10);
    }
}
