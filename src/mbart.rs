use async_trait::async_trait;
use rust_bert::mbart::MBartGenerator;
use rust_bert::mbart::{MBartConfigResources, MBartModelResources, MBartVocabResources};
use rust_bert::pipelines::generation_utils::GenerateConfig;
use rust_bert::pipelines::generation_utils::GenerateOptions;
use rust_bert::pipelines::generation_utils::LanguageGenerator;
use rust_bert::resources::RemoteResource;
use std::error;
use tch::Device;

use crate::ai::AI;

pub struct MBart {
    model: MBartGenerator,
}

impl MBart {
    pub fn new() -> Self {
        let model_resource = Box::new(RemoteResource::from_pretrained(
            MBartModelResources::MBART50_MANY_TO_MANY,
        ));
        let config_resource = Box::new(RemoteResource::from_pretrained(
            MBartConfigResources::MBART50_MANY_TO_MANY,
        ));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(
            MBartVocabResources::MBART50_MANY_TO_MANY,
        ));
        let merges_resource = Box::new(RemoteResource::from_pretrained(
            MBartVocabResources::MBART50_MANY_TO_MANY,
        ));
        let device = Device::cuda_if_available();
        let generate_config = GenerateConfig {
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            min_length: 10,
            max_length: 96,
            do_sample: true,
            early_stopping: false,
            repetition_penalty: 1.0,
            temperature: 3.5,
            top_p: 0.9,
            top_k: 55,
            device,
            ..Default::default()
        };
        let model = std::thread::spawn(move || MBartGenerator::new(generate_config).unwrap())
            .join()
            .expect("Thread panicked");

        MBart { model }
    }
}

unsafe impl Send for MBart {}

unsafe impl Sync for MBart {}

#[async_trait]
impl AI for MBart {
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
            temperature: Some(temperature as f64),
            top_p: Some(top_p as f64),
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
        "mbart".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response() {
        let ai = MBart::new();
        let context = "Lots of Tesla cars to deliver before year end! Your support in taking delivery is much appreciated.".to_string();
        let output = ai
            .response(context.to_string(), 42, 0.9, 4.0, None)
            .await
            .unwrap();
        println!("{}", output);
        assert_ne!(output, context);
        assert_ne!(output.len(), 0);
        assert!(output.len() > 10);
    }
}
