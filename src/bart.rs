use async_trait::async_trait;
use rust_bert::bart::{
    BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::RemoteResource;
use std::error;
use tch::Device;

use crate::ai::AI;

pub struct Bart {
    model: TextGenerationModel,
}

impl Bart {
    pub fn new() -> Self {
        let config_resource = Box::new(RemoteResource::from_pretrained(BartConfigResources::BART));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(BartVocabResources::BART));
        let merges_resource = Box::new(RemoteResource::from_pretrained(BartMergesResources::BART));
        let model_resource = Box::new(RemoteResource::from_pretrained(BartModelResources::BART));
        let generate_config = TextGenerationConfig {
            model_type: ModelType::Bart,
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
            device: Device::Cpu,
            ..Default::default()
        };

        let model = std::thread::spawn(move || {
            let mut model = TextGenerationModel::new(generate_config).unwrap();
            model.set_device(Device::cuda_if_available());
            model
        })
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
        stop_sequence: Option<String>,
    ) -> Result<String, Box<dyn error::Error>> {
        let output = self.model.generate(&[context.to_string()], None);
        let response = output[0].to_string();
        let response = response.replace(context.as_str(), "");
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
        let ai = Bart::new();
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
