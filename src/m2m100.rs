use async_trait::async_trait;
use rust_bert::m2m_100::{
    M2M100ConfigResources, M2M100Generator, M2M100MergesResources, M2M100ModelResources,
    M2M100VocabResources,
};
use rust_bert::pipelines::generation_utils::GenerateConfig;
use rust_bert::pipelines::generation_utils::GenerateOptions;
use rust_bert::pipelines::generation_utils::LanguageGenerator;
use rust_bert::resources::RemoteResource;
use std::error;
use tch::Device;

use crate::ai::AI;

pub struct M2M100 {
    model: M2M100Generator,
}

impl M2M100 {
    pub fn new(token_max_length: u16, temperature: f32, top_p: f32) -> Self {
        let model_resource = Box::new(RemoteResource::from_pretrained(
            M2M100ModelResources::M2M100_1_2B,
        ));
        let config_resource = Box::new(RemoteResource::from_pretrained(
            M2M100ConfigResources::M2M100_1_2B,
        ));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(
            M2M100VocabResources::M2M100_1_2B,
        ));
        let merges_resource = Box::new(RemoteResource::from_pretrained(
            M2M100MergesResources::M2M100_1_2B,
        ));
        let device = Device::cuda_if_available();
        let generate_config = GenerateConfig {
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            max_length: token_max_length.into(),
            top_p: top_p.into(),
            do_sample: true,
            num_beams: 5,
            temperature: temperature.into(),
            num_return_sequences: 3,
            device,
            ..Default::default()
        };
        let model = std::thread::spawn(move || M2M100Generator::new(generate_config).unwrap())
            .join()
            .expect("Thread panicked");

        M2M100 { model }
    }
}

unsafe impl Send for M2M100 {}

unsafe impl Sync for M2M100 {}

#[async_trait]
impl AI for M2M100 {
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
        "m2m100".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response() {
        let ai = M2M100::new(42, 0.9, 1.1);
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
