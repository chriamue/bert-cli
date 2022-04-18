use async_trait::async_trait;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::prophetnet::{
    ProphetNetConfigResources, ProphetNetModelResources, ProphetNetVocabResources,
};
use rust_bert::resources::RemoteResource;
use std::error;
use tch::Device;

use crate::ai::AI;

pub struct ProphetNet {
    model: TextGenerationModel,
}

impl ProphetNet {
    pub fn new(token_max_length: u16, temperature: f32, top_p: f32) -> Self {
        let config_resource = Box::new(RemoteResource::from_pretrained(
            ProphetNetConfigResources::PROPHETNET_LARGE_UNCASED,
        ));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(
            ProphetNetVocabResources::PROPHETNET_LARGE_UNCASED,
        ));
        let merges_resource = Box::new(RemoteResource::from_pretrained(
            ProphetNetVocabResources::PROPHETNET_LARGE_UNCASED,
        ));
        let model_resource = Box::new(RemoteResource::from_pretrained(
            ProphetNetModelResources::PROPHETNET_LARGE_UNCASED,
        ));
        let generate_config = TextGenerationConfig {
            model_type: ModelType::ProphetNet,
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            min_length: 10,
            max_length: token_max_length.into(),
            do_sample: true,
            early_stopping: false,
            repetition_penalty: 1.1,
            temperature: temperature.into(),
            top_p: top_p.into(),
            top_k: 55,
            device: Device::cuda_if_available(),
            ..Default::default()
        };

        let model = std::thread::spawn(move || {
            let mut model = TextGenerationModel::new(generate_config).unwrap();
            model.set_device(Device::cuda_if_available());
            model
        })
        .join()
        .expect("Thread panicked");
        ProphetNet { model }
    }
}

unsafe impl Send for ProphetNet {}

unsafe impl Sync for ProphetNet {}

#[async_trait]
impl AI for ProphetNet {
    async fn response(
        &self,
        context: String,
        _token_max_length: u16,
        _temperature: f32,
        _top_p: f32,
        _stop_sequence: Option<String>,
    ) -> Result<String, Box<dyn error::Error>> {
        let output = self.model.generate(&[context.to_string()], None);
        let response = output[0].to_string();
        let response = response.replace(context.as_str(), "");
        Ok(response)
    }

    fn name(&self) -> String {
        "prophetnet".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response() {
        let ai = ProphetNet::new(42, 1.1, 0.9);
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
