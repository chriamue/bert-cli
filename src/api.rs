use rocket::form::FromForm;
use rocket::{get, post, serde::json::Json};
use rocket_okapi::okapi::schemars;
use rocket_okapi::okapi::schemars::JsonSchema;
use rocket_okapi::settings::UrlObject;
use rocket_okapi::{openapi, openapi_get_routes, rapidoc::*, swagger_ui::*};
use serde::{Deserialize, Serialize};

fn example_context() -> &'static str {
    "Hello World!"
}

fn example_top_p() -> f32 {
    0.9
}

fn example_temp() -> f32 {
    0.8
}

fn example_response_length() -> u32 {
    128
}

fn example_remove_input() -> Option<bool> {
    Some(true)
}

fn example_generated_text() -> &'static str {
    "I want to generate a number sequence that includes the words \"Hello world\"."
}

#[derive(Serialize, Deserialize, JsonSchema, FromForm)]
struct GenerationRequest {
    #[schemars(example = "example_context")]
    context: String,
    #[schemars(example = "example_top_p")]
    top_p: f32,
    #[schemars(example = "example_temp")]
    temp: f32,
    #[schemars(example = "example_response_length")]
    response_length: u32,
    #[schemars(example = "example_remove_input")]
    remove_input: Option<bool>,
}

#[derive(Serialize, Deserialize, JsonSchema, FromForm)]
struct GenerationResponse {
    #[schemars(example = "example_generated_text")]
    generated_text: String,
}

#[openapi(tag = "Generation")]
#[get("/completion?<request..>")]
fn get_completion(request: GenerationRequest) -> Json<GenerationResponse> {
    Json(GenerationResponse {
        generated_text: request.context.to_string(),
    })
}

#[openapi(tag = "Generation")]
#[post("/completion", data = "<request>")]
fn post_completion(request: Json<GenerationRequest>) -> Json<GenerationResponse> {
    Json(GenerationResponse {
        generated_text: request.context.to_string(),
    })
}

#[rocket::main]
async fn main() {
    let launch_result = rocket::build()
        .mount(
            "/api/",
            openapi_get_routes![get_completion, post_completion],
        )
        .mount(
            "/swagger-ui/",
            make_swagger_ui(&SwaggerUIConfig {
                url: "../api/openapi.json".to_owned(),
                ..Default::default()
            }),
        )
        .mount(
            "/rapidoc/",
            make_rapidoc(&RapiDocConfig {
                general: GeneralConfig {
                    spec_urls: vec![UrlObject::new("General", "../api/openapi.json")],
                    ..Default::default()
                },
                hide_show: HideShowConfig {
                    allow_spec_url_load: false,
                    allow_spec_file_load: false,
                    ..Default::default()
                },
                ..Default::default()
            }),
        )
        .launch()
        .await;
    match launch_result {
        Ok(()) => println!("Rocket shut down gracefully."),
        Err(err) => println!("Rocket had an error: {}", err),
    };
}
