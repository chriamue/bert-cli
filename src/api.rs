#[macro_use]
extern crate rocket_include_static_resources;

use bert_cli::{create_ai, Bert};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::form::FromForm;
use rocket::http::Header;
use rocket::State;
use rocket::{get, post, routes, serde::json::Json};
use rocket::{Request, Response};
use rocket_include_static_resources::{EtagIfNoneMatch, StaticContextManager, StaticResponse};
use rocket_okapi::okapi::schemars;
use rocket_okapi::okapi::schemars::JsonSchema;
use rocket_okapi::settings::UrlObject;
use rocket_okapi::{openapi, openapi_get_routes, rapidoc::*, swagger_ui::*};
use serde::{Deserialize, Serialize};
use std::time::Instant;

mod config;
use config::Config;

fn example_context() -> &'static str {
    "Hello World!"
}

fn example_top_p() -> f32 {
    0.9
}

fn example_temp() -> f32 {
    0.8
}

fn example_response_length() -> u16 {
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
    response_length: u16,
    #[schemars(example = "example_remove_input")]
    remove_input: Option<bool>,
}

#[derive(Serialize, Deserialize, JsonSchema, FromForm)]
struct GenerationResponse {
    #[schemars(example = "example_generated_text")]
    generated_text: String,
    duration: u128,
}

cached_static_response_handler! {
    259_200;
    "/index.js" => cached_indexjs => "indexjs",
    "/index.css" => cached_indexcss => "indexcss",
}

#[get("/")]
fn default_index(
    static_resources: &State<StaticContextManager>,
    etag_if_none_match: EtagIfNoneMatch,
) -> StaticResponse {
    static_resources.build(&etag_if_none_match, "index")
}

#[openapi(tag = "Generation")]
#[get("/completion?<request..>")]
async fn get_completion(
    bert: &State<Bert>,
    request: GenerationRequest,
) -> Json<GenerationResponse> {
    let start = Instant::now();
    let response = bert
        .generate(
            request.context.to_string(),
            request.response_length,
            request.temp,
            request.top_p,
            None,
        )
        .await
        .unwrap();
    let duration = start.elapsed().as_millis();
    Json(GenerationResponse {
        generated_text: response.text,
        duration,
    })
}

#[openapi(tag = "Generation")]
#[post("/completion", data = "<request>")]
async fn post_completion(
    bert: &State<Bert>,
    request: Json<GenerationRequest>,
) -> Json<GenerationResponse> {
    get_completion(bert, request.into_inner()).await
}

pub struct CORS;

#[rocket::async_trait]
impl Fairing for CORS {
    fn info(&self) -> Info {
        Info {
            name: "Attaching CORS headers to responses",
            kind: Kind::Response,
        }
    }

    async fn on_response<'r>(&self, _request: &'r Request<'_>, response: &mut Response<'r>) {
        response.set_header(Header::new("Access-Control-Allow-Origin", "*"));
        response.set_header(Header::new(
            "Access-Control-Allow-Methods",
            "POST, GET, PATCH, OPTIONS",
        ));
        response.set_header(Header::new("Access-Control-Allow-Headers", "*"));
        response.set_header(Header::new("Access-Control-Allow-Credentials", "true"));
    }
}

#[rocket::main]
async fn main() {
    let rocket = rocket::build();

    let figment = rocket.figment();
    let config: Config = figment.extract().expect("config");

    let bert = Bert {
        ai: create_ai(
            config.model,
            config.token_max_length,
            config.temperature,
            config.top_p,
        ),
    };

    println!("Model {} loaded.", bert.ai.name());
    let launch_result = rocket
        .attach(static_resources_initializer!(
            "indexjs" => "static/index.js",
            "indexcss" => "static/index.css",
            "index" => ("static", "index.html"),
        ))
        .attach(CORS)
        .mount("/", routes![cached_indexjs, cached_indexcss, default_index])
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
        .manage(bert)
        .launch()
        .await;
    match launch_result {
        Ok(()) => println!("Rocket shut down gracefully."),
        Err(err) => println!("Rocket had an error: {}", err),
    };
}
