# bert-cli

CLI for rust bert

## bin

Generate some text:

```sh
cargo run -- "hello world"
```

```sh
, I'm GM from csgo. this is the stream from me playing Apex Legends, I'm doing my best to play competitively and I have some games. I'm playing week 2 as a support
```

## web

```sh
cargo run --bin bert-web
```

Visit localhost:8000/swagger-ui

### config

You can configure the model in the `Rocket.toml` file.

```toml
[default]
ident = "bert-web"
# [gptneo, gpt2]
model = "gpt2"

[debug]
port = 8000

[release]
address = "0.0.0.0"
port = 8080
```

## docker

```sh
docker-compose build
docker-compose up
```

Visit localhost:8080/swagger-ui
