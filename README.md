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

## docker

```sh
docker-compose build
docker-compose up
```

Visit localhost:8080/swagger-ui
