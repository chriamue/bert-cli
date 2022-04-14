FROM rust:bullseye AS builder
WORKDIR /usr/src/

RUN USER=root cargo new bert-cli
WORKDIR /usr/src/bert-cli
COPY Cargo.toml Cargo.lock ./
RUN echo "fn main() {}" > src/bin.rs
RUN echo "fn main() {}" > src/api.rs
RUN cargo build --release
RUN rm src/*.rs
COPY src ./src
RUN touch src/bin.rs
RUN cargo build --bin bert-web --release
COPY Rocket.toml /Rocket.toml

ENTRYPOINT [ "cargo", "run", "--bin", "bert-web", "--release" ]