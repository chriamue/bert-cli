
async function generate() {

    let data = {
        context: document.getElementById("context").value,
        top_p: document.getElementById("top_p").value / 100.0,
        temp: document.getElementById("temperature").value / 100.0,
        response_length: 200,
    };

    fetch("/api/completion", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }).then(response => response.json())
        .then(data => {
            const { generated_text } = data;
            document.getElementById("generated_text").innerHTML = generated_text;
        }
        );
}

const generate_button = document.getElementById("generate_button");
generate_button.addEventListener("click", generate);
