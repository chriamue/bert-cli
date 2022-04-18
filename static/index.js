
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
            const { generated_text, duration } = data;
            document.getElementById("generated_text").innerHTML = generated_text;
            document.getElementById("generated_duration").innerHTML = `Generated in ${duration / 1000}s`;
        }
        );
}

const generate_button = document.getElementById("generate_button");
generate_button.addEventListener("click", generate);
