const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");

function addMessage(message, sender) {
    const messageElement = document.createElement("div");

    if (sender === "user") {
        messageElement.className = "message user-message";

        messageElement.innerHTML = `
            <div class="message-content">
                <div class="message-name">You</div>
                <div class="message-bubble">${escapeHtml(message)}</div>
            </div>
        `;
    } else {
        messageElement.className = "message bot-message";

        messageElement.innerHTML = `
            <div class="message-avatar">🤖</div>

            <div class="message-content">
                <div class="message-name">Support Bot</div>
                <div class="message-bubble">${escapeHtml(message)}</div>
            </div>
        `;
    }

    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function addLoadingMessage() {
    const loadingElement = document.createElement("div");

    loadingElement.className = "message bot-message";
    loadingElement.id = "loadingMessage";

    loadingElement.innerHTML = `
        <div class="message-avatar">🤖</div>

        <div class="message-content">
            <div class="message-name">Support Bot</div>

            <div class="message-bubble">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;

    chatMessages.appendChild(loadingElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function removeLoadingMessage() {
    const loadingMessage =
        document.getElementById("loadingMessage");

    if (loadingMessage) {
        loadingMessage.remove();
    }
}


async function sendMessage(message) {
    const cleanedMessage = message.trim();

    if (!cleanedMessage) {
        return;
    }

    addMessage(cleanedMessage, "user");

    messageInput.value = "";
    sendButton.disabled = true;

    addLoadingMessage();

    try {
        const response = await fetch("/chat", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: cleanedMessage
            })
        });

        const data = await response.json();

        removeLoadingMessage();

        if (!response.ok) {
            throw new Error(
                data.detail || "Something went wrong."
            );
        }

        addMessage(
            data.response,
            "bot"
        );

    } catch (error) {

        removeLoadingMessage();

        addMessage(
            "Sorry, I couldn't connect to the support service. Please try again.",
            "bot"
        );

        console.error(error);

    } finally {
        sendButton.disabled = false;
        messageInput.focus();
    }
}


sendButton.addEventListener(
    "click",
    function () {
        sendMessage(messageInput.value);
    }
);


messageInput.addEventListener(
    "keydown",
    function (event) {

        if (event.key === "Enter") {
            event.preventDefault();

            sendMessage(
                messageInput.value
            );
        }
    }
);


document
    .querySelectorAll("[data-question]")
    .forEach(function (button) {

        button.addEventListener(
            "click",
            function () {

                const question =
                    button.getAttribute(
                        "data-question"
                    );

                sendMessage(question);
            }
        );
    });


function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}