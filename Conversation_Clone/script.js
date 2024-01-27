document.getElementById('send-button').addEventListener('click', function() {
    var input = document.getElementById('message-input');
    var message = input.value.trim();
    if(message) {
        displayUserMessage(message);
        input.value = '';
        // Simulate a bot response
        simulateBotResponse(message);
    }
});

function displayUserMessage(message) {
    var chatBox = document.getElementById('chat-box');
    var newMessage = document.createElement('div');
    newMessage.textContent = message;
    newMessage.classList.add('message', 'user-message');
    chatBox.appendChild(newMessage);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto scroll to bottom
}



function displayMessage(message, sender) {
    var chatBox = document.getElementById('chat-box');
    var messageContainer = document.createElement('div');
    var messageText = document.createElement('div');

    messageText.textContent = message;
    
    messageContainer.appendChild(messageText);

    messageContainer.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    messageText.classList.add('message-text');

    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
}





function hideTypingIndicator() {
    var indicator = document.getElementById('typing-indicator');
    indicator.style.display = 'none';
    indicator.innerHTML = ''; // Clear the dots
}

// Simulate bot response
function simulateBotResponse(userMessage) {
    var typingMessageContainer = createTypingIndicator();
    setTimeout(function() {
        replaceTypingIndicatorWithMessage(typingMessageContainer, "Bot's reply to: " + userMessage, 'bot');
    }, 2000); // Simulate bot response time
}

function createTypingIndicator() {
    var chatBox = document.getElementById('chat-box');
    var messageContainer = document.createElement('div');
    messageContainer.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    messageContainer.classList.add('message', 'bot-message');
    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;

    var dots = messageContainer.querySelectorAll('.dot');
    dots.forEach((dot, index) => {
        dot.style.animation = `dotBounce 1.4s ${index * 0.2}s infinite ease-in-out both`;
    });
    return messageContainer;
}


function replaceTypingIndicatorWithMessage(container, message, sender) {
    container.textContent = ''; // Clear the typing indicator
    var messageText = document.createElement('div');
    messageText.textContent = message;
    messageText.classList.add('message-text');
    container.appendChild(messageText);
}