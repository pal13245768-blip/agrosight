/**
 * AgroSight - Premium Frontend Logic
 * Handles SSE streaming, session management, and UI transitions.
 */

const elements = {
    chatForm: document.getElementById('chat-form'),
    userInput: document.getElementById('user-input'),
    chatMessages: document.getElementById('chat-messages'),
    welcomeScreen: document.getElementById('welcome-screen'),
    sessionHistory: document.getElementById('session-history'),
    newChatBtn: document.getElementById('new-chat-btn'),
    sendBtn: document.getElementById('send-btn'),
    weatherDisplay: document.getElementById('weather-display')
};

let currentSessionId = localStorage.getItem('agrosight_session_id') || '';
let isGenerating = false;

// Initialize
function init() {
    if (!currentSessionId) {
        createNewSession();
    }
    loadHistory();
    fetchWeatherMock();
}

function createNewSession() {
    currentSessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('agrosight_session_id', currentSessionId);
    elements.chatMessages.innerHTML = '';
    elements.welcomeScreen.style.opacity = '1';
    elements.welcomeScreen.style.pointerEvents = 'auto';
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('agrosight_history') || '[]');
    if (history.length > 0) {
        elements.sessionHistory.innerHTML = history.map(item => `
            <div class="history-item" onclick="loadSession('${item.id}')">
                <i data-lucide="message-square"></i>
                <span>${item.title || 'Previous Chat'}</span>
            </div>
        `).join('');
        lucide.createIcons();
    }
}

async function fetchWeatherMock() {
    // In a real app, we might call /search or a separate endpoint
    elements.weatherDisplay.textContent = 'Ahmedabad: , Sunny';
}

// UI Helpers
function appendMessage(role, content = '') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    // Convert basic markdown to HTML
    messageDiv.innerHTML = formatMarkdown(content);

    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    if (elements.welcomeScreen.style.opacity !== '0') {
        elements.welcomeScreen.style.opacity = '0';
        elements.welcomeScreen.style.pointerEvents = 'none';
    }

    return messageDiv;
}

function formatMarkdown(text) {
    if (!text) return '';
    let html = text
        // Bold text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic text
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Headings
        .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
        .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
        .replace(/^# (.*?)$/gm, '<h1>$1</h1>')
        // Lists
        .replace(/^\s*\-\s(.*)$/gm, '<li>$1</li>')
        .replace(/^\s*\d+\.\s(.*)$/gm, '<li>$1</li>')
        // Wrap contiguous list items in ul/ol tag manually or just rely on CSS
        // Source tags
        .replace(/\[Source: (.*?)\]/g, '<span class="source-tag">Source: $1</span>');

    // Multiple newlines to paragraphs
    html = html.replace(/\n{2,}/g, '</p><p>');
    // Single newlines to line breaks (unless they follow a block tag like h2/h3/li)
    html = html.replace(/(?<!<\/h[1-6]>|<\/li>|<\/p>)\n/g, '<br>');

    // Wrap the whole thing in paragraphs if missing
    return `<p>${html}</p>`.replace(/<p><\/p>/g, '');
}

// Core Chat Logic
async function sendMessage(question) {
    if (isGenerating || !question.trim()) return;

    isGenerating = true;
    elements.userInput.value = '';
    elements.sendBtn.disabled = true;

    // Add user message
    appendMessage('user', question);

    // Prepare AI message bubble for streaming
    const aiMessageDiv = appendMessage('ai', '...');
    let fullAnswer = "";

    try {
        // We use the same /chat endpoint (stream=true by default)
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: currentSessionId,
                stream: true
            })
        });

        if (!response.ok) throw new Error('API request failed');

        // Manual SSE handling
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let currentEvent = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                if (trimmed.startsWith('event: ')) {
                    currentEvent = trimmed.slice(7);
                } else if (trimmed.startsWith('data: ')) {
                    const data = trimmed.slice(6);
                    if (data === '[DONE]') continue;

                    if (currentEvent === 'token') {
                        try {
                            const parsedData = JSON.parse(data);
                            fullAnswer += parsedData;
                        } catch (e) {
                            fullAnswer += data;
                        }
                        aiMessageDiv.innerHTML = formatMarkdown(fullAnswer);
                        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
                    } else if (currentEvent === 'session') {
                        console.log('Session ID:', data);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Streaming error:', error);
        aiMessageDiv.innerHTML = `<span class="error">Sorry, I encountered an error. Please try again.</span>`;
    } finally {
        isGenerating = false;
        elements.sendBtn.disabled = false;
        saveToLocalHistory(question, fullAnswer);
    }
}

function saveToLocalHistory(question, answer) {
    // Simplified local persistence for the demo
    let history = JSON.parse(localStorage.getItem('agrosight_history') || '[]');
    const existing = history.findIndex(h => h.id === currentSessionId);

    if (existing === -1) {
        history.unshift({ id: currentSessionId, title: question.substring(0, 30) + '...' });
    }

    localStorage.setItem('agrosight_history', JSON.stringify(history.slice(0, 10)));
    loadHistory();
}

// Event Listeners
elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage(elements.userInput.value);
});

elements.newChatBtn.addEventListener('click', () => {
    createNewSession();
});

function quickQuery(text) {
    elements.userInput.value = text;
    sendMessage(text);
}

// Start
init();
