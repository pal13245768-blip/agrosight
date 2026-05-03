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
    loadHistory();
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('agrosight_history') || '[]');
    if (history.length > 0) {
        elements.sessionHistory.innerHTML = history.map(item => {
            const activeClass = item.id === currentSessionId ? 'active' : '';
            return `
                <div class="history-item ${activeClass}" onclick="loadSession('${item.id}')">
                    <i data-lucide="message-square"></i>
                    <span>${item.title || 'Previous Chat'}</span>
                </div>
            `;
        }).join('');
        lucide.createIcons();
    } else {
        elements.sessionHistory.innerHTML = '<div class="history-empty">No recent chats</div>';
    }
}

function loadSession(sessionId) {
    currentSessionId = sessionId;
    localStorage.setItem('agrosight_session_id', currentSessionId);
    loadHistory();
    elements.chatMessages.innerHTML = '';
    elements.welcomeScreen.style.opacity = '0';
    elements.welcomeScreen.style.pointerEvents = 'none';

    const sessions = JSON.parse(localStorage.getItem('agrosight_sessions') || '{}');
    const conversation = sessions[currentSessionId] || [];

    if (conversation.length === 0) {
        elements.welcomeScreen.style.opacity = '1';
        elements.welcomeScreen.style.pointerEvents = 'auto';
        return;
    }

    conversation.forEach((turn) => {
        appendMessage(turn.role, turn.content);
    });
}

async function fetchWeatherMock() {
    // First try browser geolocation, then fallback to IP-based geolocation
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const { latitude, longitude } = position.coords;
                await fetchWeatherByCoordinates(latitude, longitude);
            },
            () => {
                // Geolocation denied or unavailable, use IP-based geolocation
                fetchWeatherByIP();
            }
        );
    } else {
        fetchWeatherByIP();
    }
}

async function fetchWeatherByIP() {
    try {
        // Try multiple IP geolocation services
        let locationData = null;
        
        // Primary: ip-api.com
        try {
            const response = await fetch('https://ip-api.com/json/?fields=city,lat,lon');
            if (response.ok) {
                locationData = await response.json();
                if (locationData.lat && locationData.lon) {
                    await fetchWeatherByCoordinates(locationData.lat, locationData.lon, locationData.city);
                    return;
                }
            }
        } catch (e) {
            console.warn('ip-api.com failed:', e);
        }
        
        // Secondary: ipapi.co
        try {
            const response = await fetch('https://ipapi.co/json/');
            if (response.ok) {
                locationData = await response.json();
                if (locationData.latitude && locationData.longitude) {
                    await fetchWeatherByCoordinates(locationData.latitude, locationData.longitude, locationData.city);
                    return;
                }
            }
        } catch (e) {
            console.warn('ipapi.co failed:', e);
        }
        
        // Tertiary: geojs.io
        try {
            const response = await fetch('https://get.geojs.io/v1/ip/geo.json');
            if (response.ok) {
                locationData = await response.json();
                if (locationData.latitude && locationData.longitude) {
                    await fetchWeatherByCoordinates(locationData.latitude, locationData.longitude, locationData.city);
                    return;
                }
            }
        } catch (e) {
            console.warn('geojs.io failed:', e);
        }
        
        // If all fail
        elements.weatherDisplay.textContent = 'Location: Unable to determine';
    } catch (error) {
        console.error('IP geolocation fallback error:', error);
        elements.weatherDisplay.textContent = 'Location: Unable to access';
    }
}



async function fetchWeatherByCoordinates(latitude, longitude, cityHint = null) {
    try {
        // Fetch weather from OpenWeatherMap
        const response = await fetch(
            `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=fcc4f3fe373e37c3b00e0bf86236f3bc`
        );
        
        if (response.ok) {
            const data = await response.json();
            const location = data.name || cityHint || 'Current Location';
            const temp = Math.round(data.main.temp);
            const weather = data.weather[0].main;
            elements.weatherDisplay.textContent = `${location}: ${temp}°C, ${weather}`;
        } else {
            getLocationFromCoordinates(latitude, longitude, cityHint);
        }
    } catch (error) {
        console.error('Weather fetch error:', error);
        getLocationFromCoordinates(latitude, longitude, cityHint);
    }
}

async function getLocationFromCoordinates(latitude, longitude, cityHint = null) {
    try {
        // Fallback: Use Nominatim (OpenStreetMap) for reverse geocoding
        const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
        );
        
        if (response.ok) {
            const data = await response.json();
            const location = cityHint || data.address?.city || data.address?.town || data.address?.village || 'Current Location';
            elements.weatherDisplay.textContent = `${location}: ${Math.round(latitude)}°N, ${Math.round(longitude)}°E`;
        }
    } catch (error) {
        console.error('Reverse geocoding error:', error);
        elements.weatherDisplay.textContent = 'Location: Unable to fetch';
    }
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
    // Session metadata for sidebar history
    let history = JSON.parse(localStorage.getItem('agrosight_history') || '[]');
    const existingIndex = history.findIndex(h => h.id === currentSessionId);
    const title = question.substring(0, 30) + '...';

    if (existingIndex === -1) {
        history.unshift({ id: currentSessionId, title, updatedAt: Date.now() });
    } else {
        history[existingIndex].title = title;
        history[existingIndex].updatedAt = Date.now();
        history.unshift(history.splice(existingIndex, 1)[0]);
    }

    localStorage.setItem('agrosight_history', JSON.stringify(history.slice(0, 10)));

    // Persist full conversation for the session
    const sessions = JSON.parse(localStorage.getItem('agrosight_sessions') || '{}');
    sessions[currentSessionId] = sessions[currentSessionId] || [];
    sessions[currentSessionId].push({ role: 'user', content: question });
    sessions[currentSessionId].push({ role: 'assistant', content: answer });
    localStorage.setItem('agrosight_sessions', JSON.stringify(sessions));

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

function queryCurrentWeather() {
    // Get current location and query weather for it
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            async (position) => {
                const { latitude, longitude } = position.coords;
                await getLocationNameForQuery(latitude, longitude);
            },
            () => {
                // Geolocation denied, use IP-based approach
                queryWeatherByIP();
            }
        );
    } else {
        queryWeatherByIP();
    }
}

async function queryWeatherByIP() {
    try {
        let locationData = null;
        
        // Primary: ip-api.com
        try {
            const response = await fetch('https://ip-api.com/json/?fields=city,lat,lon');
            if (response.ok) {
                locationData = await response.json();
                if (locationData.city) {
                    quickQuery(`Current weather in ${locationData.city}`);
                    return;
                }
                if (locationData.lat && locationData.lon) {
                    quickQuery(`Current weather at coordinates ${locationData.lat}, ${locationData.lon}`);
                    return;
                }
            }
        } catch (e) {
            console.warn('ip-api.com failed:', e);
        }
        
        // Secondary: ipapi.co
        try {
            const response = await fetch('https://ipapi.co/json/');
            if (response.ok) {
                locationData = await response.json();
                if (locationData.city) {
                    quickQuery(`Current weather in ${locationData.city}`);
                    return;
                }
                if (locationData.latitude && locationData.longitude) {
                    quickQuery(`Current weather at coordinates ${locationData.latitude}, ${locationData.longitude}`);
                    return;
                }
            }
        } catch (e) {
            console.warn('ipapi.co failed:', e);
        }
        
        // Tertiary: geojs.io
        try {
            const response = await fetch('https://get.geojs.io/v1/ip/geo.json');
            if (response.ok) {
                locationData = await response.json();
                if (locationData.city) {
                    quickQuery(`Current weather in ${locationData.city}`);
                    return;
                }
                if (locationData.latitude && locationData.longitude) {
                    quickQuery(`Current weather at coordinates ${locationData.latitude}, ${locationData.longitude}`);
                    return;
                }
            }
        } catch (e) {
            console.warn('geojs.io failed:', e);
        }
        
        // If all fail
        quickQuery('Current weather in my location');
    } catch (error) {
        console.error('IP geolocation fallback error:', error);
        quickQuery('Current weather in my location');
    }
}

async function getLocationNameForQuery(latitude, longitude) {
    try {
        const response = await fetch(
            `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=fcc4f3fe373e37c3b00e0bf86236f3bc`
        );
        
        if (response.ok) {
            const data = await response.json();
            const location = data.name || `${latitude}, ${longitude}`;
            quickQuery(`Current weather in ${location}`);
        } else {
            quickQuery(`Current weather at coordinates ${latitude}, ${longitude}`);
        }
    } catch (error) {
        quickQuery(`Current weather at coordinates ${latitude}, ${longitude}`);
    }
}

// Start
init();
