// Session & Auth State
let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
let jwtToken = null;
let verifiedUser = null;
// Clear previous session data on load to force fresh login
localStorage.removeItem('ucsi_jwt');
localStorage.removeItem('ucsi_user');
localStorage.removeItem('ucsi_high_until');
let highSecurityExpiresAt = 0;
if (isNaN(highSecurityExpiresAt) || Date.now() > highSecurityExpiresAt) {
    highSecurityExpiresAt = null;
}

function isHighSecurityActive() {
    return verifiedUser && highSecurityExpiresAt && Date.now() < highSecurityExpiresAt;
}

function clearHighSecurityState() {
    highSecurityExpiresAt = null;
    localStorage.removeItem('ucsi_high_until');
}

function updateSecurityBadge() {
    const statusEl = document.getElementById('security-status-text');
    const helperEl = document.getElementById('security-helper-text');
    const btn = document.getElementById('security-action-btn');
    const panel = document.getElementById('security-panel');
    if (!statusEl || !helperEl || !btn || !panel) return;

    if (!verifiedUser || !jwtToken) {
        statusEl.textContent = 'Login required to unlock grade-related answers.';
        helperEl.textContent = 'Step 1 of 2';
        btn.textContent = 'Login';
        btn.disabled = false;
        panel.classList.remove('ring-2', 'ring-primary', 'animate-pulse');
        return;
    }

    if (isHighSecurityActive()) {
        const remainingMs = highSecurityExpiresAt - Date.now();
        const minutes = Math.max(1, Math.floor(remainingMs / 60000));
        statusEl.textContent = `Unlocked for ${minutes} min${minutes > 1 ? 's' : ''}.`;
        helperEl.textContent = 'High-security mode active';
        btn.textContent = 'Unlocked';
        btn.disabled = true;
        panel.classList.remove('ring-2', 'ring-primary', 'animate-pulse');
    } else {
        statusEl.textContent = 'Locked. Enter your password to view grades.';
        helperEl.textContent = 'Step 2 of 2';
        btn.textContent = 'Enter Password';
        btn.disabled = false;
        panel.classList.remove('ring-2', 'ring-primary', 'animate-pulse');
    }
}

function openHighSecurityPanel() {
    if (!verifiedUser || !jwtToken) {
        showVerifyScreen();
        return;
    }
    showPasswordScreen();
}

function pulseSecurityPanel() {
    updateSecurityBadge();
    const panel = document.getElementById('security-panel');
    if (!panel) return;
    panel.classList.add('ring-2', 'ring-primary', 'animate-pulse');
    setTimeout(() => panel.classList.remove('ring-2', 'ring-primary', 'animate-pulse'), 1800);
}

// Initialize UI State
window.onload = function () {
    // Recover login state if token exists
    if (jwtToken && verifiedUser) {
        updateLoginButton();
    } else {
        localStorage.removeItem('ucsi_jwt');
        localStorage.removeItem('ucsi_user');
        clearHighSecurityState();
    }

    if (highSecurityExpiresAt && Date.now() > highSecurityExpiresAt) {
        clearHighSecurityState();
    }

    updateSecurityBadge();

    // Allow interaction
    document.getElementById('chips-area').classList.remove('opacity-50', 'pointer-events-none');
    document.getElementById('input-area').classList.remove('opacity-50', 'pointer-events-none');

    setInterval(() => {
        if (highSecurityExpiresAt && Date.now() > highSecurityExpiresAt) {
            clearHighSecurityState();
        }
        updateSecurityBadge();
    }, 30000);
};

function toggleChatbot() {
    document.getElementById('chatbot-window').classList.toggle('hidden');
}

// --- Screens ---
function showVerifyScreen() {
    if (verifiedUser) {
        if (confirm(`Logged in as ${verifiedUser.name}. Logout?`)) logout();
        return;
    }
    document.getElementById('verify-screen').classList.remove('hidden');
}

function hideVerifyScreen() {
    document.getElementById('verify-screen').classList.add('hidden');
}

function showPasswordScreen() {
    document.getElementById('password-screen').classList.remove('hidden');
}

function hidePasswordScreen() {
    document.getElementById('password-screen').classList.add('hidden');
    document.getElementById('secure-password').value = '';
}

function updateLoginButton() {
    const btn = document.getElementById('header-login-btn');
    const btnText = document.getElementById('login-btn-text');
    const btnIcon = btn.querySelector('.material-icons');

    if (verifiedUser) {
        btnText.textContent = verifiedUser.name.split(' ')[0];
        btnIcon.textContent = 'person';
        btn.classList.add('bg-green-500/30');
    } else {
        btnText.textContent = 'Login';
        btnIcon.textContent = 'login';
        btn.classList.remove('bg-green-500/30');
    }
}

// --- API Interaction (JWT Aware) ---

async function verifyStudent() {
    const studentNumber = document.getElementById('verify-student-number').value.trim();
    const name = document.getElementById('verify-name').value.trim();
    const btn = document.querySelector('#verify-screen button:first-of-type');

    if (!studentNumber || !name) return alert('Please enter both fields.');

    btn.innerText = 'Verifying...';

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ student_number: studentNumber, name: name })
        });

        const data = await response.json();

        if (data.token) {
            // Success: Store Token
            jwtToken = data.token;
            verifiedUser = { name: data.user.name, student_number: data.user.student_number };

            localStorage.setItem('ucsi_jwt', jwtToken);
            localStorage.setItem('ucsi_user', JSON.stringify(verifiedUser));

            hideVerifyScreen();
            updateLoginButton();
            appendMessage(`🔐Identity Verified! Welcome, ${data.user.name}.`, 'ai');
        } else {
            alert(data.message || 'Verification failed.');
        }
    } catch (e) {
        console.error(e);
        alert('Server error.');
    } finally {
        btn.innerText = 'Verify Identity';
    }
}

async function submitPassword() {
    const password = document.getElementById('secure-password').value;
    if (!password) return;

    try {
        const response = await fetch('/api/verify_password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${jwtToken}`
            },
            body: JSON.stringify({ password: password })
        });

        const data = await response.json();

        if (data.success) {
            const ttlSeconds = Number(data.expires_in_seconds || 600);
            highSecurityExpiresAt = Date.now() + (ttlSeconds * 1000);
            localStorage.setItem('ucsi_high_until', String(highSecurityExpiresAt));
            updateSecurityBadge();
            hidePasswordScreen();
            appendMessage("🔓 High Security Mode Enabled (10 mins). You can now verify your grades.", 'ai');
            // Automatically retry the last logical request if possible, or just ask user to ask again.
            // For simplicity, ask user to re-type.
        } else {
            alert("Incorrect Password.");
        }
    } catch (e) {
        alert("Error verifying password.");
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    appendMessage(message, 'user');
    input.value = '';

    // Clear old suggestions
    const chips = document.getElementById('chips-area');
    if (chips) chips.innerHTML = '';

    const loadingId = appendLoading();

    try {
        const headers = { 'Content-Type': 'application/json' };
        if (jwtToken) headers['Authorization'] = `Bearer ${jwtToken}`;

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ message: message, session_id: sessionId })
        });

        const data = await response.json();
        document.getElementById(loadingId).remove();

        // Handle Auth/Security Challenges
        if (data.error) {
            appendMessage("⚠️ Error: " + data.error, 'ai');
        }
        else if (data.type === 'verify_required' || data.type === 'login_hint') {
            appendMessage("🔒 " + data.response, 'ai');
            showVerifyScreen();
        }
        else if (data.type === 'password_required' || data.type === 'password_prompt') {
            appendMessage("🔒 " + data.response, 'ai');
            pulseSecurityPanel();
            openHighSecurityPanel();
        }
        else {
            // PARSE JSON RESPONSE FROM AI ENGINE
            let aiText = data.response;
            let suggestions = [];

            try {
                // Check if response is JSON string
                const parsed = JSON.parse(data.response);
                if (parsed.text) {
                    aiText = parsed.text;
                    suggestions = parsed.suggestions || [];
                }
            } catch (e) {
                // Not JSON, use as is
            }

            appendMessage(aiText, 'ai', message);

            // Render suggestions
            if (suggestions.length > 0) {
                renderSuggestions(suggestions);
            }
        }

    } catch (error) {
        if (document.getElementById(loadingId)) document.getElementById(loadingId).remove();
        appendMessage("Connection failed.", 'ai');
    }
}

function renderSuggestions(list) {
    const chips = document.getElementById('chips-area');
    if (!chips) return;

    chips.innerHTML = '';
    list.forEach(item => {
        const btn = document.createElement('button');
        btn.onclick = () => sendChip(item);
        btn.className = "shrink-0 whitespace-nowrap bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-full px-3 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-300 hover:bg-red-50 hover:text-primary hover:border-red-200 transition-colors shadow-sm animate-fade-in-up";
        btn.textContent = "💡" + item;
        chips.appendChild(btn);
    });
    // Auto scroll chips to start
    chips.scrollLeft = 0;
}

async function logout() {
    localStorage.removeItem('ucsi_jwt');
    localStorage.removeItem('ucsi_user');
    jwtToken = null;
    verifiedUser = null;
    updateLoginButton();
    clearHighSecurityState();
    updateSecurityBadge();
    appendMessage('Logged out.', 'ai');

    // Optional: Call server to blacklist token (not implemented strictly in backend yet)
    fetch('/api/logout', { method: 'POST' });
}

// --- UI Helpers ---
function handleKeyPress(e) { if (e.key === 'Enter') sendMessage(); }
function sendChip(msg) { document.getElementById('chat-input').value = msg; sendMessage(); }

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function parseStructuredProfileText(text) {
    const raw = String(text ?? '').replace(/\r\n/g, '\n');
    const lines = raw.split('\n').map(line => line.trim()).filter(Boolean);
    if (lines.length < 4) return null;

    let header = '';
    let startIndex = 0;
    if (!lines[0].includes(':') && !lines[0].includes('：')) {
        header = lines[0];
        startIndex = 1;
    }

    const keyMap = new Set([
        'student number', 'student name', 'nationality', 'gender', 'programme',
        'profile status', 'intake', 'date of birth', 'department', 'gpa', 'advisor',
        '학번', '이름', '국적', '성별', '전공/프로그램', '상태', '입학 시기', '생년월일', '학과', '지도교수'
    ]);

    const rows = [];
    for (let i = startIndex; i < lines.length; i += 1) {
        const m = lines[i].match(/^([^:：]{1,60})\s*[:：]\s*(.+)$/);
        if (!m) continue;
        const key = m[1].trim();
        const value = m[2].trim();
        if (!key || !value) continue;
        rows.push({ key, value });
    }

    if (rows.length < 3) return null;

    const keyMatches = rows.filter(row => keyMap.has(row.key.toLowerCase())).length;
    const headerLooksLikeProfile = /profile information|student info|학생 정보/.test(header.toLowerCase());
    if (!headerLooksLikeProfile && keyMatches < 2) return null;

    return { header, rows };
}

function buildStructuredProfileHtml(text) {
    const parsed = parseStructuredProfileText(text);
    if (!parsed) return null;

    const titleHtml = parsed.header
        ? `<div class="profile-card-title">${escapeHtml(parsed.header)}</div>`
        : '';

    const rowsHtml = parsed.rows.map(row => `
        <div class="profile-card-row">
            <div class="profile-card-key">${escapeHtml(row.key)}</div>
            <div class="profile-card-value">${escapeHtml(row.value)}</div>
        </div>
    `).join('');

    return `
        <div class="profile-card">
            ${titleHtml}
            <div class="profile-card-grid">
                ${rowsHtml}
            </div>
        </div>
    `;
}

function appendMessage(text, sender, relatedQ = null) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    const isUser = sender === 'user';

    div.className = isUser ? 'flex gap-3 flex-row-reverse animate-fade-in-up' : 'flex gap-3 animate-fade-in-up';

    const avatar = isUser
        ? `<div class="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-600 overflow-hidden shrink-0 mt-1"><span class="material-icons text-gray-500 w-full h-full flex items-center justify-center">person</span></div>`
        : `<div class="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1"><span class="material-icons text-primary text-sm">smart_toy</span></div>`;

    const itemsBg = isUser ? 'bg-primary text-white' : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 border border-gray-100 dark:border-gray-600 markdown-body';

    let feedback = '';
    const feedbackId = 'fb-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5);
    if (!isUser && relatedQ) {
        feedback = `
        <div class="flex gap-2 mt-1 ml-1 feedback-buttons" data-feedback-id="${feedbackId}" data-question="${encodeURIComponent(relatedQ)}" data-answer="${encodeURIComponent(text)}">
            <button data-rating="positive" class="text-gray-300 hover:text-green-500 p-1 cursor-pointer"><span class="material-icons text-[14px]">thumb_up</span></button>
            <button data-rating="negative" class="text-gray-300 hover:text-red-500 p-1 cursor-pointer"><span class="material-icons text-[14px]">thumb_down</span></button>
        </div>`;
    }

    const structuredProfileHtml = !isUser ? buildStructuredProfileHtml(text) : null;
    let safeMessageHtml = `<p>${escapeHtml(text)}</p>`;

    // Parse Markdown Links [text](url) -> <a href="url" target="_blank">text</a>
    if (!isUser && !structuredProfileHtml) {
        safeMessageHtml = safeMessageHtml.replace(
            /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a href="$2" target="_blank" class="text-primary hover:underline font-bold inline-flex items-center gap-0.5">$1<span class="material-icons text-[12px]">open_in_new</span></a>'
        );
    }

    const contentHtml = structuredProfileHtml || safeMessageHtml;

    div.innerHTML = `${avatar}<div class="flex flex-col gap-1 max-w-[80%]"><div class="${itemsBg} p-3 rounded-2xl ${isUser ? 'rounded-tr-none' : 'rounded-tl-none'} shadow-sm text-sm">${contentHtml}</div>${feedback}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function appendLoading() {
    const c = document.getElementById('chat-messages');
    const id = 'l-' + Date.now();
    const d = document.createElement('div');
    d.id = id;
    d.className = 'flex gap-3 animate-fade-in-up';
    d.innerHTML = `<div class="w-8 h-8 rounded-full bg-primary/10 flex center shrink-0 mt-1"><span class="material-icons text-primary text-sm">smart_toy</span></div><div class="bg-white dark:bg-gray-700 p-3 rounded-2xl rounded-tl-none border border-gray-100 w-12 flex justify-center gap-1"><div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div><div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce delay-75"></div><div class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce delay-150"></div></div>`;
    c.appendChild(d);
    c.scrollTop = c.scrollHeight;
    return id;
}

// Event Delegation for Feedback Buttons
document.getElementById('chat-messages').addEventListener('click', async function (e) {
    const btn = e.target.closest('[data-rating]');
    if (!btn) return;

    const container = btn.closest('.feedback-buttons');
    if (!container) return;

    const rating = btn.getAttribute('data-rating');
    const u = container.getAttribute('data-question');
    const a = container.getAttribute('data-answer');

    container.innerHTML = `<span class="text-xs text-gray-400 ml-1">Thanks! ${rating === 'positive' ? '👍' : '👎'}</span>`;

    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${jwtToken}` },
            body: JSON.stringify({ user_message: decodeURIComponent(u), ai_response: decodeURIComponent(a), rating: rating })
        });
    } catch (e) { console.error('Feedback error:', e); }
});
