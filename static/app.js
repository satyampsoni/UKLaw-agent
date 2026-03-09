/* ═══════════════════════════════════════════════════════════════
   UK LawAssistant — Frontend Logic
   Vanilla JS — no framework, no build step.
   Talks to the FastAPI backend at /api/*.
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = '';  // Same origin — no CORS issues

// ── State ────────────────────────────────────────────────────────
let isLoading = false;
let conversationHistory = [];

// ── DOM references ───────────────────────────────────────────────
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const actFilter = document.getElementById('act-filter');
const strictMode = document.getElementById('strict-mode');
const answerArea = document.getElementById('answer-area');
const answerContent = document.getElementById('answer-content');
const answerMeta = document.getElementById('answer-meta');
const sourcesList = document.getElementById('sources-list');
const sourceCount = document.getElementById('source-count');
const historyDiv = document.getElementById('history');
const searchInput = document.getElementById('search-input');

// ── Initialisation ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadStats();

    // Enter key to submit
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            searchLegislation();
        }
    });
});

// ── Load header stats ────────────────────────────────────────────
async function loadStats() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        document.getElementById('stat-acts').textContent = data.documents;
        document.getElementById('stat-nodes').textContent = data.nodes.toLocaleString();
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

// ── Ask Question (main RAG) ──────────────────────────────────────
async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question || isLoading) return;

    setLoading(true);

    const body = {
        question: question,
        document_id: actFilter.value || null,
        strict_mode: strictMode.checked,
    };

    try {
        const resp = await fetch(`${API_BASE}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        // Move current answer to history (if exists)
        if (answerContent.innerHTML && conversationHistory.length > 0) {
            pushToHistory(
                conversationHistory[conversationHistory.length - 1].question,
                conversationHistory[conversationHistory.length - 1].answerHtml
            );
        }

        // Display new answer
        renderAnswer(data);
        renderSources(data.sources);

        // Track in conversation
        conversationHistory.push({
            question: question,
            answerHtml: answerContent.innerHTML,
        });

        // Clear input
        questionInput.value = '';

    } catch (e) {
        renderError(e.message);
    } finally {
        setLoading(false);
    }
}

// ── Search Legislation (no LLM) ─────────────────────────────────
async function searchLegislation() {
    const query = searchInput.value.trim();
    if (!query) return;

    const searchActFilter = document.getElementById('search-act-filter');
    const body = {
        query: query,
        document_id: searchActFilter.value || null,
        limit: 10,
    };

    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '<div class="empty-state">Searching<span class="loading-dots"></span></div>';

    try {
        const resp = await fetch(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (data.sections.length === 0) {
            resultsDiv.innerHTML = '<div class="empty-state">No matching legislation found.</div>';
            return;
        }

        resultsDiv.innerHTML = data.sections.map((sec, i) => `
            <div class="search-result-card" onclick="this.classList.toggle('expanded')">
                <div class="search-result-citation">${escapeHtml(sec.citation)}</div>
                <div class="search-result-title">${escapeHtml(sec.title)}</div>
                <div class="search-result-hierarchy">${escapeHtml(sec.hierarchy)}</div>
                <div class="search-result-text">${escapeHtml(sec.text)}</div>
            </div>
        `).join('');

    } catch (e) {
        resultsDiv.innerHTML = `<div class="empty-state" style="color: var(--error)">Search failed: ${escapeHtml(e.message)}</div>`;
    }
}

// ── Render Answer ────────────────────────────────────────────────
function renderAnswer(data) {
    answerArea.classList.remove('hidden');

    // Render Markdown-like content
    answerContent.innerHTML = renderMarkdown(data.answer);

    // Metrics
    answerMeta.innerHTML = `
        <span>⏱ ${formatMs(data.total_latency_ms)}</span>
        <span>🔍 ${formatMs(data.search_latency_ms)}</span>
        <span>🤖 ${formatMs(data.llm_latency_ms)}</span>
        <span>📊 ${data.tokens_used} tokens</span>
    `;
}

function renderSources(sources) {
    if (!sources || sources.length === 0) {
        sourcesList.innerHTML = '<div class="empty-state">No sources found.</div>';
        sourceCount.textContent = '';
        return;
    }

    sourceCount.textContent = `${sources.length} sections`;

    sourcesList.innerHTML = sources.map((src, i) => `
        <div class="source-card" onclick="this.classList.toggle('expanded')">
            <div class="source-citation">${escapeHtml(src.citation)}</div>
            <div class="source-title">${escapeHtml(src.title)}</div>
            <div class="source-hierarchy">${escapeHtml(src.hierarchy)}</div>
            <div class="source-score">Relevance: ${src.score.toFixed(2)}</div>
        </div>
    `).join('');
}

function renderError(message) {
    answerArea.classList.remove('hidden');
    answerContent.innerHTML = `
        <div style="color: var(--error); padding: 1rem; background: rgba(239,68,68,0.1); border-radius: var(--radius);">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
    answerMeta.innerHTML = '';
}

function pushToHistory(question, answerHtml) {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
        <div class="history-question">Q: ${escapeHtml(question)}</div>
        <div class="history-answer">${answerHtml}</div>
    `;
    historyDiv.prepend(item);
}

// ── Loading state ────────────────────────────────────────────────
function setLoading(loading) {
    isLoading = loading;
    askBtn.disabled = loading;
    askBtn.querySelector('.btn-text').classList.toggle('hidden', loading);
    askBtn.querySelector('.btn-spinner').classList.toggle('hidden', !loading);

    if (loading) {
        answerArea.classList.remove('hidden');
        answerContent.innerHTML = `
            <div style="color: var(--text-muted); padding: 2rem; text-align: center;">
                Searching legislation and generating answer<span class="loading-dots"></span>
            </div>
        `;
        answerMeta.innerHTML = '';
    }
}

// ── Simple Markdown renderer ─────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Headers: ### → h3, #### → h4, ## → h2
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

    // Bold: **text**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic: *text*
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Inline code: `text`
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');

    // Blockquote: > text
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered list items: - text or * text
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');

    // Numbered list items: 1. text
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Horizontal rules: ---
    html = html.replace(/^---$/gm, '<hr>');

    // Paragraphs: double newline
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[234]>)/g, '$1');
    html = html.replace(/(<\/h[234]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<hr>)/g, '$1');
    html = html.replace(/(<hr>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<blockquote>)/g, '$1');
    html = html.replace(/(<\/blockquote>)\s*<\/p>/g, '$1');

    return html;
}

// ── Helpers ──────────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMs(ms) {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}
