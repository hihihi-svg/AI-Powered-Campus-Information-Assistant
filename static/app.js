const form = document.getElementById('ask-form');
const input = document.getElementById('question-input');
const button = document.getElementById('ask-button');
const clearButton = document.getElementById('clear-button');
const strategySelect = document.getElementById('strategy-select');
const loadingIndicator = document.getElementById('loading-indicator');
const chatMessages = document.getElementById('chat-messages');
const suggestionButtons = document.querySelectorAll('.suggestion-button');

let isSubmitting = false;

function setLoadingState(isLoading) {
  isSubmitting = isLoading;
  button.disabled = isLoading;
  input.disabled = isLoading;
  if (strategySelect) {
    strategySelect.disabled = isLoading;
  }
  if (loadingIndicator) {
    loadingIndicator.style.display = isLoading ? 'inline-flex' : 'none';
  }
  if (isLoading) {
    button.textContent = 'Searching...';
  } else {
    button.textContent = 'Ask';
  }
}

function addUserMessage(text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message user';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  bubble.appendChild(paragraph);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

function addAssistantMessage(text, options = {}) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  // Badges Header: Status & Retrieval Strategy
  const metaHeader = document.createElement('div');
  metaHeader.className = 'message-meta-header';

  if (options.status) {
    const statusBadge = document.createElement('span');
    statusBadge.className = `badge ${options.status.type === 'success' ? 'badge-status-answered' : 'badge-status-insufficient'}`;
    statusBadge.textContent = options.status.text;
    metaHeader.appendChild(statusBadge);
  }

  if (options.strategy) {
    const stratBadge = document.createElement('span');
    stratBadge.className = 'badge badge-strategy';
    stratBadge.textContent = `Strategy: ${options.strategy.toUpperCase()}`;
    metaHeader.appendChild(stratBadge);
  }

  if (metaHeader.hasChildNodes()) {
    bubble.appendChild(metaHeader);
  }

  // Answer Text
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  bubble.appendChild(paragraph);

  // Sources Section (Document Title + Clickable URLs)
  if (options.sources && options.sources.length > 0) {
    const sourceList = document.createElement('div');
    sourceList.className = 'source-list';

    const sourceHeading = document.createElement('div');
    sourceHeading.className = 'meta';
    sourceHeading.style.fontWeight = '700';
    sourceHeading.textContent = 'Sources:';
    sourceList.appendChild(sourceHeading);

    options.sources.forEach((source) => {
      const item = document.createElement('div');
      item.className = 'source-item';

      const link = document.createElement('a');
      link.className = 'source-link';
      link.textContent = source.title || 'Source Document';

      const safeUrl = validateUrl(source.url);
      if (safeUrl) {
        link.href = safeUrl;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
      } else {
        link.href = '#';
        link.setAttribute('aria-disabled', 'true');
      }

      item.appendChild(link);

      if (source.heading) {
        const heading = document.createElement('div');
        heading.className = 'meta';
        heading.textContent = source.heading;
        item.appendChild(heading);
      }

      if (safeUrl) {
        const urlText = document.createElement('div');
        urlText.className = 'meta';
        urlText.textContent = safeUrl;
        item.appendChild(urlText);
      }

      sourceList.appendChild(item);
    });

    bubble.appendChild(sourceList);
  }

  // Collapsible Diagnostic Evidence Section (Evidence passages + Relevance scores)
  if (options.evidence && options.evidence.length > 0) {
    const accordion = document.createElement('details');
    accordion.className = 'evidence-accordion';

    const summary = document.createElement('summary');
    summary.className = 'evidence-summary';
    summary.textContent = `Retrieved Evidence Passages (${options.evidence.length}) & Relevance Scores`;
    accordion.appendChild(summary);

    const container = document.createElement('div');
    container.className = 'evidence-container';

    options.evidence.forEach((item) => {
      const evidenceItem = document.createElement('div');
      evidenceItem.className = 'evidence-item';

      const evidenceText = document.createElement('p');
      evidenceText.textContent = item.text || 'Evidence snippet.';
      evidenceItem.appendChild(evidenceText);

      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = `${item.title ? item.title + ' — ' : ''}${item.heading || 'Campus document'}`;
      evidenceItem.appendChild(meta);

      if (item.relevance_score !== undefined && item.relevance_score !== null) {
        const scoreTag = document.createElement('span');
        scoreTag.className = 'evidence-score-tag';
        scoreTag.textContent = `Relevance Score: ${Number(item.relevance_score).toFixed(4)}`;
        evidenceItem.appendChild(scoreTag);
      }

      container.appendChild(evidenceItem);
    });

    accordion.appendChild(container);
    bubble.appendChild(accordion);
  }

  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

function renderError(message) {
  addAssistantMessage(message, {
    status: {
      type: 'warning',
      text: 'Request failed',
    },
  });
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function validateUrl(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }

  try {
    const url = new URL(value);
    if (url.protocol === 'http:' || url.protocol === 'https:') {
      return url.toString();
    }
  } catch (error) {
    return null;
  }

  return null;
}

function renderReply(payload, requestedStrategy) {
  const answer = payload && typeof payload.answer === 'string' ? payload.answer : 'No answer was returned.';
  const sources = Array.isArray(payload && payload.sources) ? payload.sources : [];
  const evidence = Array.isArray(payload && payload.evidence) ? payload.evidence : [];
  const abstained = Boolean(payload && payload.abstention_status);
  const strategy = (payload && payload.strategy) || requestedStrategy || 'tfidf';

  if (abstained) {
    addAssistantMessage(answer, {
      sources: sources,
      evidence: evidence,
      strategy: strategy,
      status: {
        type: 'warning',
        text: 'Insufficient Evidence',
      },
    });
    return;
  }

  addAssistantMessage(answer, {
    sources: sources,
    evidence: evidence,
    strategy: strategy,
    status: {
      type: 'success',
      text: 'Answered',
    },
  });
}

async function askQuestion(question) {
  if (isSubmitting || typeof question !== 'string') {
    return;
  }

  const trimmedQuestion = question.trim();
  if (!trimmedQuestion) {
    renderError('Please enter a question before submitting.');
    return;
  }

  const strategy = strategySelect ? strategySelect.value : 'tfidf';
  const rerank = strategy === 'hybrid_rerank';

  addUserMessage(trimmedQuestion);
  setLoadingState(true);

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: trimmedQuestion,
        strategy: strategy,
        rerank: rerank,
      }),
    });

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}`;
      try {
        const errorPayload = await response.json();
        if (errorPayload && errorPayload.detail) {
          errorDetail = errorPayload.detail;
        }
      } catch (_) {}
      renderError(`Server error (${errorDetail}): Unable to answer question.`);
      return;
    }

    const payload = await response.json();
    renderReply(payload, strategy);
  } catch (error) {
    renderError('Unable to reach the campus assistant service. Please verify that the API server is running at http://127.0.0.1:8000.');
  } finally {
    setLoadingState(false);
    input.value = '';
    input.focus();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  askQuestion(input.value);
});

clearButton.addEventListener('click', () => {
  chatMessages.innerHTML = `
    <div class="message assistant">
      <div class="message-bubble">
        <p>Hello! Ask me a question about RVCE admissions, procedures, or campus information.</p>
      </div>
    </div>
  `;
  input.value = '';
  input.focus();
});

suggestionButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const question = button.dataset.question || '';
    if (question) {
      input.value = question;
      input.focus();
      askQuestion(question);
    }
  });
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    askQuestion(input.value);
  }
});

if (input) {
  input.focus();
}
