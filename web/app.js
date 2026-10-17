/* Same-origin UI: render user/model data only via textContent, never HTML. */
(() => {
  'use strict';
  let token = '';
  let conversationId = null;
  let providers = [];
  const $ = id => document.getElementById(id);
  const notice = (message) => { $('notice').textContent = message || ''; $('notice').hidden = !message; };
  const node = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  };
  async function api(path, method = 'GET', body) {
    const response = await fetch(path, {
      method,
      headers: { Authorization: 'Bearer ' + token, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const result = await response.json();
    if (!response.ok) {
      const error = new Error(result.error || 'Ошибка API (' + response.status + ')');
      error.conversationId = result.conversation_id || null;
      throw error;
    }
    return result;
  }
  function tab(id) {
    document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === id));
    document.querySelectorAll('[data-tab]').forEach(el => el.classList.toggle('active', el.dataset.tab === id));
    notice('');
  }
  document.querySelectorAll('[data-tab]').forEach(el => el.addEventListener('click', () => { tab(el.dataset.tab); if (el.dataset.tab === 'approvals') refreshApprovals(); }));
  const dialog = $('loginDialog');
  $('lockButton').addEventListener('click', () => { token = ''; dialog.showModal(); });
  $('loginForm').addEventListener('submit', async event => {
    event.preventDefault();
    const newToken = $('accessToken').value.trim();
    token = newToken;
    try {
      await api('/api/status');
      $('accessToken').value = ''; $('loginError').textContent = '';
      dialog.close();
      await refreshAll();
    } catch (error) {
      token = ''; $('loginError').textContent = error.message;
    }
  });
  dialog.showModal();
  async function refreshAll() {
    const [status, result, history] = await Promise.all([api('/api/status'), api('/api/providers'), api('/api/conversations')]);
    $('systemStatus').textContent = 'Версия ' + status.version + ' · подключено: ' + status.providers;
    $('approvalCount').textContent = status.approvals_pending;
    providers = result.providers;
    const currentProvider = $('modelSelect').value;
    $('modelSelect').replaceChildren();
    for (const p of providers) {
      const opt = node('option', '', p.name + ' · ' + p.model);
      opt.value = p.id; $('modelSelect').append(opt);
    }
    if (providers.some(p => p.id === currentProvider)) $('modelSelect').value = currentProvider;
    const current = conversationId;
    $('conversationSelect').replaceChildren();
    const option = node('option', '', 'Новый диалог'); option.value = ''; $('conversationSelect').append(option);
    for (const c of history.conversations) {
      const opt = node('option', '', c.title); opt.value = c.id; $('conversationSelect').append(opt);
    }
    if (current && history.conversations.some(c => c.id === current)) $('conversationSelect').value = current;
    renderProviders();
    await refreshApprovals();
    if (current && $('conversationSelect').value === current) await loadMessages();
    else { conversationId = null; renderMessages([]); }
  }
  function renderProviders() {
    const container = $('providerList'); container.replaceChildren();
    if (!providers.length) { container.append(node('p', 'muted', 'Провайдеров пока нет. Добавьте первый справа.')); return; }
    for (const p of providers) {
      const item = node('div', 'listItem');
      const info = node('div'); info.append(node('strong', '', p.name), node('small', '', p.kind + ' · ' + p.model), node('small', '', p.base_url));
      const remove = node('button', 'danger ghost', 'Удалить'); remove.type = 'button';
      remove.addEventListener('click', async () => {
        if (!confirm('Удалить провайдера «' + p.name + '»?')) return;
        try { await api('/api/providers/' + p.id, 'DELETE'); await refreshAll(); } catch (error) { notice(error.message); }
      });
      item.append(info, remove); container.append(item);
    }
  }
  $('kindSelect').addEventListener('change', event => {
    $('baseUrl').value = event.target.value === 'openai' ? 'https://api.openai.com/v1' : event.target.value === 'ollama' ? 'http://127.0.0.1:11434' : '';
  });
  $('providerForm').addEventListener('submit', async event => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = Object.fromEntries(new FormData(form).entries());
    try { await api('/api/providers', 'POST', data); form.reset(); await refreshAll(); tab('chat'); }
    catch (error) { notice(error.message); }
  });
  function renderMessages(messages) {
    const box = $('messages'); box.replaceChildren();
    if (!messages.length) { box.append(node('div', 'empty', 'Ваш AI-агент готов к диалогу. Выберите модель и отправьте сообщение.')); return; }
    for (const item of messages) {
      const block = node('article', 'bubble ' + (item.role === 'user' ? 'user' : 'assistant'));
      block.append(node('div', 'role', item.role === 'user' ? 'Вы' : 'AI-агент'), node('div', 'content', item.content));
      box.append(block);
    }
    box.scrollTop = box.scrollHeight;
  }
  async function loadMessages() {
    if (!conversationId) { renderMessages([]); return; }
    const res = await api('/api/conversations/' + conversationId + '/messages');
    renderMessages(res.messages);
  }
  $('conversationSelect').addEventListener('change', async event => {
    conversationId = event.target.value || null;
    try { await loadMessages(); } catch (error) { notice(error.message); }
  });
  $('newChat').addEventListener('click', () => { conversationId = null; $('conversationSelect').value = ''; renderMessages([]); });
  $('chatForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (!$('modelSelect').value) return notice('Сначала добавьте AI-провайдера.');
    const message = $('messageInput').value.trim(); if (!message) return;
    $('sendButton').disabled = true;
    $('sendButton').textContent = 'Отправка…'; notice('');
    const id = conversationId;
    try {
      const result = await api('/api/chat', 'POST', { provider_id: $('modelSelect').value, message, ...(id ? { conversation_id: id } : {}) });
      conversationId = result.conversation_id;
      $('messageInput').value = '';
      await refreshAll();
    } catch (error) {
      if (error.conversationId) conversationId = error.conversationId;
      notice(error.message);
      await refreshAll().catch(() => {});
    } finally { $('sendButton').disabled = false; $('sendButton').textContent = 'Отправить ↗'; }
  });
  async function refreshApprovals() {
    const result = await api('/api/approvals');
    const target = $('approvalList'); target.replaceChildren();
    if (!result.approvals.length) target.append(node('p', 'muted', 'Нет заявок на действия.'));
    for (const item of result.approvals) {
      const box = node('article', 'panel approval');
      box.append(node('div', 'eyebrow', item.status), node('h2', '', item.action + ' → ' + item.target), node('p', '', item.reason));
      if (item.status === 'pending') {
        const buttons = node('div', 'actions');
        for (const [decision, label] of [['approve', 'Разрешить (не выполнять)'], ['reject', 'Отклонить']]) {
          const button = node('button', decision === 'reject' ? 'ghost' : '', label);
          button.addEventListener('click', async () => {
            try { await api('/api/approvals/' + item.id + '/decision', 'POST', { decision }); await refreshAll(); }
            catch (error) { notice(error.message); }
          });
          buttons.append(button);
        }
        box.append(buttons);
      }
      target.append(box);
    }
  }
})();
