/** Session-scoped detail loading. Selection changes are intentionally not scope changes. */
export function createDetailLoader({getScope, fetchDetail, onDetail, onError}) {
  const cache = new Map(), inFlight = new Map();
  const key = (session, id) => JSON.stringify([session, id]);
  const current = scope => {
    const now = getScope();
    return now.mode === 'live' && now.sessionId === scope.sessionId && now.generation === scope.generation;
  };
  return {
    get: (session, id) => cache.get(key(session, id)),
    clear: () => cache.clear(),
    load(id) {
      const scope = {...getScope()};
      if(scope.mode !== 'live' || !scope.sessionId) return Promise.resolve(false);
      const requestKey = JSON.stringify([scope.sessionId, scope.generation, id]);
      // A poll and an export may request the same branch concurrently. Share
      // that response instead of letting out-of-order requests overwrite it.
      if(inFlight.has(requestKey)) return inFlight.get(requestKey);
      const pending = (async () => {
        try {
          const detail = await fetchDetail(id);
          if(!current(scope)) return false;
          cache.set(key(scope.sessionId, id), detail);
          onDetail(id, detail);
          return true;
        } catch(error) {
          if(current(scope)) onError(error);
          return false;
        } finally { inFlight.delete(requestKey); }
      })();
      inFlight.set(requestKey, pending);
      return pending;
    },
  };
}

export function liveBody(detail) {
  if(typeof detail.output === 'string' && detail.output.trim()) {
    return {body: detail.output, bodySource: 'Full live output · model-generated'};
  }
  const assistantText = (detail.messages ?? [])
    .filter(message => message.role === 'assistant' && typeof message.content === 'string' && message.content.trim())
    .map(message => message.content);
  if(assistantText.length) return {
    body: assistantText.join('\n\n'),
    bodySource: 'Full assistant conversation text · no final output yet',
  };
  return {body: '', bodySource: 'No final output or assistant text yet'};
}
