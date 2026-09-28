// Run with: node --test tests/test_web_worker.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

test('worker speed includes the current frame and restarts after reset', async (t) => {
  const frames = [];
  let nextTick;
  let ready, failed;
  const booted = new Promise((resolve, reject) => { ready = resolve; failed = reject; });
  globalThis.self = {
    postMessage(message) {
      if (message.type === 'ready') ready();
      if (message.type === 'error') failed(new Error(message.message));
      if (message.type === 'frame') frames.push(message.frame);
    },
  };
  t.after(() => { delete globalThis.self; });
  t.mock.method(globalThis, 'fetch', async (url) => new Response(await readFile(new URL(url))));
  t.mock.method(globalThis, 'setTimeout', (callback) => { nextTick = callback; return 1; });
  t.mock.method(globalThis, 'clearTimeout', () => {});
  let wall = 0;
  t.mock.method(performance, 'now', () => { wall += 20; return wall; });
  await import('../web/js/worker.js');
  self.onmessage({ data: { cmd: 'boot' } });
  await booted;

  self.onmessage({ data: { cmd: 'play' } });
  nextTick();
  self.onmessage({ data: { cmd: 'reset' } });
  self.onmessage({ data: { cmd: 'play' } });
  self.onmessage({ data: { cmd: 'pause' } });

  assert.deepEqual(frames.map(f => f.t_ms), [2, 4, 2]);
  for (const frame of frames) assert.equal(frame.realtime_factor, 0.1);
});
