// Deterministic checks for hookhaus-v2-copy.json: structure from CAMPAIGN-V2,
// the blader/humanizer patterns a script can catch, and no invented numbers.
// Run: node --test campaign/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const copy = JSON.parse(readFileSync(new URL('./hookhaus-v2-copy.json', import.meta.url), 'utf8'));
const texts = [
  ['bio', copy.bio],
  ...copy.pins.flatMap((p) => [[`${p.id}.onScreen`, p.onScreen], [`${p.id}.caption`, p.caption]]),
  ...copy.hooks.flatMap((h, i) => [[`hook${i + 1}.onScreen`, h.onScreen], [`hook${i + 1}.caption`, h.caption]]),
];

// Numbers already set by Sorin in CAMPAIGN.md / CAMPAIGN-V2.md, or plain formats.
const ALLOWED_NUMBERS = new Set(['16:9', '9:16', '25', '150']);
const AI_WORDS = /\b(delve|pivotal|landscape|testament|seamless(ly)?|elevate|unlock|game[- ]?changer|supercharge|leverage|vibrant|stunning|nestled|crucial|robust|transformative|journey|realm|tapestry|dive in|let's dive|here's the thing|the real question|at its core)\b/i;

test('structure matches CAMPAIGN-V2: bio, 3 pins, 10 hooks', () => {
  assert.equal(copy.pins.length, 3);
  assert.deepEqual(copy.pins.map((p) => p.format), ['before-after', 'prices', 'free-sample']);
  assert.equal(copy.hooks.length, 10);
  for (const h of copy.hooks) assert.ok(['before-after', 'process', 'free-sample'].includes(h.format));
});

test('bio fits the TikTok 80-character limit and has a DM-style call to action', () => {
  assert.ok(copy.bio.length <= 80, `bio is ${copy.bio.length} chars`);
  assert.match(copy.bio, /send me/i);
});

test('no em dashes, curly quotes, emojis or AI vocabulary', () => {
  for (const [id, t] of texts) {
    assert.doesNotMatch(t, /[—–]/, `${id}: dash`);
    assert.doesNotMatch(t, /[‘’“”]/, `${id}: curly quote`);
    assert.doesNotMatch(t, /\p{Extended_Pictographic}/u, `${id}: emoji`);
    assert.doesNotMatch(t, AI_WORDS, `${id}: AI vocabulary`);
    assert.doesNotMatch(t, /\bnot (just|only)\b.*\bbut\b/i, `${id}: not-X-but-Y`);
  }
});

test('only numbers already decided in the campaign docs', () => {
  for (const [id, t] of texts) {
    for (const n of t.match(/\d+(?::\d+)?/g) ?? []) {
      assert.ok(ALLOWED_NUMBERS.has(n), `${id}: unapproved number ${n}`);
    }
  }
});

test('no performance claims (views, clients, results)', () => {
  for (const [id, t] of texts) {
    assert.doesNotMatch(t, /\b(views|followers|clients|guaranteed|viral|went viral|results)\b/i, id);
  }
});

test('onScreen hooks are short enough to read in two seconds', () => {
  for (const h of copy.hooks) {
    const words = h.onScreen.split(/\s+/).length;
    assert.ok(words <= 12, `"${h.onScreen}" has ${words} words`);
  }
});

test('3 to 5 hashtags per post, lowercase, no #fyp', () => {
  for (const p of [...copy.pins, ...copy.hooks]) {
    assert.ok(p.hashtags.length >= 3 && p.hashtags.length <= 5);
    for (const tag of p.hashtags) {
      assert.match(tag, /^#[a-z]+$/);
      assert.ok(!['#fyp', '#foryou'].includes(tag));
    }
  }
});
