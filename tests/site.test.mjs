import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = name => fs.readFileSync(path.join(ROOT, name), 'utf8');
const html = read('index.html');
const attrs = (source, name) => [...source.matchAll(new RegExp(`${name}="([^"]+)"`, 'g'))].map(m => m[1].replaceAll('&amp;', '&'));

test('every local href and src points to an existing file', () => {
  for (const page of ['index.html', 'credits.html']) {
    for (const ref of [...attrs(read(page), 'href'), ...attrs(read(page), 'src')]) {
      if (/^(#|mailto:|https?:)/.test(ref)) continue;
      const file = ref.split(/[?#]/)[0];
      assert.ok(fs.existsSync(path.join(ROOT, file)), `${page}: missing ${file}`);
    }
  }
});

test('every in-page anchor has a target', () => {
  const ids = new Set(attrs(html, 'id'));
  for (const ref of attrs(html, 'href').filter(r => r.startsWith('#') && r.length > 1)) {
    assert.ok(ids.has(ref.slice(1)), `missing anchor ${ref}`);
  }
});

test('packages match docs/pricing.md and each has a prefilled inquiry', () => {
  const section = html.match(/<section class="packages-section[\s\S]*?<\/section>/)?.[0];
  assert.ok(section, 'packages section missing');
  const prices = [...section.matchAll(/class="package-price">([^<]+)</g)].map(m => m[1]);
  assert.deepEqual(prices, ['149 €', '390 €', '790 €', '1,690 €']);
  const links = attrs(section, 'href');
  assert.equal(links.length, 4);
  for (const link of links) {
    const url = new URL(link);
    assert.equal(url.protocol, 'mailto:');
    assert.match(url.searchParams.get('subject'), /^HookHaus /);
    assert.match(url.searchParams.get('body'), /Link to footage/);
  }
});
