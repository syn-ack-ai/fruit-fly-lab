/*
 * Run a scenario through web/js/engine.js and dump the full final state
 * (v, g, spike counts) plus per-step spike totals, for native/verify_native.py.
 * Usage: node native/dump_js.mjs <scenario.json> <out.bin>
 */
import fs from 'node:fs';
import { LIFEngine, decodeConnectome } from '../web/js/engine.js';

const [, , scenarioPath, outPath] = process.argv;
const sc = JSON.parse(fs.readFileSync(scenarioPath, 'utf8'));
const D = new URL('../web/data/', import.meta.url);
const rd = p => { const b = fs.readFileSync(new URL(p, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const meta = JSON.parse(fs.readFileSync(new URL('meta.json', D)));
const eng = new LIFEngine(decodeConnectome(rd('connectome.bin'), meta.n, meta.nnz), undefined, sc.seed);
if (sc.silence.length) eng.silence(sc.silence, true);
for (const [i, val] of sc.inject_g || []) eng.g[i] = val;
if (sc.poisson_idx.length) eng.setPoisson(sc.poisson_idx, sc.poisson_rate);
const steps = Math.round(sc.duration_ms / 0.1);
const per = new Int32Array(steps);
const t0 = performance.now();
for (let s = 0; s < steps; s++) per[s] = eng.step().length;
console.error(`[js] ${steps} steps in ${((performance.now() - t0) / 1000).toFixed(2)} s`);
const out = Buffer.concat([eng.v, eng.g, eng.spikeCounts, per].map(a => Buffer.from(a.buffer, a.byteOffset, a.byteLength)));
fs.writeFileSync(outPath, out);
