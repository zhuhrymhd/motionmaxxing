#!/usr/bin/env node
// score.mjs - NATIVE music + sound effects for a film, synthesized locally (no ElevenLabs, no samples, no licences).
//
// Two steps, so the score is decided on purpose and can be reviewed before it is rendered:
//
//   node score.mjs plan   --events out.mp4.events.json --duration 30.3 --hero 660 [--mood product] [--roles hook,proof,...]
//                         [--vo-wav audio/vo_only.wav | --vo-regions "0.3-2.3,4.2-8.4"] [--fps 30] [--out audio/score.json]
//       -> audio/score.json: BPM + bar offset solved so the hero hit and the closing cut land on downbeats, one section per
//          act (from the film's cuts) with an energy arc by role, a drop on the hero, a stinger on the close, the VO's
//          speech regions (no melodic accents over speech), and every SFX placed on its picture event (transient aligned).
//          Every decision carries a "why". Read it, edit it (mood, energy, accents, sfx), then render.
//
//   node score.mjs render audio/score.json [--music-out audio/music.native.wav] [--sfx-out audio/sfx.native.wav] [--stems]
//       -> film-length WAVs (48 kHz stereo) that start at film time 0, so they mix at @0:
//          python3 mix.py --duration D --out mix.wav --vo ... --music audio/music.native.wav --music-gain -17 --duck --sfx audio/sfx.native.wav@0:0
//          music is mastered to -14 LUFS / -1 dBTP (ffmpeg two-pass loudnorm); the SFX stem is left at its authored gains.
//
//   node score.mjs moods          list the moods and what they are for
//
// score.json (all times in FILM seconds):
//   { "fps": 30, "duration": 30.3, "bpm": 96, "offset": -0.5,        // offset = film time of a bar-1 downbeat (<= 0)
//     "mood": "product", "progression": ["Cmaj7","G","Am7","Fmaj7"], "transpose": 0, "seed": 7, "fade_out": 0.8,
//     "sections": [{ "name": "hook", "role": "hook", "from": 0, "to": 9.0, "energy": 0.3, "why": "..." }, ...],
//     "accents":  [{ "t": 22.0, "kind": "drop", "stop": true, "why": "hero" }, { "t": 27.0, "kind": "stinger", "why": "close" }],
//     "vo":       { "regions": [[0.3, 2.3], ...], "lead_duck_db": 6 },
//     "sfx":      [{ "t": 13.8, "kind": "tick", "gain": -6, "why": "gE lands" }, ...] }
// Section energy decides which layers play (>=0.15 chords, >=0.3 hats/shaker, >=0.45 kick+bass, >=0.6 snare, lead from the
// mood's threshold) and how loud the section sits. Accent kinds: drop (riser into t, impact on t; "stop": true silences the
// music for the beat before t), stinger (final chord + soft impact), lift (riser only), hit (impact only).
// SFX kinds: tick click pop type whoosh swish lock thud impact ding chime focus sparkle paper glitch riser downsweep reverse.
//
// Deterministic (seeded). Node 18+, no dependencies. ffmpeg is used only for loudness mastering (skipped if missing).
// Engine ported from the x-pipeline music engine (motion/audio/music.mjs) and extended: film-time grid, product/playful/
// tense moods, voice-led chords, humanised timing and velocity, VO-aware melody, event-driven SFX.

import fs from "node:fs";
import path from "node:path";
import { execFileSync, spawnSync } from "node:child_process";

const SR = 48000;
const TAU = Math.PI * 2;

// ---------- utilities -------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const mtof = (m) => 440 * Math.pow(2, (m - 69) / 12);
const dbToGain = (db) => Math.pow(10, db / 20);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const die = (msg, code = 1) => { console.error("error: " + msg); process.exit(code); };

class Bus {
  constructor(n) { this.L = new Float32Array(n); this.R = new Float32Array(n); this.n = n; }
  add(start, mono, gain = 1, pan = 0) {
    const a = (clamp(pan, -1, 1) + 1) * Math.PI / 4, gl = Math.cos(a) * gain, gr = Math.sin(a) * gain;
    const s0 = Math.round(start);
    for (let i = 0; i < mono.length; i++) {
      const j = s0 + i;
      if (j < 0) continue;
      if (j >= this.n) break;
      this.L[j] += mono[i] * gl; this.R[j] += mono[i] * gr;
    }
  }
  addStereo(start, L, R, gain = 1) {
    const s0 = Math.round(start);
    for (let i = 0; i < L.length; i++) {
      const j = s0 + i;
      if (j < 0) continue;
      if (j >= this.n) break;
      this.L[j] += L[i] * gain; this.R[j] += R[i] * gain;
    }
  }
}

function polyblep(t, dt) {
  if (t < dt) { t /= dt; return t + t - t * t - 1; }
  if (t > 1 - dt) { t = (t - 1) / dt; return t * t + t + t + 1; }
  return 0;
}
function osc(wave, phase, dt, pw = 0.5) {
  switch (wave) {
    case "sine": return Math.sin(TAU * phase);
    case "tri": return 1 - 4 * Math.abs(phase - 0.5);
    case "square": {
      let v = phase < pw ? 1 : -1;
      v += polyblep(phase, dt) - polyblep((phase + 1 - pw) % 1, dt);
      return v;
    }
    default: return 2 * phase - 1 - polyblep(phase, dt);   // saw
  }
}

// TPT state-variable filter (Zavalishin), per-sample cutoff.
class SVF {
  constructor() { this.ic1 = 0; this.ic2 = 0; }
  run(x, fc, q = 0.707, mode = "lp") {
    fc = clamp(fc, 20, SR * 0.45);
    const g = Math.tan(Math.PI * fc / SR), k = 1 / q;
    const a1 = 1 / (1 + g * (g + k)), a2 = g * a1, a3 = g * a2;
    const v3 = x - this.ic2, v1 = a1 * this.ic1 + a2 * v3, v2 = this.ic2 + a2 * this.ic1 + a3 * v3;
    this.ic1 = 2 * v1 - this.ic1; this.ic2 = 2 * v2 - this.ic2;
    if (mode === "bp") return v1;
    if (mode === "hp") return x - k * v1 - v2;
    return v2;
  }
}

function adsr(t, gate, a, d, s, r) {
  if (t < a) return t / a;
  if (t < a + d) return 1 - (1 - s) * (t - a) / d;
  if (t < gate) return s;
  const rel = t - gate;
  return rel < r ? s * (1 - rel / r) : 0;
}

// ---------- instruments -----------------------------------------------------------------

function synth({ freq, dur, wave = "saw", a = 0.005, d = 0.15, s = 0.6, r = 0.15,
  cutoff = 2400, cutoffEnv = 0, cutoffDecay = 0.15, q = 0.8, voices = 1, detune = 0, pw = 0.5, sub = 0, vibrato = 0 }) {
  const n = Math.ceil((dur + r) * SR), out = new Float32Array(n), filt = new SVF();
  const phases = Array.from({ length: voices }, (_, v) => (v * 0.37) % 1);
  const ratios = Array.from({ length: voices }, (_, v) => voices === 1 ? 1 : Math.pow(2, (detune * ((v / (voices - 1)) * 2 - 1)) / 1200));
  let subPhase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR, f = freq * (1 + vibrato * Math.sin(TAU * 5.5 * t) * Math.min(1, t / 0.3));
    let x = 0;
    for (let v = 0; v < voices; v++) { const dt = f * ratios[v] / SR; phases[v] = (phases[v] + dt) % 1; x += osc(wave, phases[v], dt, pw); }
    x /= Math.sqrt(voices);
    if (sub) { subPhase = (subPhase + f / 2 / SR) % 1; x += sub * Math.sin(TAU * subPhase); }
    out[i] = filt.run(x, cutoff + cutoffEnv * Math.exp(-t / cutoffDecay), q) * adsr(t, dur, a, d, s, r);
  }
  return out;
}

// Soft electric-piano / felt-keys tone: a few slightly stretched partials, each decaying at its own rate, a soft hammer.
function keys(freq, dur, vel, rnd) {
  const r = 0.25, n = Math.ceil((dur + r) * SR), out = new Float32Array(n);
  const partials = [[1, 1, 1.4], [2.001, 0.42, 2.6], [3.004, 0.16, 4.2], [4.01, 0.07, 6], [0.5, 0.12, 1.1]];
  const ph = partials.map(() => rnd());
  const bright = 0.6 + 0.4 * vel;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    let x = 0;
    for (let k = 0; k < partials.length; k++) {
      const [m, amp, dec] = partials[k];
      ph[k] = (ph[k] + freq * m / SR) % 1;
      x += Math.sin(TAU * ph[k]) * amp * (k > 1 ? bright : 1) * Math.exp(-t * dec);
    }
    if (t < 0.006) x += (rnd() * 2 - 1) * 0.08 * vel * (1 - t / 0.006);
    const rel = t > dur ? Math.max(0, 1 - (t - dur) / r) : 1;
    out[i] = x * Math.min(1, t / 0.003) * rel * 0.55;
  }
  return out;
}

function noiseBurst(secs, rnd, fn) {
  const n = Math.max(1, Math.round(secs * SR)), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(i / SR, rnd() * 2 - 1);
  return out;
}

const DRUMS = {
  kick(rnd, { punch = 1, soft = false } = {}) {
    const n = Math.round(0.45 * SR), out = new Float32Array(n);
    let ph = 0;
    for (let i = 0; i < n; i++) {
      const t = i / SR, f = (soft ? 50 : 46) + (soft ? 70 : 120) * Math.exp(-t * (soft ? 26 : 32));
      ph = (ph + f / SR) % 1;
      let x = Math.sin(TAU * ph) * Math.exp(-t * (soft ? 8 : 6.5));
      if (!soft && t < 0.004) x += (rnd() * 2 - 1) * 0.5 * (1 - t / 0.004) * punch;
      out[i] = Math.tanh(x * (soft ? 1.2 : 1.6));
    }
    return out;
  },
  snare(rnd) {
    const bp = new SVF(); let ph = 0;
    return noiseBurst(0.28, rnd, (t, nz) => { ph = (ph + 185 / SR) % 1; return Math.sin(TAU * ph) * Math.exp(-t * 28) * 0.55 + bp.run(nz, 1900, 0.7, "bp") * Math.exp(-t * 17) * 1.6; });
  },
  clap(rnd) {
    const bp = new SVF();
    return noiseBurst(0.3, rnd, (t, nz) => {
      let env = Math.exp(-Math.max(0, t - 0.022) * 22);
      for (const o of [0, 0.009, 0.018]) if (t >= o && t < o + 0.008) env = Math.max(env, 1 - (t - o) / 0.008);
      return bp.run(nz, 1250, 0.9, "bp") * env * 1.8;
    });
  },
  hat(rnd, { open = false } = {}) {
    const hp = new SVF();
    return noiseBurst(open ? 0.3 : 0.06, rnd, (t, nz) => hp.run(nz, 7500, 0.7, "hp") * Math.exp(-t * (open ? 11 : 70)));
  },
  shaker(rnd) {
    const bp = new SVF();
    return noiseBurst(0.09, rnd, (t, nz) => bp.run(nz, 6200, 0.9, "bp") * Math.min(1, t / 0.012) * Math.exp(-t * 45) * 1.4);
  },
  rim(rnd) {
    const bp = new SVF();
    return noiseBurst(0.08, rnd, (t, nz) => bp.run(nz, 3200, 4, "bp") * Math.exp(-t * 60) * 2 + Math.sin(TAU * 420 * t) * Math.exp(-t * 80) * 0.4);
  },
};

// One palette for accents and SFX. Every sound starts on its transient except the swells (whoosh, riser, reverse, focus):
// SWELL[kind] is where the loudest point sits, as a share of the sound's length.
const SWELL = { whoosh: 0.5, swish: 0.5, riser: 1.0, reverse: 1.0, focus: 0.35 };
function fxSound(kind, rnd, { len = 0, pitch = 1 } = {}) {
  const make = (secs, fn) => { const n = Math.max(1, Math.round(secs * SR)), out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = fn(i / SR, i, secs); return out; };
  switch (kind) {
    case "riser": { const L = len || 1.7, f = new SVF(); let ph = 0;
      return make(L, (t) => { const p = t / L; ph = (ph + (180 + 1400 * p * p) * pitch / SR) % 1; return (f.run(rnd() * 2 - 1, 300 + 9000 * p * p, 1.2, "bp") * 1.4 + 0.25 * (2 * ph - 1)) * p * p; }); }
    case "downsweep": { const L = len || 1.2, f = new SVF();
      return make(L, (t) => f.run(rnd() * 2 - 1, 9000 * Math.exp(-t * 4) + 200, 1, "bp") * Math.exp(-t * 2.2) * 1.3); }
    case "reverse": { const L = len || 1.0, f = new SVF();
      return make(L, (t) => f.run(rnd() * 2 - 1, 6000, 0.7, "hp") * Math.pow(t / L, 3) * 1.2); }
    case "impact": { const L = len || 2.4, f = new SVF(); let ph = 0;
      return make(L, (t) => { ph = (ph + (52 + 60 * Math.exp(-t * 20)) / SR) % 1; return Math.tanh((Math.sin(TAU * ph) * Math.exp(-t * 2.4) + f.run(rnd() * 2 - 1, 5000 * Math.exp(-t * 3) + 300, 0.7) * Math.exp(-t * 3.5) * 0.9) * 1.4); }); }
    case "lock": { let ph = 0; const f = new SVF();     // a soft low "decided" hit: rounded thud + a short dry click on top
      return make(len || 0.9, (t) => { ph = (ph + (62 + 70 * Math.exp(-t * 30)) * pitch / SR) % 1;
        return Math.tanh(Math.sin(TAU * ph) * Math.exp(-t * 5.5) * 1.3 + f.run(rnd() * 2 - 1, 3400 * pitch, 2.2, "bp") * Math.exp(-t * 140) * 0.9); }); }
    case "tick": { const f = new SVF();                  // a needle / detent: tiny metallic click with a short ring
      return make(0.12, (t) => f.run(rnd() * 2 - 1, 4200 * pitch, 6, "bp") * Math.exp(-t * 180) * 1.2 + Math.sin(TAU * 2900 * pitch * t) * Math.exp(-t * 90) * 0.25); }
    case "focus": { const L = len || 0.9, f = new SVF();  // lens focus ring: band-limited whir with a few detents
      return make(L, (t) => { const p = t / L, env = Math.sin(Math.PI * Math.min(1, p / 0.7)) * (p < 0.7 ? 1 : Math.max(0, 1 - (p - 0.7) / 0.3));
        const det = (Math.sin(TAU * 38 * t) > 0.96 ? 1 : 0) * 0.5;
        return (f.run(rnd() * 2 - 1, (1400 + 900 * p) * pitch, 3, "bp") * 0.9 + det * (rnd() * 2 - 1)) * env; }); }
    case "whoosh": { const L = len || 0.45, f = new SVF();
      return make(L, (t) => { const p = t / L, env = Math.sin(Math.PI * p); return f.run(rnd() * 2 - 1, (500 + 5000 * Math.sin(Math.PI * p)) * pitch, 1.6, "bp") * env * 1.8; }); }
    case "swish": return fxSound("whoosh", rnd, { len: len || 0.22, pitch: 1.6 * pitch });
    case "click": return make(0.03, (t) => (Math.sin(TAU * 2600 * pitch * t) * 0.6 + (rnd() * 2 - 1) * 0.4) * Math.exp(-t * 220));
    case "type": return make(0.025, (t) => (rnd() * 2 - 1) * Math.exp(-t * 260) * 0.7 + Math.sin(TAU * 1700 * pitch * t) * Math.exp(-t * 300) * 0.3);
    case "pop": return make(0.12, (t) => Math.sin(TAU * (500 + 1100 * t / 0.12) * pitch * t) * Math.exp(-t * 32));
    case "thud": return make(0.35, (t) => Math.sin(TAU * (70 + 90 * Math.exp(-t * 25)) * t) * Math.exp(-t * 10));
    case "ding": return make(1.4, (t) => (Math.sin(TAU * 1318.5 * pitch * t) + 0.35 * Math.sin(TAU * 2637 * pitch * t)) * Math.exp(-t * 3.2) * 0.6);
    case "chime": return make(1.6, (t) => [523.25, 659.25, 783.99, 1046.5].reduce((s, f, k) => s + (t > k * 0.07 ? Math.sin(TAU * f * pitch * (t - k * 0.07)) * Math.exp(-(t - k * 0.07) * 3) : 0), 0) * 0.35);
    case "glitch": { const L = len || 0.3; let held = 0;
      return make(L, (t, i) => { if (i % 160 === 0) held = (rnd() * 2 - 1) * (rnd() < 0.5 ? 1 : 0.3); return Math.round(held * 4) / 4 * (Math.sin(TAU * 220 * t) > 0 ? 1 : 0.5) * (1 - t / L); }); }
    case "paper": { const L = len || 0.5, f = new SVF();
      return make(L, (t) => f.run(rnd() * 2 - 1, 4200, 0.6, "bp") * (0.5 + 0.5 * Math.sin(TAU * 23 * t)) * Math.sin(Math.PI * t / L) * 1.4); }
    case "sparkle": return make(0.9, (t) => [2093, 2637, 3136].reduce((s, f, k) => s + Math.sin(TAU * f * (t + k * 0.003)) * Math.exp(-t * (5 + k)), 0) * 0.22);
    default: return null;
  }
}
const SFX_KINDS = ["tick", "click", "pop", "type", "whoosh", "swish", "lock", "thud", "impact", "ding", "chime", "focus", "sparkle", "paper", "glitch", "riser", "downsweep", "reverse"];

// ---------- harmony ---------------------------------------------------------------------

const NOTE = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const QUALITY = {
  "": [0, 4, 7], m: [0, 3, 7], "7": [0, 4, 7, 10], m7: [0, 3, 7, 10], maj7: [0, 4, 7, 11], "6": [0, 4, 7, 9], m9: [0, 3, 7, 10, 14],
  sus2: [0, 2, 7], sus4: [0, 5, 7], dim: [0, 3, 6], add9: [0, 4, 7, 14], madd9: [0, 3, 7, 14], maj9: [0, 4, 7, 11, 14],
};
function parseChord(name) {
  const m = /^([A-G])([#b]?)(.*)$/.exec(String(name).trim());
  if (!m || !(m[3] in QUALITY)) die(`unknown chord "${name}" (try Am, F, C, G, Dm7, Fmaj7, Esus4, Cadd9)`, 2);
  const root = NOTE[m[1]] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
  return { root: (root + 12) % 12, intervals: QUALITY[m[3]] };
}
// Voice leading: pick the octave of each chord tone (inside lo..hi) that moves least from the previous voicing.
function voice(chord, key, prev, lo = 55, hi = 74) {
  const pcs = chord.intervals.map((iv) => (chord.root + key + iv) % 12);
  const cand = pcs.map((pc) => { const out = []; for (let m = lo; m <= hi; m++) if ((m % 12 + 12) % 12 === pc) out.push(m); return out; });
  if (!prev) return cand.map((c) => c[0] + (c.length > 1 && c[0] < 57 ? 12 : 0)).sort((a, b) => a - b);
  const center = prev.reduce((s, x) => s + x, 0) / prev.length;
  return cand.map((c) => c.reduce((b, m) => (Math.abs(m - center) < Math.abs(b - center) ? m : b), c[0])).sort((a, b) => a - b);
}

// ---------- moods -----------------------------------------------------------------------

const MOODS = {
  product:   { about: "SaaS / analytics / B2B product film: felt keys, soft pulse bass, shaker, sparse bell; confident, unhurried",
               bpm: [88, 108], progression: ["Cmaj7", "G", "Am7", "Fmaj7"], kick: "half-soft", snare: "rim", hats: "shaker", bass: "pulse8",
               chords: "keys", lead: "bell", leadMin: 0.65, sidechain: 0.25, reverb: 0.3, swing: 0.04 },
  calm:      { about: "wellness, editorial, slow reveals: pads, sustained bass, rim, airy bell, slight swing",
               bpm: [70, 92], progression: ["Cmaj7", "Am7", "Fmaj7", "G"], kick: "half", snare: "rim", hats: "eighths-soft", bass: "sustain",
               chords: "pad", lead: "bell", leadMin: 0.6, sidechain: 0.2, reverb: 0.35, swing: 0.08 },
  cinematic: { about: "launch reveals, big statements, a lot of space: big pads, sparse kick, low sustain",
               bpm: [60, 86], progression: ["Am", "F", "Dm", "E"], kick: "sparse", snare: "none", hats: "none", bass: "sustain",
               chords: "pad", lead: "bell", leadMin: 0.6, sidechain: 0.1, reverb: 0.45, swing: 0 },
  tense:     { about: "problem / risk framing, security, 'before' acts: minor pulses, low drone, little melody",
               bpm: [84, 104], progression: ["Am", "Am", "F", "E"], kick: "half-soft", snare: "none", hats: "shaker", bass: "pulse8",
               chords: "pad", lead: "none", leadMin: 1, sidechain: 0.3, reverb: 0.35, swing: 0 },
  playful:   { about: "consumer apps, friendly brands, jokes with a character: plucks, claps, bouncy bass, swing",
               bpm: [100, 120], progression: ["C", "G", "Am", "F"], kick: "four-soft", snare: "clap", hats: "offbeat", bass: "offbeat8",
               chords: "stab", lead: "pluck", leadMin: 0.55, sidechain: 0.45, reverb: 0.25, swing: 0.12 },
  punchy:    { about: "fast feature montages, dev tools with energy: four-on-the-floor house drive",
               bpm: [118, 128], progression: ["Am", "F", "C", "G"], kick: "four", snare: "clap", hats: "offbeat", bass: "offbeat8",
               chords: "stab", lead: "pluck", leadMin: 0.6, sidechain: 0.65, reverb: 0.22, swing: 0 },
  hype:      { about: "launch countdowns, gaming, sports: fast and heavy, 16th hats and arps",
               bpm: [128, 140], progression: ["Am", "F", "C", "G"], kick: "four", snare: "clap", hats: "sixteenths", bass: "sixteenths",
               chords: "stab", lead: "arp", leadMin: 0.6, sidechain: 0.75, reverb: 0.18, swing: 0 },
};

// ---------- reverb (Freeverb-style) -----------------------------------------------------------

function reverb(L, R, { size = 0.82, damp = 0.35, width = 1 } = {}) {
  const combs = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116].map((d) => Math.round(d * SR / 44100));
  const aps = [556, 441, 341, 225].map((d) => Math.round(d * SR / 44100));
  const proc = (inp, spread) => {
    const n = inp.length, out = new Float32Array(n);
    for (const d0 of combs) { const d = d0 + spread, buf = new Float32Array(d); let idx = 0, store = 0;
      for (let i = 0; i < n; i++) { const y = buf[idx]; store = y * (1 - damp) + store * damp; buf[idx] = inp[i] * 0.015 + store * size; idx = (idx + 1) % d; out[i] += y; } }
    for (const d0 of aps) { const d = d0 + spread, buf = new Float32Array(d); let idx = 0;
      for (let i = 0; i < n; i++) { const b = buf[idx], x = out[i]; out[i] = -x + b; buf[idx] = x + b * 0.5; idx = (idx + 1) % d; } }
    return out;
  };
  const wl = proc(L, 0), wr = proc(R, 23), w1 = width / 2 + 0.5, w2 = (1 - width) / 2;
  const oL = new Float32Array(L.length), oR = new Float32Array(R.length);
  for (let i = 0; i < L.length; i++) { oL[i] = wl[i] * w1 + wr[i] * w2; oR[i] = wr[i] * w1 + wl[i] * w2; }
  return [oL, oR];
}

// ---------- music render (film-time grid) ------------------------------------------------------

function renderMusic(sc) {
  const mood = { ...MOODS[sc.mood || "product"], ...(sc.style || {}) };
  if (!MOODS[sc.mood || "product"]) die(`unknown mood "${sc.mood}" (${Object.keys(MOODS).join(", ")})`, 2);
  const dur = sc.duration, N = Math.ceil(dur * SR);
  const beat = 60 / sc.bpm, bar = beat * 4, step = beat / 4, off = sc.offset || 0, key = sc.transpose || 0;
  const rnd = mulberry32(sc.seed ?? 7), hum = mulberry32((sc.seed ?? 7) * 7 + 3);
  const sections = [...(sc.sections || [{ name: "all", from: 0, to: dur, energy: 0.6 }])].sort((a, b) => a.from - b.from);
  const secAt = (t) => sections.find((s) => t >= s.from && t < s.to) || null;
  const vo = (sc.vo && sc.vo.regions) || [];
  const inVO = (t, pad = 0.12) => vo.some(([a, b]) => t >= a - pad && t <= b + pad);
  const accents = sc.accents || [];
  const stops = accents.filter((a) => a.kind === "drop" && a.stop).map((a) => [a.t - beat, a.t]);
  const stopped = (t) => stops.some(([a, b]) => t >= a && t < b);
  const B = { drums: new Bus(N), bass: new Bus(N), chords: new Bus(N), lead: new Bus(N), fx: new Bus(N) };
  const kickTimes = [];
  const human = (t, ms = 5) => t + (hum() - 0.5) * 2 * ms / 1000;
  const vel = (v) => v * (0.88 + 0.24 * hum());
  let prevVoicing = null, lastBar = null, voicing = null;

  const k0 = Math.ceil((-0.05 - off) / step), k1 = Math.floor((dur - off) / step);
  for (let k = k0; k <= k1; k++) {
    const t = off + k * step;
    if (t < -0.05 || stopped(t)) continue;
    const sec = secAt(Math.max(0, t));
    if (!sec) continue;
    const q = ((k % 16) + 16) % 16, barNo = Math.floor(k / 16), beatNo = Math.floor(q / 4) + 1, onBeat = q % 4 === 0;
    const energy = sec.energy ?? 0.6;
    const prog = (sec.progression || sc.progression || mood.progression).map(parseChord);
    const chordBars = sec.chordBars || sc.chordBars || 1;
    const chord = prog[((Math.floor(barNo / chordBars) % prog.length) + prog.length) % prog.length];
    if (barNo !== lastBar) { voicing = voice(chord, key, prevVoicing); prevVoicing = voicing; lastBar = barNo; }
    const rootMidi = 40 + ((chord.root + key) % 12) + ((chord.root + key) % 12 > 7 ? -12 : 0) + 12;   // E2..D#3 region
    const layers = new Set(sec.layers || [
      ...(energy >= 0.15 ? ["chords"] : []), ...(energy >= 0.3 ? ["hats"] : []),
      ...(energy >= 0.45 ? ["bass", "kick"] : []), ...(energy >= 0.6 ? ["snare"] : [])]);
    if (sec.breakdown) ["kick", "snare", "bass"].forEach((l) => layers.delete(l));
    const swing = q % 2 === 1 ? mood.swing * beat / 2 : 0;
    const at = (tt) => (tt + swing) * SR;
    const firstStep = Math.abs(t - sec.from) < step / 2;

    // kick
    if (layers.has("kick")) {
      const p = mood.kick;
      const hit = p === "four" || p === "four-soft" ? onBeat : p === "half" || p === "half-soft" ? (q === 0 || (q === 8 && energy > 0.6)) : p === "sparse" ? q === 0 && barNo % 2 === 0 : false;
      if (hit) { B.drums.add(at(t), DRUMS.kick(rnd, { soft: p.endsWith("soft") }), 0.95 * (0.7 + 0.3 * energy) * (p.endsWith("soft") ? 0.75 : 1)); kickTimes.push(at(t)); }
    }
    // snare / clap / rim on 2 and 4
    if (layers.has("snare") && (beatNo === 2 || beatNo === 4) && onBeat && mood.snare !== "none") {
      const s = mood.snare === "clap" ? DRUMS.clap(rnd) : mood.snare === "rim" ? DRUMS.rim(rnd) : DRUMS.snare(rnd);
      B.drums.add(at(human(t, 3)), s, vel(mood.snare === "rim" ? 0.35 : 0.5), 0.05);
    }
    // hats / shaker
    if (layers.has("hats") && mood.hats !== "none") {
      const h = mood.hats;
      if (h === "shaker") {
        const accent = [0.55, 0.22, 0.38, 0.22][q % 4];
        B.drums.add(at(human(t, 4)), DRUMS.shaker(rnd), vel(accent * (0.5 + 0.5 * energy) * 0.5), 0.3);
      } else {
        const hit = h === "offbeat" ? q % 4 === 2 || (energy >= 0.8 && q % 2 === 1) : h === "sixteenths" ? true : h === "eighths-soft" ? q % 2 === 0 : false;
        if (hit) B.drums.add(at(human(t, 3)), DRUMS.hat(rnd, { open: h === "offbeat" && q === 14 && energy >= 0.8 }), vel((q % 4 === 2 ? 0.32 : 0.18) * (0.6 + 0.4 * energy) * (h === "eighths-soft" ? 0.6 : 1)), 0.35 * (q % 2 ? 1 : -1));
      }
    }
    // bass
    if (layers.has("bass")) {
      const st = mood.bass, f = mtof(rootMidi);
      if (st === "offbeat8" && q % 4 === 2) B.bass.add(at(t), synth({ freq: f, dur: beat * 0.45, wave: "saw", cutoff: 300, cutoffEnv: 900 * energy, cutoffDecay: 0.08, q: 1.1, sub: 0.6, a: 0.003, d: 0.1, s: 0.7, r: 0.04 }), vel(0.55));
      else if (st === "sixteenths" && q % 2 === 0 && q % 4 !== 0) B.bass.add(at(t), synth({ freq: f, dur: beat * 0.2, wave: "saw", cutoff: 380, cutoffEnv: 1200, cutoffDecay: 0.05, q: 1.2, sub: 0.6, r: 0.03 }), vel(0.5));
      else if (st === "pulse8" && q % 2 === 0) B.bass.add(at(human(t, 2)), synth({ freq: f, dur: beat * 0.38, wave: "tri", cutoff: 520 + 500 * energy, q: 0.7, sub: 0.5, a: 0.006, d: 0.18, s: 0.45, r: 0.08 }), vel(q === 0 ? 0.5 : 0.34));
      else if (st === "sustain" && q === 0 && barNo % chordBars === 0) B.bass.add(at(t), synth({ freq: f, dur: bar * chordBars * 0.95, wave: "sine", sub: 0.3, a: 0.05, d: 0.3, s: 0.8, r: 0.3, cutoff: 600 }), 0.5);
    }
    // chords
    if (layers.has("chords")) {
      if (mood.chords === "pad" && q === 0 && barNo % chordBars === 0) {
        voicing.forEach((m, i) => B.chords.add(at(t), synth({ freq: mtof(m), dur: bar * chordBars * 0.98, wave: "saw", voices: 3, detune: 14, a: 0.35, d: 0.4, s: 0.8, r: 0.6, cutoff: 900 + 1400 * energy, q: 0.6 }), 0.12, (i / Math.max(1, voicing.length - 1)) * 1.2 - 0.6));
      } else if (mood.chords === "keys") {
        // broken chord on the keys: low root on 1, then the voicing walks up; density follows energy
        const pattern = energy >= 0.6 ? [0, 3, 6, 8, 10, 12, 14] : energy >= 0.4 ? [0, 4, 6, 8, 12, 14] : [0, 6, 8, 14];
        const pi = pattern.indexOf(q);
        if (pi >= 0) {
          const notes = q === 0 ? [voicing[0] - 12, ...voicing] : [voicing[(pi - 1) % voicing.length] + (pi > voicing.length ? 12 : 0)];
          notes.forEach((m, i) => B.chords.add(at(human(t, q === 0 ? 2 + i * 6 : 6)), keys(mtof(m), q === 0 ? bar * 0.9 : beat * 0.9, vel(0.55 + 0.35 * energy), rnd), vel(q === 0 ? 0.16 : 0.13) * (0.75 + 0.35 * energy), (m - 64) / 24));
        }
      } else if (mood.chords === "stab" && energy >= 0.45 && (q === 2 || q === 6 || q === 10 || q === 14 || (energy >= 0.8 && q === 7))) {
        voicing.forEach((m, i) => B.chords.add(at(human(t, 3)), synth({ freq: mtof(m + 12), dur: beat * 0.18, wave: "saw", voices: 2, detune: 10, a: 0.002, d: 0.12, s: 0.2, r: 0.08, cutoff: 1200, cutoffEnv: 3200 * energy, cutoffDecay: 0.07, q: 0.9 }), vel(0.09), (i / Math.max(1, voicing.length - 1)) * 0.8 - 0.4));
      } else if (mood.chords === "stab" && energy < 0.45 && q === 0) {
        voicing.forEach((m, i) => B.chords.add(at(t), synth({ freq: mtof(m), dur: bar * 0.98, wave: "saw", voices: 3, detune: 14, a: 0.3, d: 0.4, s: 0.75, r: 0.5, cutoff: 900 + 1400 * energy, q: 0.6 }), 0.11, i * 0.3 - 0.45));
      }
    }
    // lead: never on top of speech; only from the mood's energy threshold (or an explicit section lead)
    const lead = sec.lead ?? (energy >= mood.leadMin ? mood.lead : "none");
    if (lead !== "none" && !inVO(t)) {
      const pool = [...voicing, ...voicing.map((x) => x + 12)];
      if (lead === "pluck" && [0, 3, 6, 8, 10, 12, 14].includes(q) && hum() < 0.8) {
        B.lead.add(at(human(t, 3)), synth({ freq: mtof(pool[Math.floor(hum() * pool.length)] + 12), dur: beat * 0.25, wave: "saw", a: 0.002, d: 0.16, s: 0, r: 0.12, cutoff: 700, cutoffEnv: 5200, cutoffDecay: 0.06, q: 1.4 }), vel(0.16), (hum() - 0.5) * 0.5);
      } else if (lead === "arp") {
        B.lead.add(at(t), synth({ freq: mtof(pool[q % pool.length] + 12), dur: beat * 0.22, wave: "saw", voices: 2, detune: 8, a: 0.002, d: 0.1, s: 0.3, r: 0.06, cutoff: 1500, cutoffEnv: 3500, cutoffDecay: 0.05 }), vel(0.11), q % 2 ? 0.3 : -0.3);
      } else if (lead === "bell" && (q === 0 || q === 8 || (energy >= 0.7 && q === 12))) {
        const f = mtof(pool[Math.floor(hum() * pool.length)] + 12);
        const n = synth({ freq: f, dur: beat * 1.5, wave: "sine", a: 0.003, d: 0.9, s: 0, r: 0.6, cutoff: 9000 }), n2 = synth({ freq: f * 3.5, dur: beat * 0.6, wave: "sine", a: 0.002, d: 0.3, s: 0, r: 0.2, cutoff: 9000 });
        for (let i = 0; i < n2.length && i < n.length; i++) n[i] += n2[i] * 0.25;
        B.lead.add(at(human(t, 4)), n, vel(0.15), (hum() - 0.5) * 0.7);
      } else if (lead === "keys" && [0, 6, 10].includes(q)) {
        B.lead.add(at(human(t, 5)), keys(mtof(pool[Math.floor(hum() * pool.length)] + 12), beat * 1.2, vel(0.7), rnd), vel(0.12), (hum() - 0.5) * 0.6);
      }
    }
    // section openers that sit on a downbeat get a soft cymbal-ish lift when the energy rises
    if (firstStep && q === 0 && energy >= 0.5) B.fx.add(at(t), fxSound("downsweep", rnd, { len: 1.4 }), 0.08, -0.2);
  }

  // accents
  for (const ac of accents) {
    const t0 = ac.t * SR;
    if (ac.kind === "drop" || ac.kind === "lift") { const r = fxSound("riser", rnd, { len: Math.min(bar, 2.4) }); B.fx.add(t0 - r.length, r, ac.kind === "lift" ? 0.22 : 0.28); }
    if (ac.kind === "drop" || ac.kind === "hit") { B.fx.add(t0, fxSound("impact", rnd), ac.kind === "hit" ? 0.45 : 0.55); B.fx.add(t0, fxSound("downsweep", rnd), 0.15, 0.2); }
    if (ac.kind === "stinger") {
      const sec = secAt(ac.t) || sections[sections.length - 1];
      const ch = (sec.progression || sc.progression || mood.progression).map(parseChord)[0];
      for (const m of voice(ch, key, prevVoicing)) B.fx.add(t0, synth({ freq: mtof(m + 12), dur: beat * 0.5, wave: "saw", voices: 3, detune: 12, a: 0.002, d: 0.4, s: 0.2, r: 1.2, cutoff: 3000, cutoffEnv: 3000, cutoffDecay: 0.2 }), 0.1);
      B.fx.add(t0, fxSound("impact", rnd, { len: 2 }), 0.3);
    }
  }

  // sidechain on kicks
  const scg = new Float32Array(N).fill(1), depth = mood.sidechain, rel = 0.16 * SR;
  for (const k of kickTimes) for (let i = 0; i < rel; i++) { const j = Math.round(k) + i; if (j < 0) continue; if (j >= N) break; scg[j] = Math.min(scg[j], 1 - depth * Math.pow(1 - i / rel, 2)); }
  for (const name of ["bass", "chords", "lead"]) { const g = name === "bass" ? 1 : 0.8; for (let i = 0; i < N; i++) { const m = 1 - (1 - scg[i]) * g; B[name].L[i] *= m; B[name].R[i] *= m; } }

  // lead sits lower under speech even when it rings into it
  if (vo.length) {
    const g = dbToGain(-(sc.vo.lead_duck_db ?? 6)), env = new Float32Array(N).fill(1);
    for (const [a, b] of vo) for (let i = Math.max(0, Math.round((a - 0.08) * SR)); i < Math.min(N, Math.round((b + 0.2) * SR)); i++) env[i] = g;
    let acc = 1; const sm = 0.05 * SR;
    for (let i = 0; i < N; i++) { acc += (env[i] - acc) / sm; B.lead.L[i] *= acc; B.lead.R[i] *= acc; }
  }

  // reverb send
  const send = new Bus(N);
  send.addStereo(0, B.chords.L, B.chords.R, 1); send.addStereo(0, B.lead.L, B.lead.R, 0.8); send.addStereo(0, B.fx.L, B.fx.R, 0.6);
  const [rvL, rvR] = reverb(send.L, send.R);

  // section dynamics, the stop gate, the end fade
  const lvl = new Float32Array(N);
  for (let i = 0; i < N; i++) { const s = secAt(i / SR); lvl[i] = s ? dbToGain(-9 * (1 - (s.energy ?? 0.6))) : 0; }
  for (const [a, b] of stops) for (let i = Math.max(0, Math.round(a * SR)); i < Math.min(N, Math.round(b * SR)); i++) lvl[i] = 0;
  const fade = (sc.fade_out ?? 0.8) * SR;
  for (let i = Math.max(0, N - fade); i < N; i++) lvl[i] *= (N - i) / fade;
  let acc = lvl[0]; const sm = Math.round(0.03 * SR);
  const gate = new Float32Array(N);
  for (let i = 0; i < N; i++) { acc += (lvl[i] - acc) / sm; gate[i] = acc; }

  const mixG = { drums: 1.0, bass: 0.9, chords: 0.9, lead: 0.85 };
  const L = new Float32Array(N), R = new Float32Array(N);
  for (const [name, g] of Object.entries(mixG)) for (let i = 0; i < N; i++) { L[i] += B[name].L[i] * g * gate[i]; R[i] += B[name].R[i] * g * gate[i]; }
  for (let i = 0; i < N; i++) { L[i] += rvL[i] * mood.reverb * 3 * gate[i] + B.fx.L[i] * 0.9; R[i] += rvR[i] * mood.reverb * 3 * gate[i] + B.fx.R[i] * 0.9; }
  let peak = 0;
  for (let i = 0; i < N; i++) { L[i] = Math.tanh(L[i] * 1.1); R[i] = Math.tanh(R[i] * 1.1); peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i])); }
  const norm = peak > 0 ? dbToGain(-3) / peak : 1;
  for (let i = 0; i < N; i++) { L[i] *= norm; R[i] *= norm; }
  return { L, R, B, N };
}

// ---------- SFX render ------------------------------------------------------------------------

function renderSfx(sc) {
  const N = Math.ceil(sc.duration * SR), bus = new Bus(N), rnd = mulberry32((sc.seed ?? 7) * 13 + 5);
  for (const ev of sc.sfx || []) {
    const count = ev.count || 1, every = ev.every ?? 0.09;
    for (let c = 0; c < count; c++) {
      const pitch = (ev.pitch || 1) * (ev.rise ? Math.pow(2, c * ev.rise / 12) : 1) * (ev.kind === "type" ? 0.9 + 0.2 * rnd() : 1);
      const s = fxSound(ev.kind, rnd, { len: ev.len, pitch });
      if (!s) { console.warn(`warn: unknown sfx "${ev.kind}" ignored`); break; }
      let pk = 0; for (const x of s) pk = Math.max(pk, Math.abs(x));
      const g = (pk > 0 ? dbToGain(-3) / pk : 1) * dbToGain(ev.gain ?? -8);         // each sound peak-normalised to -3 dBFS, then its gain
      const swell = (SWELL[ev.kind] ?? 0) * s.length;                                 // land the loudest point (or the transient) on t
      bus.add((ev.t + c * every) * SR - swell, s, g, ev.pan ?? 0);
    }
  }
  return bus;
}

// ---------- plan: derive the score from the film --------------------------------------------------

function parseRegions(str) {
  return String(str).split(",").map((p) => p.trim()).filter(Boolean).map((p) => { const [a, b] = p.split("-").map(Number); if (!(b > a)) die(`bad region "${p}"`, 2); return [a, b]; });
}
function voRegionsFromWav(file) {
  const ff = process.env.FFMPEG_BIN || "ffmpeg";
  const r = spawnSync(ff, ["-hide_banner", "-nostats", "-i", file, "-af", "silencedetect=n=-40dB:d=0.25", "-f", "null", "-"], { encoding: "utf8" });
  if (r.error) die("ffmpeg is needed for --vo-wav", 2);
  const dur = Number((/Duration: (\d+):(\d+):([\d.]+)/.exec(r.stderr) || []).slice(1).reduce((s, x, i) => s + Number(x) * [3600, 60, 1][i], 0));
  const sil = []; let st = null;
  for (const ln of r.stderr.split("\n")) {
    const a = /silence_start: ([\d.]+)/.exec(ln); if (a) st = Number(a[1]);
    const b = /silence_end: ([\d.]+)/.exec(ln); if (b) { sil.push([st ?? 0, Number(b[1])]); st = null; }
  }
  if (st !== null) sil.push([st, dur]);
  const out = []; let cur = 0;
  for (const [a, b] of sil) { if (a > cur + 0.05) out.push([+cur.toFixed(3), +a.toFixed(3)]); cur = b; }
  if (dur > cur + 0.05) out.push([+cur.toFixed(3), +dur.toFixed(3)]);
  return out;
}

// SFX map: picture event -> sound. Labels let a film name its own hero hit; everything else follows the type.
function sfxFor(ev, heroFrame) {
  const lab = (ev.label || "").toLowerCase();
  if (ev.frame === heroFrame) return { kind: "lock", gain: -5, why: `hero hit (${ev.label || ev.type})` };
  switch (ev.type) {
    case "hit": return /poly|point|land.*(a|chart)|apex|vektor|pop/.test(lab) && !/needle|gauge/.test(lab) ? { kind: "pop", gain: -14, why: `data lands (${ev.label})` } : { kind: "tick", gain: -7, why: `value lands (${ev.label})` };
    case "press": return { kind: "click", gain: -6, why: "cursor press" };
    case "type": return { kind: "type", gain: -14, why: "typing in the product" };
    case "snap": return { kind: "tick", gain: -8, why: `snap (${ev.label})` };
    case "whip": return { kind: "whoosh", gain: -9, lead: 4, why: "whip pan" };
    case "dive": return { kind: "whoosh", gain: -9, lead: 8, why: "dive" };
    case "focus": return { kind: "focus", gain: -11, lead: 3, why: `focus (${ev.label})` };
    case "count": return { kind: "tick", gain: -15, why: "counter settles" };
    default: return null;
  }
}

function plan(opts) {
  const fps = opts.fps, dur = opts.duration;
  const evs = JSON.parse(fs.readFileSync(opts.events, "utf8"));
  const events = (Array.isArray(evs) ? evs : evs.events || []).map((e) => ({ ...e, frame: e.frame ?? Math.round(e.t * fps) }));
  const mood = opts.mood;
  if (!MOODS[mood]) die(`unknown mood "${mood}" (${Object.keys(MOODS).join(", ")})`, 2);
  // the hero: a frame number, or a label substring of a "hit" event
  let heroFrame = /^\d+$/.test(String(opts.hero)) ? Number(opts.hero) : null;
  if (heroFrame === null && opts.hero) { const h = events.find((e) => (e.label || "").toLowerCase().includes(String(opts.hero).toLowerCase())); if (!h) die(`no event label contains "${opts.hero}"`, 2); heroFrame = h.frame; }
  if (heroFrame === null) die("--hero <frame | label text> is required: name the one moment the music should land on", 2);
  const heroT = heroFrame / fps;
  const cuts = [...new Set(events.filter((e) => e.type === "cut").map((e) => e.frame))].sort((a, b) => a - b).map((f) => f / fps);
  const closeT = opts.close !== undefined ? Number(opts.close) / fps : (cuts.filter((c) => c > heroT).pop() ?? null);

  // tempo: inside the mood's range, the hero AND the closing cut both on bar downbeats (else the close on a beat)
  const [lo, hi] = mood && MOODS[mood].bpm, mid = (lo + hi) / 2;
  let bpm = null, how = "";
  if (closeT !== null && closeT > heroT + 0.5) {
    const d = closeT - heroT, cands = [];
    for (let k = 1; k <= 16; k++) { const b = 240 * k / d; if (b >= lo && b <= hi) cands.push({ b, k, unit: "bar" }); }
    if (!cands.length) for (let j = 1; j <= 64; j++) { const b = 60 * j / d; if (b >= lo && b <= hi) cands.push({ b, k: j, unit: "beat" }); }
    if (cands.length) { const c = cands.sort((x, y) => Math.abs(x.b - mid) - Math.abs(y.b - mid))[0]; bpm = +c.b.toFixed(3); how = `hero ${heroT.toFixed(2)} s and close ${closeT.toFixed(2)} s are ${c.k} ${c.unit}${c.k > 1 ? "s" : ""} apart`; }
  }
  if (bpm === null) { bpm = mid; how = "no closing cut to pair with the hero; mood centre tempo"; }
  const bar = 240 / bpm, offset = +(heroT - Math.ceil(heroT / bar) * bar).toFixed(4);
  const beat = bar / 4, offBeat = (t) => { const x = ((t - offset) / beat) % 1; return Math.min(x, 1 - x) * beat; };

  // sections: one per act between cuts; roles by position unless given
  const bounds = [0, ...cuts.filter((c) => c > 0.05 && c < dur - 0.05), dur];
  const acts = bounds.slice(0, -1).map((a, i) => ({ from: +a.toFixed(3), to: +bounds[i + 1].toFixed(3) }));
  const roles = opts.roles ? opts.roles.split(",").map((s) => s.trim()) : null;
  if (roles && roles.length !== acts.length) die(`--roles has ${roles.length} entries but the film has ${acts.length} acts (cuts at ${cuts.map((c) => c.toFixed(2)).join(", ")})`, 2);
  const heroAct = acts.findIndex((a) => heroT >= a.from && heroT < a.to);
  const proofs = acts.map((_, i) => i).filter((i) => i > 0 && i < acts.length - 1 && i !== heroAct);
  const ENERGY = { hook: 0.3, turn: 0.72, cta: 0.3, proof: 0.5 };
  const sections = acts.map((a, i) => {
    const role = roles ? roles[i] : i === 0 ? "hook" : i === heroAct ? "turn" : i === acts.length - 1 ? "cta" : "proof";
    let energy = ENERGY[role] ?? 0.5, why;
    if (role === "proof") { const r = proofs.indexOf(i); energy = +(0.45 + 0.12 * (proofs.length > 1 ? r / (proofs.length - 1) : 0)).toFixed(2); why = "proof: the bed builds act by act toward the turn"; }
    else why = { hook: "hook: low and spacious so the claim and the voice lead", turn: "turn / hero: the peak; the drop lands on the hero hit", cta: "close: resolves smaller than it began; stinger on the cut" }[role] || role;
    return { name: role + (role === "proof" ? proofs.indexOf(i) + 1 : ""), role, from: a.from, to: a.to, energy, why };
  });
  const accents = [{ t: +heroT.toFixed(3), kind: "drop", stop: true, why: "the hero hit: one beat of silence, then the drop (silence before the hit)" }];
  if (closeT !== null && closeT > heroT) accents.push({ t: +closeT.toFixed(3), kind: "stinger", why: "the closing cut lands on a downbeat with a final chord" });

  // VO
  let regions = [];
  if (opts.voWav) regions = voRegionsFromWav(opts.voWav);
  else if (opts.voRegions) regions = parseRegions(opts.voRegions);

  // SFX from the picture's own events, deduplicated within 2 frames per kind
  const sfx = [], seen = [];
  for (const e of events) {
    const m = sfxFor(e, heroFrame);
    if (!m) continue;
    const t = (e.frame + (m.lead || 0)) / fps;
    if (seen.some(([k, tt]) => k === m.kind && Math.abs(tt - t) < 2 / fps)) continue;
    seen.push([m.kind, t]);
    sfx.push({ t: +t.toFixed(3), kind: m.kind, gain: m.gain, why: m.why });
  }

  const report = cuts.map((c) => `${c.toFixed(2)}s ${(offBeat(c) * 1000).toFixed(0)}ms off-beat`).join(", ");
  return {
    fps, duration: dur, bpm, offset, mood, progression: MOODS[mood].progression, transpose: 0, seed: 7, fade_out: 0.8,
    why: { tempo: `${bpm} BPM (${mood} range ${lo}-${hi}): ${how}; bar 1 at ${offset} s so the hero is a downbeat`, mood: MOODS[mood].about,
           cut_alignment: report || "no cuts" },
    sections, accents, vo: { regions, lead_duck_db: 6 }, sfx,
  };
}

// ---------- output ------------------------------------------------------------------------

function writeWav(file, L, R) {
  const n = L.length, buf = Buffer.alloc(44 + n * 4);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 4, 4); buf.write("WAVE", 8); buf.write("fmt ", 12); buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22); buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 4, 28); buf.writeUInt16LE(4, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) { buf.writeInt16LE(Math.round(clamp(L[i], -1, 1) * 32767), 44 + i * 4); buf.writeInt16LE(Math.round(clamp(R[i], -1, 1) * 32767), 46 + i * 4); }
  fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true });
  fs.writeFileSync(file, buf);
}
function ffmpegJson(args) {
  const r = spawnSync(process.env.FFMPEG_BIN || "ffmpeg", args, { encoding: "utf8" });
  if (r.error && r.error.code === "ENOENT") return { missing: true };
  const m = /\{[\s\S]*\}/.exec(r.stderr || "");
  if (!m) die("ffmpeg loudnorm failed: " + (r.stderr || "").slice(-300));
  return JSON.parse(m[0]);
}
function master(raw, out) {
  const j = ffmpegJson(["-hide_banner", "-nostats", "-i", raw, "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json", "-f", "null", "-"]);
  if (j.missing) return null;
  const af = `loudnorm=I=-14:TP=-1:LRA=11:measured_I=${j.input_i}:measured_TP=${j.input_tp}:measured_LRA=${j.input_lra}:measured_thresh=${j.input_thresh}:offset=${j.target_offset}:linear=true`;
  execFileSync(process.env.FFMPEG_BIN || "ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-i", raw, "-af", af, "-ar", String(SR), "-ac", "2", "-c:a", "pcm_s16le", out]);
  const m = ffmpegJson(["-hide_banner", "-nostats", "-i", out, "-af", "loudnorm=I=-14:TP=-1:print_format=json", "-f", "null", "-"]);
  return { lufs: +m.input_i, tp: +m.input_tp };
}

// ---------- CLI ---------------------------------------------------------------------------

function args(argv) {
  const o = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) { const k = a.slice(2), v = argv[i + 1]; if (v === undefined || v.startsWith("--")) o[k] = true; else { o[k] = v; i++; } }
    else o._.push(a);
  }
  return o;
}
const HELP = fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").slice(1).filter((l) => l.startsWith("//")).map((l) => l.slice(3)).join("\n");

function main() {
  const a = args(process.argv.slice(2)), cmd = a._[0];
  if (!cmd || a.help || cmd === "help") { console.log(HELP); process.exit(cmd ? 0 : 2); }
  if (cmd === "moods") { for (const [k, m] of Object.entries(MOODS)) console.log(`${k.padEnd(10)} ${m.bpm[0]}-${m.bpm[1]} BPM  ${m.about}`); return; }
  if (cmd === "plan") {
    if (!a.events || !a.duration) die("plan needs --events <render events.json> and --duration <seconds>", 2);
    const sc = plan({ events: a.events, duration: Number(a.duration), fps: Number(a.fps || 30), hero: a.hero, close: a.close, mood: a.mood || "product",
                      roles: a.roles, voWav: a["vo-wav"], voRegions: a["vo-regions"] });
    const out = a.out || "audio/score.json";
    fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
    fs.writeFileSync(out, JSON.stringify(sc, null, 1) + "\n");
    console.log(`${out}\n  tempo   ${sc.why.tempo}\n  mood    ${sc.mood}: ${sc.why.mood}\n  acts    ${sc.sections.map((s) => `${s.name} ${s.from}-${s.to}s e${s.energy}`).join(" | ")}`);
    console.log(`  accents ${sc.accents.map((x) => `${x.kind}@${x.t}s`).join(", ")}\n  vo      ${sc.vo.regions.length} speech regions${sc.vo.regions.length ? "" : " (none given: melody may sit on top of speech; pass --vo-wav or --vo-regions)"}\n  sfx     ${sc.sfx.length} placed from picture events\n  cuts    ${sc.why.cut_alignment}`);
    console.log("REVIEW the score (mood, energy arc, accents, sfx) against the storyboard and the brand before rendering: node score.mjs render " + out);
    return;
  }
  if (cmd === "render") {
    const file = a._[1];
    if (!file) die("render needs a score.json", 2);
    const sc = JSON.parse(fs.readFileSync(file, "utf8"));
    for (const k of ["bpm", "duration"]) if (!sc[k]) die(`score is missing "${k}"`, 2);
    const dir = path.dirname(file), t0 = Date.now();
    const musicOut = a["music-out"] || path.join(dir, "music.native.wav"), sfxOut = a["sfx-out"] || path.join(dir, "sfx.native.wav");
    if (a["sfx-out"] !== "none") {
      const s = renderSfx(sc); writeWav(sfxOut, s.L, s.R);
      console.log(`${sfxOut}  ${(sc.sfx || []).length} sfx`);
    }
    if (a["music-out"] !== "none") {
      const { L, R, B } = renderMusic(sc);
      const raw = musicOut.replace(/\.wav$/, ".raw.wav");
      writeWav(raw, L, R);
      if (a.stems) for (const [n, bus] of Object.entries(B)) writeWav(musicOut.replace(/\.wav$/, `.${n}.wav`), bus.L, bus.R);
      const m = master(raw, musicOut);
      if (m) fs.unlinkSync(raw); else fs.renameSync(raw, musicOut);
      console.log(`${musicOut}  ${sc.duration.toFixed(2)} s @ ${sc.bpm} BPM, mood ${sc.mood}` + (m ? `, ${m.lufs.toFixed(1)} LUFS, TP ${m.tp.toFixed(1)} dBTP` : ", not mastered (no ffmpeg)") + `  [${((Date.now() - t0) / 1000).toFixed(1)} s]`);
    }
    console.log("LISTEN before shipping: a synthesized bed is a floor, not a score. Mix with mix.py (--music <music> --duck, --sfx <sfx>@0:0).");
    return;
  }
  die(`unknown command "${cmd}" (plan | render | moods)`, 2);
}
main();
