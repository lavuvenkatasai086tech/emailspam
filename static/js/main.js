/* ================================================================
   SpamGuard AI — Main JavaScript
   Handles API calls, UI state, animations, counters
   ================================================================ */

"use strict";

// ── Sample emails ─────────────────────────────────────────────────
const SAMPLES = {
  spam: `CONGRATULATIONS! You've been SELECTED as our LUCKY WINNER! 🎉

You have WON a guaranteed prize of $10,000 cash OR a FREE iPhone 15 Pro!

This is a LIMITED TIME OFFER — act NOW before it expires!

✅ No purchase necessary
✅ Claim your FREE gift card INSTANTLY
✅ 100% GUARANTEED prize

Click here to CLAIM YOUR PRIZE: http://claim-your-prize-now.win/free

You must respond within 24 HOURS or your prize will be forfeited!
Call us now: 1-800-WIN-FREE

Unsubscribe | Terms | Privacy`,

  ham: `Hi Sarah,

Hope you're doing well! I wanted to follow up on the project proposal
we discussed in last Tuesday's meeting.

I've finished reviewing the budget breakdown and I have a few minor
suggestions. Could we schedule a quick 30-minute call this week to
go over the changes before we send it to the client?

I'm available Thursday afternoon or Friday morning — let me know
what works best for you.

Also, don't forget the team lunch is this Friday at 12:30 PM at
The Garden Bistro. Looking forward to it!

Best regards,
James`,
};

// ── DOM refs ──────────────────────────────────────────────────────
const emailInput    = document.getElementById("email-input");
const charCounter   = document.getElementById("char-counter");
const analyzeBtn    = document.getElementById("analyze-btn");
const btnText       = analyzeBtn.querySelector(".btn-text");
const btnLoader     = analyzeBtn.querySelector(".btn-loader");
const emptyState    = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");

// ── Char counter ──────────────────────────────────────────────────
emailInput.addEventListener("input", () => {
  const len = emailInput.value.length;
  charCounter.textContent = `${len} / 5000`;
  charCounter.style.color = len > 4500 ? "var(--spam-clr)" : "var(--text-dim)";
});

// ── Sample loaders ────────────────────────────────────────────────
function loadSample(type) {
  emailInput.value = SAMPLES[type];
  emailInput.dispatchEvent(new Event("input"));
  emailInput.focus();
  emailInput.scrollTop = 0;
}

function clearInput() {
  emailInput.value = "";
  emailInput.dispatchEvent(new Event("input"));
  showEmptyState();
  emailInput.focus();
}

// ── UI helpers ────────────────────────────────────────────────────
function showEmptyState() {
  emptyState.hidden    = false;
  resultContent.hidden = true;
}

function showResults() {
  emptyState.hidden    = true;
  resultContent.hidden = false;
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  btnText.hidden      = on;
  btnLoader.hidden    = !on;
}

function animateBar(el, pct) {
  el.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { el.style.width = pct + "%"; });
  });
}

function animateCounter(el) {
  const target = parseFloat(el.dataset.target || 0);
  const duration = 1200;
  const start = performance.now();
  const update = (now) => {
    const t   = Math.min((now - start) / duration, 1);
    const val = (target * easeOut(t)).toFixed(1);
    el.textContent = val;
    if (t < 1) requestAnimationFrame(update);
    else el.textContent = target.toFixed(1);
  };
  requestAnimationFrame(update);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

// ── Stat card counters on scroll ──────────────────────────────────
const counters = document.querySelectorAll(".counter");
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCounter(e.target);
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.5 });
counters.forEach(c => observer.observe(c));

// ── Main analyze function ─────────────────────────────────────────
async function analyzeEmail() {
  const text = emailInput.value.trim();
  if (!text) {
    shake(emailInput);
    return;
  }

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Network error — is the Flask server running?");
  } finally {
    setLoading(false);
  }
}

// ── Render result ─────────────────────────────────────────────────
function renderResult(d) {
  const isSpam = d.is_spam;

  // Verdict banner
  const banner = document.getElementById("verdict-banner");
  banner.className = "verdict-banner " + (isSpam ? "spam-verdict" : "ham-verdict");

  document.getElementById("verdict-icon").textContent  = isSpam ? "🚨" : "✅";
  document.getElementById("verdict-label").textContent = isSpam ? "SPAM DETECTED" : "LEGITIMATE EMAIL";
  document.getElementById("verdict-label").style.color = isSpam ? "var(--spam-clr)" : "var(--ham-clr)";
  document.getElementById("verdict-sublabel").textContent =
    isSpam
      ? `${d.confidence}% confident this is spam`
      : `${d.confidence}% confident this is legitimate`;

  // Risk badge
  const rb = document.getElementById("risk-badge");
  rb.textContent  = riskLabel(d.risk_level);
  rb.className    = `risk-badge risk-${d.risk_level}`;

  // Confidence bars
  animateBar(document.getElementById("spam-bar"), d.spam_prob);
  animateBar(document.getElementById("ham-bar"),  d.ham_prob);
  document.getElementById("spam-pct").textContent = d.spam_prob + "%";
  document.getElementById("ham-pct").textContent  = d.ham_prob  + "%";

  // Feature grid
  const f   = d.features;
  const fg  = document.getElementById("feature-grid");
  fg.innerHTML = featureHTML("Words",        f.word_count, "")
               + featureHTML("Characters",   f.char_count, "")
               + featureHTML("Caps Ratio",   f.caps_ratio, "%")
               + featureHTML("Exclamations", f.exclamations, "")
               + featureHTML("URLs Found",   f.url_count, "")
               + featureHTML("Digit Density",f.digit_density, "%");

  // Spam keywords
  const ks    = document.getElementById("keywords-section");
  const chips = document.getElementById("keyword-chips");
  if (f.spam_keywords && f.spam_keywords.length > 0) {
    chips.innerHTML = f.spam_keywords
      .map(k => `<span class="keyword-chip">${escHtml(k)}</span>`)
      .join("");
    ks.hidden = false;
  } else {
    ks.hidden = true;
  }

  // Timestamp
  document.getElementById("result-timestamp").textContent =
    "Analyzed at " + d.timestamp;

  showResults();
}

function featureHTML(label, value, unit) {
  return `<div class="feature-item">
    <div class="feature-key">${label}</div>
    <div class="feature-val">${value}${unit}</div>
  </div>`;
}

function riskLabel(r) {
  return { high: "⚠ High Risk", medium: "⚡ Medium Risk", low: "🔎 Low Risk", safe: "✓ Safe" }[r] || r;
}

function escHtml(str) {
  return str.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

// ── Error display ─────────────────────────────────────────────────
function showError(msg) {
  showResults();
  const rb = document.getElementById("verdict-banner");
  rb.className  = "verdict-banner spam-verdict";
  rb.innerHTML  = `<div class="verdict-icon">⚠️</div>
    <div class="verdict-info">
      <div class="verdict-label" style="color:var(--warn-clr)">Error</div>
      <div class="verdict-sublabel">${escHtml(msg)}</div>
    </div>`;
  document.getElementById("spam-bar").style.width   = "0%";
  document.getElementById("ham-bar").style.width    = "0%";
  document.getElementById("spam-pct").textContent   = "—";
  document.getElementById("ham-pct").textContent    = "—";
  document.getElementById("feature-grid").innerHTML = "";
  document.getElementById("keywords-section").hidden = true;
  document.getElementById("result-timestamp").textContent = "";
}

// ── Shake animation for empty input ──────────────────────────────
function shake(el) {
  el.style.animation = "none";
  el.offsetHeight; // reflow
  el.style.animation = "shake 0.4s ease";
  el.addEventListener("animationend", () => { el.style.animation = ""; }, { once: true });
}

// Inject shake keyframes
const shakeCSS = document.createElement("style");
shakeCSS.textContent = `
@keyframes shake {
  0%,100% { transform: translateX(0); }
  20%      { transform: translateX(-8px); }
  40%      { transform: translateX( 8px); }
  60%      { transform: translateX(-5px); }
  80%      { transform: translateX( 5px); }
}`;
document.head.appendChild(shakeCSS);

// ── Enter key shortcut ────────────────────────────────────────────
emailInput.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") analyzeEmail();
});

// ── Smooth reveal on scroll ───────────────────────────────────────
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add("fade-up");
      revealObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".pipeline-step, .stat-card, .dataset-item, .tech-item")
  .forEach(el => revealObserver.observe(el));
