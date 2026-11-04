/**
 * Popup Script - Simple Student TP Style
 */

let appState = {
  apiUrl: "http://localhost:7860",
  predictions: [],
  statistics: null,
  activeFilter: "all",
  searchQuery: ""
};

// Elements
const statusText = document.getElementById("statusText");
const toggleConfigBtn = document.getElementById("toggleConfigBtn");
const configSection = document.getElementById("configSection");
const apiUrlInput = document.getElementById("apiUrlInput");
const saveConfigBtn = document.getElementById("saveConfigBtn");
const testConfigBtn = document.getElementById("testConfigBtn");
const configMsg = document.getElementById("configMsg");

const commentLimitSelect = document.getElementById("commentLimitSelect");
const analyzeBtn = document.getElementById("analyzeBtn");
const alertBanner = document.getElementById("alertBanner");
const alertMessage = document.getElementById("alertMessage");
const loadingState = document.getElementById("loadingState");
const resultsSection = document.getElementById("resultsSection");

// Summary elements
const predominantText = document.getElementById("predominantText");
const sentimentScore = document.getElementById("sentimentScore");
const avgConfidence = document.getElementById("avgConfidence");

const posCount = document.getElementById("posCount");
const posPct = document.getElementById("posPct");
const neuCount = document.getElementById("neuCount");
const neuPct = document.getElementById("neuPct");
const negCount = document.getElementById("negCount");
const negPct = document.getElementById("negPct");

const barPositive = document.getElementById("barPositive");
const barNeutral = document.getElementById("barNeutral");
const barNegative = document.getElementById("barNegative");

const tabCountAll = document.getElementById("tabCountAll");
const tabCountPos = document.getElementById("tabCountPos");
const tabCountNeu = document.getElementById("tabCountNeu");
const tabCountNeg = document.getElementById("tabCountNeg");
const filterBtns = document.querySelectorAll(".filter-btn");
const searchInput = document.getElementById("searchInput");
const commentsFeed = document.getElementById("commentsFeed");

const copySummaryBtn = document.getElementById("copySummaryBtn");
const exportCsvBtn = document.getElementById("exportCsvBtn");
const exportJsonBtn = document.getElementById("exportJsonBtn");

// Init
document.addEventListener("DOMContentLoaded", async () => {
  const savedUrl = localStorage.getItem("tp_api_url");
  if (savedUrl) appState.apiUrl = savedUrl;
  apiUrlInput.value = appState.apiUrl;

  checkApi();
  setupEvents();
});

async function checkApi() {
  statusText.textContent = "Test...";
  statusText.style.color = "#888";
  try {
    const clean = appState.apiUrl.replace(/\/+$/, "");
    const res = await fetch(`${clean}/health`, { method: "GET" });
    if (res.ok) {
      statusText.textContent = "En ligne";
      statusText.style.color = "#28a745";
    } else {
      statusText.textContent = "Erreur HTTP " + res.status;
      statusText.style.color = "#dc3545";
    }
  } catch (e) {
    statusText.textContent = "Hors ligne (localhost:7860)";
    statusText.style.color = "#dc3545";
  }
}

function setupEvents() {
  toggleConfigBtn.addEventListener("click", () => {
    configSection.classList.toggle("hidden");
  });

  saveConfigBtn.addEventListener("click", () => {
    appState.apiUrl = apiUrlInput.value.trim().replace(/\/+$/, "");
    localStorage.setItem("tp_api_url", appState.apiUrl);
    configMsg.textContent = "Sauvegardé !";
    configMsg.style.color = "#28a745";
    checkApi();
  });

  testConfigBtn.addEventListener("click", async () => {
    configMsg.textContent = "Test en cours...";
    configMsg.style.color = "#666";
    try {
      const url = apiUrlInput.value.trim().replace(/\/+$/, "");
      const res = await fetch(`${url}/health`);
      if (res.ok) {
        configMsg.textContent = "Connexion réussie (200 OK)";
        configMsg.style.color = "#28a745";
      } else {
        configMsg.textContent = "Erreur " + res.status;
        configMsg.style.color = "#dc3545";
      }
    } catch (e) {
      configMsg.textContent = "Échec : " + e.message;
      configMsg.style.color = "#dc3545";
    }
  });

  analyzeBtn.addEventListener("click", analyzeComments);

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      appState.activeFilter = btn.dataset.filter;
      renderFeed();
    });
  });

  searchInput.addEventListener("input", (e) => {
    appState.searchQuery = e.target.value.toLowerCase().trim();
    renderFeed();
  });

  copySummaryBtn.addEventListener("click", () => {
    if (!appState.statistics) return;
    const s = appState.statistics;
    const text = `TP3 YouTube Sentiment :\nTotal: ${s.total_comments}\nPositifs: ${s.positive_count} (${s.positive_pct}%)\nNeutres: ${s.neutral_count} (${s.neutral_pct}%)\nNégatifs: ${s.negative_count} (${s.negative_pct}%)\nScore global: ${s.sentiment_score}`;
    navigator.clipboard.writeText(text).then(() => {
      copySummaryBtn.textContent = "Copié !";
      setTimeout(() => (copySummaryBtn.textContent = "📋 Copier résumé"), 1500);
    });
  });

  exportCsvBtn.addEventListener("click", exportCsv);
  exportJsonBtn.addEventListener("click", exportJson);
}

function showAlert(msg) {
  alertMessage.textContent = msg;
  alertBanner.classList.remove("hidden");
}

function hideAlert() {
  alertBanner.classList.add("hidden");
}

async function analyzeComments() {
  hideAlert();
  resultsSection.classList.add("hidden");

  // Check tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url || !tab.url.includes("youtube.com/watch")) {
    showAlert("Erreur : Vous devez être sur une vidéo YouTube (youtube.com/watch) !");
    return;
  }

  analyzeBtn.disabled = true;
  loadingState.classList.remove("hidden");

  const limit = parseInt(commentLimitSelect.value, 10) || 50;

  try {
    let extractRes;
    try {
      extractRes = await chrome.tabs.sendMessage(tab.id, {
        action: "EXTRACT_COMMENTS",
        limit: limit,
        autoScroll: true
      });
    } catch (err) {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
      });
      await new Promise((r) => setTimeout(r, 400));
      extractRes = await chrome.tabs.sendMessage(tab.id, {
        action: "EXTRACT_COMMENTS",
        limit: limit,
        autoScroll: true
      });
    }

    if (!extractRes || !extractRes.success) {
      throw new Error(extractRes?.error || "Échec de l'extraction des commentaires");
    }

    const commentsList = extractRes.comments;
    if (!commentsList || commentsList.length === 0) {
      throw new Error("Aucun commentaire trouvé sur la page.");
    }

    // Call API
    const cleanUrl = appState.apiUrl.replace(/\/+$/, "");
    const texts = commentsList.map((c) => c.text);

    const apiRes = await fetch(`${cleanUrl}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: texts })
    });

    if (!apiRes.ok) {
      throw new Error("Erreur API : HTTP " + apiRes.status);
    }

    const data = await apiRes.json();
    appState.predictions = data.predictions.map((p, i) => ({
      ...p,
      author: commentsList[i]?.author || "Auteur inconnu"
    }));
    appState.statistics = data.statistics;

    renderResults(data.statistics);
    loadingState.classList.add("hidden");
    resultsSection.classList.remove("hidden");

  } catch (err) {
    loadingState.classList.add("hidden");
    showAlert(err.message);
  } finally {
    analyzeBtn.disabled = false;
  }
}

function renderResults(s) {
  predominantText.textContent =
    s.predominant_sentiment === "positive" ? "POSITIF 🟢" :
    s.predominant_sentiment === "negative" ? "NÉGATIF 🔴" : "NEUTRE 🟡";

  sentimentScore.textContent = s.sentiment_score.toFixed(2);
  avgConfidence.textContent = (s.average_confidence * 100).toFixed(1) + "%";

  posCount.textContent = s.positive_count;
  posPct.textContent = s.positive_pct + "%";
  neuCount.textContent = s.neutral_count;
  neuPct.textContent = s.neutral_pct + "%";
  negCount.textContent = s.negative_count;
  negPct.textContent = s.negative_pct + "%";

  barPositive.style.width = s.positive_pct + "%";
  barNeutral.style.width = s.neutral_pct + "%";
  barNegative.style.width = s.negative_pct + "%";

  tabCountAll.textContent = s.total_comments;
  tabCountPos.textContent = s.positive_count;
  tabCountNeu.textContent = s.neutral_count;
  tabCountNeg.textContent = s.negative_count;

  renderFeed();
}

function renderFeed() {
  commentsFeed.innerHTML = "";

  const filtered = appState.predictions.filter((item) => {
    const matchesFilter =
      appState.activeFilter === "all" || item.sentiment === appState.activeFilter;
    const matchesSearch =
      !appState.searchQuery ||
      item.text.toLowerCase().includes(appState.searchQuery) ||
      item.author.toLowerCase().includes(appState.searchQuery);
    return matchesFilter && matchesSearch;
  });

  if (filtered.length === 0) {
    commentsFeed.innerHTML = `<div style="text-align:center; padding:10px; color:#777;">Aucun commentaire trouvé.</div>`;
    return;
  }

  filtered.forEach((c) => {
    const div = document.createElement("div");
    const cls =
      c.sentiment === "positive" ? "pos" :
      c.sentiment === "negative" ? "neg" : "neu";
    const label =
      c.sentiment === "positive" ? "POSITIF" :
      c.sentiment === "negative" ? "NÉGATIF" : "NEUTRE";

    div.className = `comment-item ${cls}`;
    div.innerHTML = `
      <div class="comment-meta">
        <b>${escape(c.author)}</b>
        <span><b>[${label}]</b> ${(c.confidence * 100).toFixed(0)}%</span>
      </div>
      <div class="comment-text">${escape(c.text)}</div>
    `;
    commentsFeed.appendChild(div);
  });
}

function escape(t) {
  if (!t) return "";
  return t.replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function exportCsv() {
  if (!appState.predictions.length) return;
  const headers = ["Auteur", "Commentaire", "Sentiment", "Confiance"];
  const rows = appState.predictions.map((p) => [
    `"${p.author.replace(/"/g, '""')}"`,
    `"${p.text.replace(/"/g, '""')}"`,
    `"${p.sentiment}"`,
    p.confidence
  ]);
  const csv = "\uFEFF" + [headers.join(","), ...rows.map((r) => r.join(","))].join("\r\n");
  download(csv, "youtube_comments_sentiment.csv", "text/csv;charset=utf-8;");
}

function exportJson() {
  if (!appState.predictions.length) return;
  const data = JSON.stringify({
    statistics: appState.statistics,
    comments: appState.predictions
  }, null, 2);
  download(data, "youtube_sentiment.json", "application/json");
}

function download(content, name, type) {
  const blob = new Blob([content], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}
