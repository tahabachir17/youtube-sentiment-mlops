/**
 * YouTube Sentiment Analyzer - Popup Controller
 * Manages UI interactions, YouTube DOM extraction triggers, API communication,
 * metric visualizations, filtering, dark mode, and file exports.
 */

// State
let appState = {
  apiUrl: "http://localhost:7860",
  theme: "dark",
  predictions: [],
  statistics: null,
  activeFilter: "all",
  searchQuery: "",
  isAnalyzing: false
};

// DOM Elements
const themeToggleBtn = document.getElementById("themeToggleBtn");
const themeIcon = document.getElementById("themeIcon");
const settingsToggleBtn = document.getElementById("settingsToggleBtn");
const settingsPanel = document.getElementById("settingsPanel");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const apiUrlInput = document.getElementById("apiUrlInput");
const testApiBtn = document.getElementById("testApiBtn");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const apiTestStatus = document.getElementById("apiTestStatus");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const latencyBadge = document.getElementById("latencyBadge");

const commentLimitSelect = document.getElementById("commentLimitSelect");
const analyzeBtn = document.getElementById("analyzeBtn");
const alertBanner = document.getElementById("alertBanner");
const alertMessage = document.getElementById("alertMessage");
const dismissAlertBtn = document.getElementById("dismissAlertBtn");

const loadingState = document.getElementById("loadingState");
const loadingText = document.getElementById("loadingText");
const resultsSection = document.getElementById("resultsSection");

// Hero Elements
const predominantBadge = document.getElementById("predominantBadge");
const sentimentScore = document.getElementById("sentimentScore");
const avgConfidence = document.getElementById("avgConfidence");
const analyzedCountSub = document.getElementById("analyzedCountSub");

// Distribution Bar
const barPositive = document.getElementById("barPositive");
const barNeutral = document.getElementById("barNeutral");
const barNegative = document.getElementById("barNegative");

// Metric counters
const posCount = document.getElementById("posCount");
const posPct = document.getElementById("posPct");
const neuCount = document.getElementById("neuCount");
const neuPct = document.getElementById("neuPct");
const negCount = document.getElementById("negCount");
const negPct = document.getElementById("negPct");

// Feed Controls
const tabCountAll = document.getElementById("tabCountAll");
const tabCountPos = document.getElementById("tabCountPos");
const tabCountNeu = document.getElementById("tabCountNeu");
const tabCountNeg = document.getElementById("tabCountNeg");
const filterTabs = document.querySelectorAll(".filter-tab");
const searchInput = document.getElementById("searchInput");
const commentsFeed = document.getElementById("commentsFeed");

// Actions
const copySummaryBtn = document.getElementById("copySummaryBtn");
const exportCsvBtn = document.getElementById("exportCsvBtn");
const exportJsonBtn = document.getElementById("exportJsonBtn");

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  await loadStoredSettings();
  applyTheme(appState.theme);
  await checkApiHealth();
  setupEventListeners();
});

async function loadStoredSettings() {
  if (chrome?.storage?.local) {
    const data = await chrome.storage.local.get(["apiUrl", "theme"]);
    if (data.apiUrl) appState.apiUrl = data.apiUrl;
    if (data.theme) appState.theme = data.theme;
  } else {
    const savedUrl = localStorage.getItem("yt_sentiment_api_url");
    const savedTheme = localStorage.getItem("yt_sentiment_theme");
    if (savedUrl) appState.apiUrl = savedUrl;
    if (savedTheme) appState.theme = savedTheme;
  }
  apiUrlInput.value = appState.apiUrl;
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
  appState.theme = theme;
  if (chrome?.storage?.local) {
    chrome.storage.local.set({ theme });
  } else {
    localStorage.setItem("yt_sentiment_theme", theme);
  }
}

// -------------------------------------------------------------
// API Health Check
// -------------------------------------------------------------
async function checkApiHealth() {
  statusDot.className = "status-dot checking";
  statusText.textContent = "Test connexion API...";
  latencyBadge.classList.add("hidden");

  const t0 = performance.now();
  try {
    const cleanUrl = appState.apiUrl.replace(/\/+$/, "");
    const response = await fetch(`${cleanUrl}/health`, {
      method: "GET",
      headers: { "Accept": "application/json" }
    });

    const elapsed = Math.round(performance.now() - t0);
    if (response.ok) {
      const data = await response.json();
      statusDot.className = "status-dot online";
      statusText.textContent = "API en ligne (" + (data.model_loaded ? "Modèle prêt" : "Non chargé") + ")";
      latencyBadge.textContent = `${elapsed} ms`;
      latencyBadge.classList.remove("hidden");
      return true;
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (err) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API hors ligne (Localhost:7860)";
    latencyBadge.classList.add("hidden");
    return false;
  }
}

// -------------------------------------------------------------
// Event Listeners
// -------------------------------------------------------------
function setupEventListeners() {
  // Theme Toggle
  themeToggleBtn.addEventListener("click", () => {
    applyTheme(appState.theme === "dark" ? "light" : "dark");
  });

  // Settings Drawer Toggle
  settingsToggleBtn.addEventListener("click", () => {
    settingsPanel.classList.toggle("hidden");
  });
  closeSettingsBtn.addEventListener("click", () => {
    settingsPanel.classList.add("hidden");
  });

  // Presets
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      apiUrlInput.value = btn.dataset.url;
    });
  });

  // Test API in settings
  testApiBtn.addEventListener("click", async () => {
    const url = apiUrlInput.value.trim().replace(/\/+$/, "");
    apiTestStatus.textContent = "Test en cours...";
    apiTestStatus.style.color = "var(--neu-color)";
    try {
      const resp = await fetch(`${url}/health`);
      if (resp.ok) {
        apiTestStatus.textContent = "Connexion réussie ! (HTTP 200 OK)";
        apiTestStatus.style.color = "var(--pos-color)";
      } else {
        apiTestStatus.textContent = `Erreur HTTP ${resp.status}`;
        apiTestStatus.style.color = "var(--neg-color)";
      }
    } catch (e) {
      apiTestStatus.textContent = `Échec de connexion : ${e.message}`;
      apiTestStatus.style.color = "var(--neg-color)";
    }
  });

  // Save Settings
  saveSettingsBtn.addEventListener("click", async () => {
    appState.apiUrl = apiUrlInput.value.trim().replace(/\/+$/, "");
    if (chrome?.storage?.local) {
      await chrome.storage.local.set({ apiUrl: appState.apiUrl });
    } else {
      localStorage.setItem("yt_sentiment_api_url", appState.apiUrl);
    }
    settingsPanel.classList.add("hidden");
    await checkApiHealth();
  });

  // Dismiss alert
  dismissAlertBtn.addEventListener("click", () => {
    alertBanner.classList.add("hidden");
  });

  // Analyze Button
  analyzeBtn.addEventListener("click", startExtractionAndAnalysis);

  // Filter Tabs
  filterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      filterTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      appState.activeFilter = tab.dataset.filter;
      renderCommentsFeed();
    });
  });

  // Search input
  searchInput.addEventListener("input", (e) => {
    appState.searchQuery = e.target.value.toLowerCase().trim();
    renderCommentsFeed();
  });

  // Copy Summary
  copySummaryBtn.addEventListener("click", copySummaryToClipboard);

  // Export CSV
  exportCsvBtn.addEventListener("click", exportAsCsv);

  // Export JSON
  exportJsonBtn.addEventListener("click", exportAsJson);
}

function showAlert(message) {
  alertMessage.textContent = message;
  alertBanner.classList.remove("hidden");
}

function hideAlert() {
  alertBanner.classList.add("hidden");
}

// -------------------------------------------------------------
// Extraction & Analysis Workflow
// -------------------------------------------------------------
async function startExtractionAndAnalysis() {
  if (appState.isAnalyzing) return;
  hideAlert();

  // 1. Check active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url || !tab.url.includes("youtube.com/watch")) {
    showAlert("Veuillez vous positionner sur une page de vidéo YouTube (youtube.com/watch?v=...) pour lancer l'analyse.");
    return;
  }

  appState.isAnalyzing = true;
  analyzeBtn.disabled = true;
  resultsSection.classList.add("hidden");
  loadingState.classList.remove("hidden");
  loadingText.textContent = "Extraction des commentaires YouTube...";

  const limit = parseInt(commentLimitSelect.value, 10) || 50;

  try {
    // 2. Ensure content script is injected
    let extractResult;
    try {
      extractResult = await chrome.tabs.sendMessage(tab.id, {
        action: "EXTRACT_COMMENTS",
        limit: limit,
        autoScroll: true
      });
    } catch (injectError) {
      // Content script may not be loaded yet; inject it programmatically
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
      });
      // Short delay for script listener setup
      await new Promise((r) => setTimeout(r, 400));
      extractResult = await chrome.tabs.sendMessage(tab.id, {
        action: "EXTRACT_COMMENTS",
        limit: limit,
        autoScroll: true
      });
    }

    if (!extractResult || !extractResult.success) {
      throw new Error(extractResult?.error || "Impossible d'extraire les commentaires. Vérifiez qu'ils ne sont pas désactivés.");
    }

    const rawCommentsData = extractResult.comments;
    if (!rawCommentsData || rawCommentsData.length === 0) {
      throw new Error("Aucun commentaire extrait. Essayez de faire défiler manuellement la page YouTube vers le bas.");
    }

    // 3. Send comments to FastAPI
    loadingText.textContent = `Inférence ML (${rawCommentsData.length} commentaires)...`;

    const cleanApiUrl = appState.apiUrl.replace(/\/+$/, "");
    const commentTexts = rawCommentsData.map((c) => c.text);

    const apiStartTime = performance.now();
    const response = await fetch(`${cleanApiUrl}/predict_batch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify({ comments: commentTexts })
    });

    if (!response.ok) {
      const errorJson = await response.json().catch(() => ({}));
      throw new Error(errorJson.detail || `L'API a retourné l'erreur HTTP ${response.status}`);
    }

    const resultData = await response.json();
    const apiDuration = Math.round(performance.now() - apiStartTime);
    latencyBadge.textContent = `${apiDuration} ms`;
    latencyBadge.classList.remove("hidden");

    // 4. Merge YouTube metadata (author, likes, time) with API predictions
    appState.predictions = resultData.predictions.map((pred, idx) => ({
      ...pred,
      author: rawCommentsData[idx]?.author || "Utilisateur YouTube",
      likes: rawCommentsData[idx]?.likes || "0",
      time: rawCommentsData[idx]?.time || ""
    }));

    appState.statistics = resultData.statistics;

    // 5. Render results
    renderResults(resultData.statistics, appState.predictions);
    loadingState.classList.add("hidden");
    resultsSection.classList.remove("hidden");

  } catch (err) {
    loadingState.classList.add("hidden");
    showAlert(err.message);
  } finally {
    appState.isAnalyzing = false;
    analyzeBtn.disabled = false;
  }
}

// -------------------------------------------------------------
// Render Results & Feed
// -------------------------------------------------------------
function renderResults(stats, predictions) {
  // Hero Badge
  predominantBadge.textContent =
    stats.predominant_sentiment === "positive" ? "Positif" :
    stats.predominant_sentiment === "negative" ? "Négatif" : "Neutre";
  predominantBadge.className = `sentiment-pill ${stats.predominant_sentiment}`;

  // Sentiment Score (-1.00 to +1.00)
  const sign = stats.sentiment_score > 0 ? "+" : "";
  sentimentScore.textContent = `${sign}${stats.sentiment_score.toFixed(2)}`;
  sentimentScore.style.color =
    stats.sentiment_score > 0.15 ? "var(--pos-color)" :
    stats.sentiment_score < -0.15 ? "var(--neg-color)" : "var(--neu-color)";

  // Confidence & Count
  avgConfidence.textContent = `${(stats.average_confidence * 100).toFixed(1)}%`;
  analyzedCountSub.textContent = `${stats.total_comments} analysés`;

  // Segmented Bar
  barPositive.style.width = `${stats.positive_pct}%`;
  barNeutral.style.width = `${stats.neutral_pct}%`;
  barNegative.style.width = `${stats.negative_pct}%`;

  // Metric counts
  posCount.textContent = stats.positive_count;
  posPct.textContent = `${stats.positive_pct}% Positifs`;

  neuCount.textContent = stats.neutral_count;
  neuPct.textContent = `${stats.neutral_pct}% Neutres`;

  negCount.textContent = stats.negative_count;
  negPct.textContent = `${stats.negative_pct}% Négatifs`;

  // Filter Tab Badges
  tabCountAll.textContent = stats.total_comments;
  tabCountPos.textContent = stats.positive_count;
  tabCountNeu.textContent = stats.neutral_count;
  tabCountNeg.textContent = stats.negative_count;

  // Render comments
  renderCommentsFeed();
}

function renderCommentsFeed() {
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
    commentsFeed.innerHTML = `
      <div class="empty-feed-msg">
        Aucun commentaire correspondant aux critères actuels.
      </div>
    `;
    return;
  }

  filtered.forEach((comment) => {
    const card = document.createElement("div");
    card.className = `comment-card ${comment.sentiment}`;

    const sentimentLabel =
      comment.sentiment === "positive" ? "Positif" :
      comment.sentiment === "negative" ? "Négatif" : "Neutre";

    const confPct = (comment.confidence * 100).toFixed(1);

    card.innerHTML = `
      <div class="comment-card-header">
        <span class="comment-author">${escapeHtml(comment.author)}</span>
        <div class="comment-badge-wrapper">
          <span class="comment-badge ${comment.sentiment}">${sentimentLabel}</span>
          <span class="confidence-pill">${confPct}%</span>
        </div>
      </div>
      <div class="comment-body">${escapeHtml(comment.text)}</div>
    `;
    commentsFeed.appendChild(card);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// -------------------------------------------------------------
// Copy & Export Actions
// -------------------------------------------------------------
function copySummaryToClipboard() {
  if (!appState.statistics) return;
  const s = appState.statistics;

  const text = [
    `📊 ANALYSE DE SENTIMENT YOUTUBE`,
    `Total analysé : ${s.total_comments} commentaires`,
    `Sentiment général : ${s.predominant_sentiment.toUpperCase()} (Score: ${s.sentiment_score > 0 ? "+" : ""}${s.sentiment_score.toFixed(2)})`,
    `Confiance moyenne : ${(s.average_confidence * 100).toFixed(1)}%`,
    ``,
    `🟢 Positifs : ${s.positive_count} (${s.positive_pct}%)`,
    `🟡 Neutres  : ${s.neutral_count} (${s.neutral_pct}%)`,
    `🔴 Négatifs : ${s.negative_count} (${s.negative_pct}%)`,
    ``,
    `Généré par YouTube Sentiment Analyzer MLOps`
  ].join("\n");

  navigator.clipboard.writeText(text).then(() => {
    const originalText = copySummaryBtn.textContent;
    copySummaryBtn.textContent = "✅ Copié !";
    setTimeout(() => {
      copySummaryBtn.textContent = originalText;
    }, 1800);
  });
}

function exportAsCsv() {
  if (!appState.predictions || appState.predictions.length === 0) return;

  const headers = ["Auteur", "Commentaire", "Sentiment", "Label", "Confiance", "Prob_Negatif", "Prob_Neutre", "Prob_Positif"];
  const rows = appState.predictions.map((p) => [
    `"${(p.author || '').replace(/"/g, '""')}"`,
    `"${(p.text || '').replace(/"/g, '""')}"`,
    `"${p.sentiment}"`,
    p.label,
    p.confidence,
    p.probabilities?.negative || 0,
    p.probabilities?.neutral || 0,
    p.probabilities?.positive || 0
  ]);

  const csvContent = "\uFEFF" + [headers.join(","), ...rows.map(r => r.join(","))].join("\r\n");
  downloadFile(csvContent, "youtube_sentiment_comments.csv", "text/csv;charset=utf-8;");
}

function exportAsJson() {
  if (!appState.predictions || appState.predictions.length === 0) return;

  const exportPayload = {
    generated_at: new Date().toISOString(),
    statistics: appState.statistics,
    comments: appState.predictions
  };

  const jsonContent = JSON.stringify(exportPayload, null, 2);
  downloadFile(jsonContent, "youtube_sentiment_analysis.json", "application/json");
}

function downloadFile(content, fileName, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
