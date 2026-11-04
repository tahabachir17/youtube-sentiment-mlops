/**
 * YouTube Sentiment Analyzer - Content Script
 * Extracts comments from YouTube watch pages dynamically, handles infinite scroll,
 * and passes comment payloads to the extension popup.
 */

console.log("[YT Sentiment] Content script injected and active.");

// Sleep utility
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Extracts currently rendered comments from YouTube DOM.
 * @param {boolean} includeReplies - Whether to extract comments inside reply threads
 * @returns {Array<{text: string, author: string, likes: string, time: string}>}
 */
function extractRenderedComments(includeReplies = true) {
  const commentElements = document.querySelectorAll(
    includeReplies ? "ytd-comment-view-model, ytd-comment-renderer" : "ytd-comment-thread-renderer #comment"
  );

  const results = [];
  const seenTexts = new Set();

  commentElements.forEach((el) => {
    // YouTube uses #content-text or span.yt-core-attributed-string for comment body
    const textEl = el.querySelector("#content-text, .yt-core-attributed-string--link-inherit-color, yt-formatted-string#content-text");
    if (!textEl) return;

    const rawText = textEl.innerText ? textEl.innerText.trim() : textEl.textContent.trim();
    if (!rawText || rawText.length < 2 || seenTexts.has(rawText)) {
      return;
    }
    seenTexts.add(rawText);

    // Extract author if available
    const authorEl = el.querySelector("#author-text span, #author-text, .ytd-channel-name");
    const author = authorEl ? authorEl.innerText.trim() : "Utilisateur YouTube";

    // Extract likes count
    const likesEl = el.querySelector("#vote-count-middle, #vote-count-left");
    const likes = likesEl ? likesEl.innerText.trim() : "0";

    // Extract published time
    const timeEl = el.querySelector(".published-time-text a, #published-time-text");
    const time = timeEl ? timeEl.innerText.trim() : "";

    results.push({
      text: rawText,
      author: author.replace(/^@/, ""),
      likes: likes || "0",
      time: time
    });
  });

  return results;
}

/**
 * Scrolls smoothly to force YouTube to trigger comment loading.
 * @param {number} targetCount - Desired number of comments
 * @param {number} maxScrollAttempts - Max scroll steps before stopping
 * @returns {Promise<Array>}
 */
async function scrollAndCollectComments(targetCount = 50, maxScrollAttempts = 6) {
  let comments = extractRenderedComments();
  let attempts = 0;

  // First scroll towards comments section if not visible
  const commentsSection = document.querySelector("#comments");
  if (commentsSection) {
    commentsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    await sleep(600);
  } else {
    window.scrollBy({ top: 700, behavior: "smooth" });
    await sleep(600);
  }

  while (comments.length < targetCount && attempts < maxScrollAttempts) {
    attempts++;
    window.scrollBy({ top: 900, behavior: "smooth" });
    await sleep(700);
    comments = extractRenderedComments();
    if (comments.length >= targetCount) break;
  }

  return comments.slice(0, targetCount);
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "EXTRACT_COMMENTS") {
    const isWatchPage = window.location.pathname.startsWith("/watch");
    if (!isWatchPage) {
      sendResponse({
        success: false,
        error: "Veuillez ouvrir une page de vidéo YouTube (youtube.com/watch?v=...)."
      });
      return true;
    }

    const limit = request.limit || 50;
    const autoScroll = request.autoScroll !== false;

    if (autoScroll) {
      scrollAndCollectComments(limit)
        .then((comments) => {
          if (comments.length === 0) {
            sendResponse({
              success: false,
              error: "Aucun commentaire trouvé. Assurez-vous que les commentaires sont activés pour cette vidéo."
            });
          } else {
            sendResponse({
              success: true,
              count: comments.length,
              comments: comments
            });
          }
        })
        .catch((err) => {
          sendResponse({
            success: false,
            error: "Erreur lors de l'extraction: " + err.message
          });
        });
      return true; // Keep message channel open for async response
    } else {
      const comments = extractRenderedComments().slice(0, limit);
      sendResponse({
        success: true,
        count: comments.length,
        comments: comments
      });
      return false;
    }
  }

  if (request.action === "CHECK_PAGE_STATUS") {
    const isWatch = window.location.pathname.startsWith("/watch");
    const videoTitle = document.querySelector("h1.ytd-watch-metadata yt-formatted-string, h1.title yt-formatted-string")?.innerText || document.title;
    sendResponse({
      isWatchPage: isWatch,
      title: videoTitle
    });
    return false;
  }
});
