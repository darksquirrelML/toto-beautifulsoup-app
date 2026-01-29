const CACHE_NAME = "streamlit-pwa-cache-v1";
const urlsToCache = [
  "/",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/icon-180.png",
];

// Install event
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(urlsToCache))
  );
});

// Fetch event
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
