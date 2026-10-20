"use strict";

(() => {
  const data = JSON.parse(document.querySelector("#gallery-data").textContent);
  const byId = (id) => document.getElementById(id);
  const observed = byId("observation");
  const truth = byId("truth");
  const prediction = byId("prediction");
  const videos = [observed, truth, prediction];
  const intentions = [byId("truth-intent"), byId("prediction-intent")];
  const timeoutMs = 12000;
  let selected = 0;
  let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  let controller;

  function assertActive(signal) {
    if (signal.aborted)
      throw new DOMException("Playback changed", "AbortError");
  }

  function mediaError(video) {
    return new Error(
      `The ${video.id} video could not be loaded. Press Play to retry.`,
    );
  }

  function waitReady(video, signal) {
    return new Promise((resolve, reject) => {
      const events = [
        "loadeddata",
        "canplay",
        "canplaythrough",
        "progress",
        "seeked",
        "error",
      ];
      let timer;
      const cleanup = () => {
        clearTimeout(timer);
        events.forEach((name) => video.removeEventListener(name, check));
        signal.removeEventListener("abort", aborted);
      };
      const finish = (error) => {
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const aborted = () =>
        finish(new DOMException("Playback changed", "AbortError"));
      const check = () => {
        if (signal.aborted) return aborted();
        if (video.error) return finish(mediaError(video));
        if (video.readyState >= 3 && !video.seeking) finish();
      };
      events.forEach((name) => video.addEventListener(name, check));
      signal.addEventListener("abort", aborted, { once: true });
      timer = setTimeout(
        () =>
          finish(new Error("Video loading timed out. Press Play to retry.")),
        timeoutMs,
      );
      check();
    });
  }

  function delay(milliseconds, signal) {
    return new Promise((resolve, reject) => {
      const aborted = () => {
        clearTimeout(timer);
        reject(new DOMException("Playback changed", "AbortError"));
      };
      const timer = setTimeout(() => {
        signal.removeEventListener("abort", aborted);
        resolve();
      }, milliseconds);
      if (signal.aborted) return aborted();
      signal.addEventListener("abort", aborted, { once: true });
    });
  }

  function playGroup(group, signal) {
    return new Promise((resolve, reject) => {
      const ended = new Set();
      const listeners = [];
      const positions = new Map(
        group.map((video) => [video, video.currentTime]),
      );
      let lastProgress = performance.now();
      let settled = false;
      let starting = false;
      let needsSync = false;
      let attempt = 0;
      let watchdog;

      const listen = (video, name, callback) => {
        video.addEventListener(name, callback);
        listeners.push([video, name, callback]);
      };
      const finish = (error) => {
        if (settled) return;
        settled = true;
        attempt += 1;
        clearInterval(watchdog);
        listeners.forEach(([video, name, callback]) =>
          video.removeEventListener(name, callback),
        );
        signal.removeEventListener("abort", aborted);
        if (error) group.forEach((video) => video.pause());
        if (error) reject(error);
        else resolve();
      };
      const aborted = () =>
        finish(new DOMException("Playback changed", "AbortError"));
      const activeVideos = () => group.filter((video) => !ended.has(video));
      const resume = () => {
        if (settled || starting || signal.aborted) return;
        const active = activeVideos();
        const failed = active.find((video) => video.error);
        if (failed) return finish(mediaError(failed));
        if (!active.every((video) => video.readyState >= 3 && !video.seeking))
          return;
        if (needsSync && active.length > 1) {
          const position = Math.min(
            ...active.map((video) => video.currentTime),
          );
          needsSync = false;
          for (const video of active) {
            if (video.currentTime - position > 0.04)
              video.currentTime = position;
          }
          if (active.some((video) => video.seeking)) return;
        }
        needsSync = false;
        if (active.every((video) => !video.paused)) return;
        starting = true;
        const currentAttempt = ++attempt;
        Promise.all(active.map((video) => video.play()))
          .then(() => {
            if (settled || signal.aborted || currentAttempt !== attempt) return;
            starting = false;
          })
          .catch((error) => {
            if (settled || signal.aborted || currentAttempt !== attempt) return;
            starting = false;
            finish(error);
          });
      };
      const buffer = () => {
        if (settled || signal.aborted) return;
        attempt += 1;
        starting = false;
        needsSync = true;
        activeVideos().forEach((video) => video.pause());
        queueMicrotask(resume);
      };
      for (const video of group) {
        listen(video, "ended", () => {
          if (signal.aborted || !video.ended) return;
          ended.add(video);
          lastProgress = performance.now();
          if (ended.size === group.length) finish();
        });
        listen(video, "error", () => finish(mediaError(video)));
        listen(video, "waiting", buffer);
        listen(video, "seeking", buffer);
        listen(video, "stalled", () => {
          if (video.readyState < 3) buffer();
        });
        for (const name of [
          "canplay",
          "canplaythrough",
          "progress",
          "seeked",
        ]) {
          listen(video, name, resume);
        }
      }
      signal.addEventListener("abort", aborted, { once: true });
      watchdog = setInterval(() => {
        if (signal.aborted) return aborted();
        for (const video of activeVideos()) {
          if (video.error) return finish(mediaError(video));
          if (Math.abs(video.currentTime - positions.get(video)) > 0.001) {
            positions.set(video, video.currentTime);
            lastProgress = performance.now();
          }
        }
        if (performance.now() - lastProgress > timeoutMs) {
          finish(new Error("Video playback stalled. Press Play to retry."));
        }
      }, 250);
      if (signal.aborted) aborted();
      else resume();
    });
  }

  function showIntent(element, value) {
    element.replaceChildren();
    for (const side of ["left", "right"]) {
      const hand = value?.[side];
      const row = document.createElement("p");
      const label = document.createElement("span");
      label.className = `hand-${side}`;
      label.textContent = side === "left" ? "L:" : "R:";
      const text = document.createElement("span");
      text.textContent = hand?.action
        ? [
            hand.action,
            hand.object,
            hand.part,
            hand.deep_action,
            hand.target,
            hand.point ? `[${hand.point.join(", ")}]` : null,
          ]
            .filter(Boolean)
            .join(", ")
        : "No intention";
      row.append(label, text);
      element.append(row);
    }
  }

  async function playCycle(index, signal) {
    await Promise.all(videos.map((video) => waitReady(video, signal)));
    assertActive(signal);
    for (const video of videos) {
      video.currentTime = 0;
      video.playbackRate = 0.65;
    }
    await Promise.all(videos.map((video) => waitReady(video, signal)));
    assertActive(signal);
    byId("status").textContent = "Observation";
    await playGroup([observed], signal);
    assertActive(signal);
    if (data.samples[index].group === "failure") {
      byId("status").textContent = "Structured intentions";
      intentions.forEach((element) => element.classList.add("emphasis"));
      await delay(1100, signal);
      assertActive(signal);
      intentions.forEach((element) => element.classList.remove("emphasis"));
    }
    await Promise.all(
      [truth, prediction].map((video) => waitReady(video, signal)),
    );
    assertActive(signal);
    byId("status").textContent = "Forecast";
    await playGroup([truth, prediction], signal);
    assertActive(signal);
    byId("status").textContent = "Completed";
    await delay(900, signal);
    assertActive(signal);
    select((index + 1) % data.samples.length);
  }

  function select(index) {
    controller?.abort();
    videos.forEach((video) => video.pause());
    intentions.forEach((element) => element.classList.remove("emphasis"));
    controller = new AbortController();
    selected = index;
    const sample = data.samples[index];
    byId("phase").textContent = sample.phase;
    byId("draw").textContent = `Draw ${sample.draw_number}`;
    byId("sample-id").textContent = sample.sample_id;
    byId("metadata").href = sample.metadata;
    byId("reference").hidden = !sample.website_reference;
    byId("reference").href = sample.website_reference || "#";
    showIntent(intentions[0], sample.ground_truth_intention);
    showIntent(intentions[1], sample.intention);
    [sample.observation, sample.ground_truth, sample.prediction].forEach(
      (source, position) => {
        videos[position].src = source;
        videos[position].load();
      },
    );
    [...byId("samples").children].forEach((button, position) => {
      button.setAttribute("aria-current", String(position === index));
    });
    byId("toggle").textContent = playing ? "Pause" : "Play";
    byId("status").textContent = playing ? "Loading example…" : "Paused";
    const signal = controller.signal;
    if (playing)
      playCycle(index, signal).catch((error) => {
        if (signal.aborted || error.name === "AbortError") return;
        playing = false;
        controller.abort();
        videos.forEach((video) => video.pause());
        intentions.forEach((element) => element.classList.remove("emphasis"));
        byId("toggle").textContent = "Play";
        byId("status").textContent =
          error.name === "NotAllowedError"
            ? "Press Play to begin."
            : error.message;
      });
  }

  data.samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.title = sample.sample_id;
    button.setAttribute("aria-label", `Example ${index + 1}: ${sample.phase}`);
    button.addEventListener("click", () => select(index));
    byId("samples").append(button);
  });
  byId("toggle").addEventListener("click", () => {
    playing = !playing;
    select(selected);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && playing) {
      playing = false;
      select(selected);
    }
  });
  if (data.samples.length) select(0);
  else {
    byId("toggle").disabled = true;
    byId("status").textContent = "No examples are available.";
  }
})();
