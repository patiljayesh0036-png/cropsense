/* ===== CropSense - Frontend Logic with 3D Integration ===== */
(() => {
  "use strict";

  const MAX_BYTES = 16 * 1024 * 1024;
  const $ = (id) => document.getElementById(id);

  const form = $("analyze-form");
  const hint = $("hint");
  const analyzeBtn = $("analyze-btn");
  const btnText = $("btn-text");
  const errorBox = $("error-box");
  const thermalCol = $("thermal-col");
  const results = $("results");

  const state = {
    crop: null,
    module: null,
    files: { rgb: null, thermal: null },
  };

  const HINTS = {
    none: "Choose a crop and a type of check above, then add your photo here.",
    "rice|water_stress":
      "Add a clear photo of the rice plants. A thermal image is optional and only has a small influence on the result.",
    "rice|nutrients":
      "Add a close photo of a rice leaf. Long, narrow leaf photos are fine. This check looks for nitrogen, phosphorus and potassium deficiency.",
    "maize|water_stress":
      "Add a close-up photo of a single maize leaf. The model learned from small cropped leaf images. Maize does not need a thermal image.",
    "maize|nutrients":
      "Add a close photo of a maize leaf. This check recognises healthy leaves and nitrogen, phosphorus, potassium or magnesium deficiency.",
    partial: "Choose both a crop and a type of check to continue.",
  };

  const LEVEL_TEXT = {
    good: "Looks good",
    warn: "Action needed",
    alert: "Act now",
  };

  const needsThermal = () =>
    state.crop === "rice" && state.module === "water_stress";
  const ready = () => Boolean(state.crop && state.module && state.files.rgb);

  /* ---------- safe animation helper ---------- */
  const safeGsap = {
    from: (target, vars) => {
      if (typeof gsap !== "undefined" && gsap && typeof gsap.from === "function") {
        try { return gsap.from(target, vars); } catch (e) { /* ignore animation error */ }
      }
    },
    to: (target, vars) => {
      if (typeof gsap !== "undefined" && gsap && typeof gsap.to === "function") {
        try { return gsap.to(target, vars); } catch (e) { /* ignore animation error */ }
      } else if (vars && typeof vars.onComplete === "function") {
        vars.onComplete();
      }
    }
  };

  /* ---------- small helpers ---------- */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    safeGsap.from(errorBox, { opacity: 0, y: -10, duration: 0.3, ease: "power2.out" });
  }
  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  /* ---------- selection ---------- */
  function updateUI() {
    // Sync state with DOM in case radio buttons or inputs were selected
    if (!state.crop) {
      const checkedCrop = form ? form.querySelector('input[name="crop"]:checked') : null;
      if (checkedCrop) state.crop = checkedCrop.value;
    }
    if (!state.module) {
      const checkedModule = form ? form.querySelector('input[name="module"]:checked') : null;
      if (checkedModule) state.module = checkedModule.value;
    }
    if (!state.files.rgb) {
      const rgbInput = $("file-rgb");
      if (rgbInput && rgbInput.files && rgbInput.files[0]) {
        state.files.rgb = rgbInput.files[0];
      }
    }

    if (state.crop && state.module) {
      hint.textContent = HINTS[`${state.crop}|${state.module}`] || HINTS.partial;
    } else if (state.crop || state.module) {
      hint.textContent = HINTS.partial;
    } else {
      hint.textContent = HINTS.none;
    }

    const showThermal = needsThermal();
    if (thermalCol) {
      thermalCol.hidden = !showThermal;
    }
    if (!showThermal && state.files.thermal) clearFile("thermal");

    const isReady = ready();

    const cropCard = $("card-crop");
    if (cropCard) cropCard.classList.toggle("is-done", Boolean(state.crop));

    const moduleCard = $("card-module");
    if (moduleCard) moduleCard.classList.toggle("is-done", Boolean(state.module));

    const uploadCard = $("card-upload");
    if (uploadCard) uploadCard.classList.toggle("is-done", Boolean(state.files.rgb));

    if (analyzeBtn) {
      // Do not keep HTML disabled so user clicks are received and can guide the user
      analyzeBtn.disabled = false;
      analyzeBtn.setAttribute("aria-disabled", String(!isReady));
      analyzeBtn.classList.toggle("is-unready", !isReady);
      analyzeBtn.classList.toggle("is-ready", isReady);
    }

    // Animate step completion
    if (state.crop) animateStepComplete("card-crop");
    if (state.module) animateStepComplete("card-module");
    if (state.files.rgb) animateStepComplete("card-upload");
  }

  function animateStepComplete(cardId) {
    const card = $(cardId);
    if (card && !card.dataset.animated) {
      card.dataset.animated = "true";
      safeGsap.from(card.querySelector(".step-num"), {
        scale: 1.3,
        rotation: 180,
        duration: 0.6,
        ease: "elastic.out(1, 0.5)",
      });
    }
  }

  form.addEventListener("change", (e) => {
    if (e.target.name === "crop") state.crop = e.target.value;
    if (e.target.name === "module") state.module = e.target.value;
    if (e.target.name === "crop" || e.target.name === "module") {
      hideError();
      updateUI();
    }
  });

  // Also support direct click on choice cards
  document.querySelectorAll(".choice").forEach((choice) => {
    choice.addEventListener("click", () => {
      const radio = choice.querySelector('input[type="radio"]');
      if (radio && !radio.checked) {
        radio.checked = true;
        if (radio.name === "crop") state.crop = radio.value;
        if (radio.name === "module") state.module = radio.value;
        hideError();
        updateUI();
      }
    });
  });

  /* ---------- file upload ---------- */
  function setupDropzone(kind) {
    const zone = $(`drop-${kind}`);
    const input = $(`file-${kind}`);
    if (!zone || !input) return;

    zone.addEventListener("click", (e) => {
      if (e.target.closest(".dz-remove")) return;
      input.click();
    });
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      if (input.files && input.files[0]) setFile(kind, input.files[0]);
    });

    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
      }),
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
      }),
    );
    zone.addEventListener("drop", (e) => {
      const file =
        e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) setFile(kind, file);
    });
  }

  function setFile(kind, file) {
    hideError();
    if (!file.type.startsWith("image/")) {
      showError("That file is not an image. Choose a JPG, PNG or WEBP photo.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showError("That image is larger than 16 MB. Choose a smaller photo.");
      return;
    }
    state.files[kind ] = file;

    const zone = $(`drop-${kind}`);
    const img = $(`preview-${kind}`);
    if (img) {
      if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
      const url = URL.createObjectURL(file);
      img.src = url;
      img.dataset.url = url;
    }
    const nameEl = $(`name-${kind}`);
    if (nameEl) nameEl.textContent = file.name;

    if (zone) {
      const emptyEl = zone.querySelector(".dz-empty");
      const previewEl = zone.querySelector(".dz-preview");
      if (emptyEl) emptyEl.hidden = true;
      if (previewEl) previewEl.hidden = false;
    }

    // Update UI state first so everything is marked ready immediately
    updateUI();

    // Optional safe preview entrance animation
    if (zone) {
      const previewEl = zone.querySelector(".dz-preview");
      if (previewEl) {
        safeGsap.from(previewEl, {
          opacity: 0,
          scale: 0.95,
          duration: 0.4,
          ease: "power2.out",
        });
      }
    }
  }

  function clearFile(kind) {
    state.files[kind] = null;
    const zone = $(`drop-${kind}`);
    const img = $(`preview-${kind}`);
    if (img) {
      if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
      img.removeAttribute("src");
      delete img.dataset.url;
    }
    const fileInput = $(`file-${kind}`);
    if (fileInput) fileInput.value = "";

    if (zone) {
      const emptyEl = zone.querySelector(".dz-empty");
      const previewEl = zone.querySelector(".dz-preview");
      if (emptyEl) emptyEl.hidden = false;
      if (previewEl) previewEl.hidden = true;
      zone.dataset.animated = "false";
    }
    updateUI();
  }

  setupDropzone("rgb");
  setupDropzone("thermal");
  document.querySelectorAll(".dz-remove").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile(btn.dataset.kind);
    }),
  );

  analyzeBtn.addEventListener("click", (e) => {
    if (!ready()) {
      e.preventDefault();
      if (!state.crop) {
        showError("Please select a crop type (Rice or Maize) in Step 1.");
        const c = $("card-crop");
        if (c) c.scrollIntoView({ behavior: "smooth", block: "center" });
      } else if (!state.module) {
        showError("Please select a check type (Water Stress or Nutrients) in Step 2.");
        const m = $("card-module");
        if (m) m.scrollIntoView({ behavior: "smooth", block: "center" });
      } else {
        showError("Please upload a standard photo in Step 3 before running analysis.");
        const u = $("card-upload");
        if (u) u.scrollIntoView({ behavior: "smooth", block: "center" });
      }
      return;
    }
  });

  /* ---------- submit ---------- */
  function setLoading(isLoading) {
    analyzeBtn.classList.toggle("is-loading", isLoading);
    analyzeBtn.disabled = isLoading;
    btnText.textContent = isLoading ? "Analysing..." : "Run Analysis";

    if (isLoading) {
      document.body.classList.add("analyzing");
    } else {
      document.body.classList.remove("analyzing");
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!ready()) {
      if (!state.crop) showError("Please select a crop type (Rice or Maize) in Step 1.");
      else if (!state.module) showError("Please select a check type (Water Stress or Nutrients) in Step 2.");
      else showError("Please upload a standard photo in Step 3 before running analysis.");
      return;
    }

    hideError();
    setLoading(true);

    const data = new FormData();
    data.append("crop", state.crop);
    data.append("module", state.module);
    data.append("image", state.files.rgb);
    if (needsThermal() && state.files.thermal)
      data.append("thermal_image", state.files.thermal);

    try {
      const response = await fetch("/predict", { method: "POST", body: data });
      let payload = null;
      try {
        payload = await response.json();
      } catch (_) {
        /* not JSON */
      }
      if (!response.ok || !payload || !payload.ok) {
        throw new Error(
          (payload && payload.error) ||
            "The server returned an error. Try again in a moment.",
        );
      }
      renderResult(payload);
    } catch (err) {
      if (err instanceof TypeError) {
        showError(
          "Could not reach the server. Check that the app is still running in the terminal.",
        );
      } else {
        showError(err.message);
      }
    } finally {
      setLoading(false);
    }
  });

  /* ---------- render result ---------- */
  function renderResult(data) {
    const rec = data.recommendation;
    const level = rec && rec.level ? rec.level : "warn";

    results.classList.remove("level-good", "level-warn", "level-alert");
    results.classList.add(`level-${level}`);

    $("result-badge").textContent = data.low_confidence
      ? "Not sure"
      : LEVEL_TEXT[level] || "Result";
    $("result-lowconf").hidden = !data.low_confidence;
    $("result-title").textContent =
      (data.low_confidence ? "Best guess: " : "") +
      (rec ? rec.title : data.prediction);
    $("result-sub").textContent = rec ? rec.summary : "";
    $("result-confidence").textContent = pct(data.confidence);

    // confidence dial: start at 0, then animate to the value
    const gauge = $("result-gauge");
    const value = Math.round(data.confidence * 100);
    gauge.style.setProperty("--pct", "0");
    gauge.setAttribute(
      "aria-label",
      `Confidence of the result: ${value} percent`,
    );
    results.hidden = false;

    // Animate results entrance
    safeGsap.from(results, {
      opacity: 0,
      y: 30,
      scale: 0.98,
      duration: 0.6,
      ease: "power3.out",
    });

    requestAnimationFrame(() =>
      requestAnimationFrame(() =>
        gauge.style.setProperty("--pct", String(value)),
      ),
    );

    // uploaded images
    const imageBox = $("result-images");
    imageBox.replaceChildren();
    const captions = { image: "Your photo", thermal_image: "Thermal image" };
    Object.entries(data.images || {}).forEach(([key, url], index) => {
      const fig = el("figure");
      const img = el("img");
      img.src = url;
      img.alt = captions[key] || "Uploaded image";
      fig.append(img, el("figcaption", "", captions[key] || ""));
      imageBox.append(fig);

      safeGsap.from(fig, {
        opacity: 0,
        y: 20,
        scale: 0.95,
        duration: 0.5,
        delay: index * 0.1,
        ease: "power2.out",
      });
    });

    // probabilities
    const list = $("prob-list");
    list.replaceChildren();
    data.probabilities.forEach((row, i) => {
      const li = el("li", `prob-row${i === 0 ? " top" : ""}`);
      const bar = el("div", "prob-bar");
      const fill = el("div", "prob-fill");
      bar.append(fill);
      li.append(
        el("span", "name", row.label),
        el("span", "pct", pct(row.prob, 1)),
        bar,
      );
      list.append(li);
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          fill.style.width = pct(row.prob, 1);
        }),
      );

      // Stagger animation
      safeGsap.from(li, {
        opacity: 0,
        x: -20,
        duration: 0.4,
        delay: i * 0.08,
        ease: "power2.out",
      });
    });

    // recommendation
    const recBlock = $("rec-block");
    recBlock.hidden = !rec;
    if (rec) {
      $("rec-title").textContent = "What to do";
      const actions = $("rec-actions");
      actions.replaceChildren();
      rec.actions.forEach((text, i) => {
        const li = el("li", "", text);
        actions.append(li);
        safeGsap.from(li, {
          opacity: 0,
          x: 20,
          duration: 0.4,
          delay: i * 0.1,
          ease: "power2.out",
        });
      });

      safeGsap.from(recBlock, {
        opacity: 0,
        y: 20,
        duration: 0.5,
        delay: 0.3,
        ease: "power2.out",
      });
    }

    // warnings
    const warnBlock = $("warnings-block");
    const warnList = $("warnings");
    warnList.replaceChildren();
    (data.warnings || []).forEach((text) =>
      warnList.append(el("li", "", text)),
    );
    warnBlock.hidden = !(data.warnings && data.warnings.length);

    if (!warnBlock.hidden) {
      safeGsap.from(warnBlock, {
        opacity: 0,
        y: 20,
        duration: 0.5,
        delay: 0.4,
        ease: "power2.out",
      });
    }

    renderDetails(data.details || {});
    $("disclaimer").textContent = data.disclaimer || "";

    const reduce = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    results.scrollIntoView({
      behavior: reduce ? "auto" : "smooth",
      block: "start",
    });
    $("result-title").focus({ preventScroll: true });

    // Trigger 3D scene reaction to result
    if (window.CropSense3D && window.CropSense3D.getHeroScene) {
      triggerResultAnimation(level);
    }
  }

  function triggerResultAnimation(level) {
    const heroLeaf = document.querySelector("#hero-3d-leaf");
    if (heroLeaf) {
      heroLeaf.style.setProperty("--result-level", level);
      heroLeaf.classList.add(`result-${level}`);
      setTimeout(() => heroLeaf.classList.remove(`result-${level}`), 3000);
    }
  }

  function renderDetails(details) {
    const box = $("tech-details");
    const body = $("details-body");
    body.replaceChildren();

    const rows = [
      ["Photo: deep network", details.rgb_cnn],
      ["Photo: colour model", details.rgb_colour_model],
      ["Photo: combined", details.rgb_combined],
      ["Thermal model", details.thermal],
    ].filter(([, probs]) => probs);

    if (rows.length) {
      const classes = Object.keys(rows[0][1]);
      const table = el("table", "tech-table");
      const head = el("tr");
      head.append(el("th", "", "Model"));
      classes.forEach((c) => head.append(el("th", "", c)));
      table.append(head);
      rows.forEach(([name, probs]) => {
        const tr = el("tr");
        tr.append(el("td", "", name));
        classes.forEach((c) => tr.append(el("td", "", pct(probs[c] ?? 0, 1))));
        table.append(tr);
      });
      body.append(table);
    }
    if (details.rgb_cnn_weight !== undefined) {
      body.append(
        el(
          "p",
          "tech-note",
          `The deep network and the colour model are averaged (deep network share ${pct(details.rgb_cnn_weight)}).`,
        ),
      );
    }
    if (details.fusion_weights) {
      body.append(
        el(
          "p",
          "tech-note",
          `Final answer: photo ${pct(details.fusion_weights.rgb)} and thermal ${pct(details.fusion_weights.thermal)}.`,
        ),
      );
    }
    if (details.tiles_used !== undefined) {
      body.append(
        el(
          "p",
          "tech-note",
          `Your photo was analysed in ${details.tiles_used} piece${details.tiles_used === 1 ? "" : "s"} and the results were averaged.`,
        ),
      );
    }
    box.hidden = body.children.length === 0;
    box.open = false;
  }

  /* ---------- reset and print ---------- */
  $("reset-btn").addEventListener("click", () => {
    safeGsap.to(results, {
      opacity: 0,
      y: 20,
      scale: 0.98,
      duration: 0.3,
      ease: "power2.in",
      onComplete: () => {
        results.hidden = true;
        clearFile("rgb");
        clearFile("thermal");
        hideError();
        document.querySelectorAll(".step").forEach((step) => {
          step.dataset.animated = "false";
        });
        $("analyse").scrollIntoView({ behavior: "smooth", block: "start" });
      },
    });
  });
  $("print-btn").addEventListener("click", () => window.print());

  /* ---------- Scroll Effects ---------- */
  function initScrollEffects() {
    let lastScroll = 0;
    window.addEventListener(
      "scroll",
      () => {
        const currentScroll = window.scrollY;
        const topbar = document.querySelector(".topbar");
        if (topbar) {
          if (currentScroll > 50) {
            topbar.classList.add("scrolled");
          } else {
            topbar.classList.remove("scrolled");
          }
        }
        lastScroll = currentScroll;
      },
      { passive: true },
    );

    const heroArt = document.querySelector(".hero-art");
    if (heroArt) {
      window.addEventListener(
        "scroll",
        () => {
          const scrolled = window.scrollY;
          const rate = scrolled * -0.3;
          heroArt.style.transform = `translateY(${rate}px)`;
        },
        { passive: true },
      );
    }
  }

  /* ---------- Intersection Observer for Animations ---------- */
  function initScrollAnimations() {
    if (typeof IntersectionObserver === "undefined") return;
    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px",
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("animate-in");
          observer.unobserve(entry.target);
        }
      });
    }, observerOptions);

    document.querySelectorAll(".step-3d, .result-block, .table-wrap").forEach((el) => {
      observer.observe(el);
    });
  }

  /* ---------- Initialize ---------- */
  function init() {
    updateUI();
    initScrollEffects();
    initScrollAnimations();

    safeGsap.from(".step", {
      opacity: 0,
      y: 30,
      duration: 0.6,
      stagger: 0.1,
      ease: "power3.out",
      delay: 0.3,
    });

    safeGsap.from(".tips", {
      opacity: 0,
      x: 50,
      duration: 0.8,
      ease: "power3.out",
      delay: 0.5,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();